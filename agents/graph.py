"""
Multi-agent graph for the customer support assistant.

Architecture (supervisor pattern):

    user question
         |
         v
    +--------------+
    |  supervisor  |--> classifies intent (sql / rag / hybrid / chitchat)
    +--------------+
         |
         +--> sql_agent       (structured data: customers, tickets, orders)
         +--> rag_agent       (policy PDFs via vector search)
         +--> hybrid_agent    (runs both in parallel)
         +--> chitchat_agent  (greetings / meta)
         |
         v
      synthesis  --> final answer to the user

Why a supervisor instead of one big agent with all tools? Smaller tool sets per
agent give the model fewer wrong choices to make, simpler prompts, and cleaner
failure modes. The synthesis step lets us apply policy to facts in hybrid
questions instead of just dumping both findings.
"""

from typing import Annotated, Literal, TypedDict
from operator import add

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from config import LLM_MODEL, OPENAI_API_KEY
from agents.tools import SQL_TOOLS, RAG_TOOLS


# ------------------------------- State -------------------------------

class AgentState(TypedDict):
    """Shared state passed between every node in the graph."""
    messages: Annotated[list[BaseMessage], add]
    route: str           # 'sql' | 'rag' | 'hybrid' | 'chitchat'
    sql_result: str      # findings from the SQL agent
    rag_result: str      # findings from the RAG agent
    user_query: str      # original user question


# ------------------------------- LLM -------------------------------

def _llm(temperature: float = 0.0) -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set. Add it to your .env file.")
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature, api_key=OPENAI_API_KEY)


# ------------------------------- Supervisor / Router Node -------------------------------

SUPERVISOR_SYSTEM = """You are the routing supervisor for a customer support AI.
Classify the user's question into exactly ONE of these categories:

- "sql"      : Question about specific customers, tickets, orders, or any
               data in the customer support database. Examples:
                 • "Tell me about customer Ema"
                 • "How many open tickets are there?"
                 • "What were John's last 3 orders?"

- "rag"      : Question about company policies / procedures / rules i.e. anything
               answered by reading a policy document. Examples:
                 • "What's the refund policy?"
                 • "How long does shipping take to Europe?"
                 • "What are GDPR rights for users?"

- "hybrid"   : Question that needs BOTH structured customer data AND policy
               information. Examples:
                 • "Is customer Ema eligible for a refund based on our policy?"
                 • "Given John's order date, does shipping policy guarantee delivery?"

- "chitchat" : Greetings, thanks, meta questions about the assistant, or
               anything that does not need a database or document lookup.

Respond with ONLY the category name - one word: sql, rag, hybrid, or chitchat.
Nothing else.
"""

def supervisor_node(state: AgentState) -> dict:
    """Classify the user's message and set the routing decision."""
    user_msg = state["messages"][-1].content if state["messages"] else ""

    response = _llm(temperature=0.0).invoke([
        SystemMessage(content=SUPERVISOR_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    route = response.content.strip().lower().split()[0] if response.content else "chitchat"
    if route not in ("sql", "rag", "hybrid", "chitchat"):
        route = "chitchat"  # safe fallback

    return {"route": route, "user_query": user_msg}


def route_decision(state: AgentState) -> Literal["sql_agent", "rag_agent", "hybrid_agent", "chitchat_agent"]:
    """Conditional edge function - picks the next node based on supervisor decision."""
    return {
        "sql": "sql_agent",
        "rag": "rag_agent",
        "hybrid": "hybrid_agent",
        "chitchat": "chitchat_agent",
    }[state["route"]]


# ------------------------------- Specialist Agents -------------------------------

def _run_tool_loop(
    llm_with_tools, tools: list, system_prompt: str,
    user_query: str, max_iterations: int = 5,
) -> str:
    """
    Generic ReAct-style loop:
      LLM proposes tool calls -> we run them -> feed results back -> repeat
    Returns the agent's final text answer when it stops calling tools
    (or hits max_iterations as a guard).
    """
    tool_map = {t.name: t for t in tools}
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]

    for _ in range(max_iterations):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        if not getattr(ai_msg, "tool_calls", None):
            return ai_msg.content or ""

        for tc in ai_msg.tool_calls:
            tool_fn = tool_map.get(tc["name"])
            if tool_fn is None:
                result = f"Error: tool '{tc['name']}' not found"
            else:
                try:
                    result = tool_fn.invoke(tc["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return "Reached max tool iterations without producing a final answer."


SQL_AGENT_SYSTEM = """You are the SQL Agent. You have tools to query a customer
support SQLite database with three tables: customers, orders, support_tickets.

Use the tools to answer the user's question:
- Prefer 'get_customer_profile' when the user asks about a specific customer
  by name or email - it returns the complete profile in one call.
- Use 'query_customer_database' for analytics / aggregate / filter queries.

Once you have enough data, write a clear, factual answer for the user.
Do not fabricate. If the data isn't there, say so."""

def sql_agent_node(state: AgentState) -> dict:
    """Specialist agent for structured-data questions."""
    llm = _llm(temperature=0.0).bind_tools(SQL_TOOLS)
    answer = _run_tool_loop(llm, SQL_TOOLS, SQL_AGENT_SYSTEM, state["user_query"])
    return {"sql_result": answer}


RAG_AGENT_SYSTEM = """You are the RAG (Retrieval-Augmented Generation) Agent.
You answer questions about company policies by searching the indexed policy
PDFs and citing the relevant passages.

Always call 'search_policy_documents' first. Base your answer ONLY on the
returned chunks. Cite the source filename for any fact you state, like
"(per refund_policy.pdf)". If the policies don't contain the answer, say
that clearly - do not guess."""

def rag_agent_node(state: AgentState) -> dict:
    """Specialist agent for unstructured-document questions."""
    llm = _llm(temperature=0.0).bind_tools(RAG_TOOLS)
    answer = _run_tool_loop(llm, RAG_TOOLS, RAG_AGENT_SYSTEM, state["user_query"])
    return {"rag_result": answer}


def hybrid_agent_node(state: AgentState) -> dict:
    """For questions needing both DB facts and policy context.
    Runs the SQL and RAG agents with question reformulations that make
    each agent retrieve the *facts* it owns, not pre-judge the answer."""

    user_q = state["user_query"]

    # SQL side: just gather the customer/order/ticket facts
    sql_subquery = (
        f"Retrieve all factual data needed to answer this question: '{user_q}'. "
        f"Pull the relevant customer's full profile, their orders (with dates, "
        f"amounts, and current status), and any related support tickets. "
        f"Do NOT try to evaluate the question - just return the raw facts."
    )
    sql_llm = _llm(temperature=0.0).bind_tools(SQL_TOOLS)
    sql_answer = _run_tool_loop(sql_llm, SQL_TOOLS, SQL_AGENT_SYSTEM, sql_subquery)

    # RAG side: just retrieve the relevant policy text
    rag_subquery = (
        f"Find the policy sections most relevant to this question: '{user_q}'. "
        f"Quote the specific eligibility criteria, conditions, exceptions, and "
        f"timeframes from the policy documents. Do NOT try to apply them to a "
        f"specific customer - just return what the policy says verbatim."
    )
    rag_llm = _llm(temperature=0.0).bind_tools(RAG_TOOLS)
    rag_answer = _run_tool_loop(rag_llm, RAG_TOOLS, RAG_AGENT_SYSTEM, rag_subquery)

    return {"sql_result": sql_answer, "rag_result": rag_answer}


def chitchat_agent_node(state: AgentState) -> dict:
    """Handles greetings and meta questions without using any tools."""
    msg = _llm(temperature=0.3).invoke([
        SystemMessage(content=(
            "You are a friendly customer support AI assistant. The user said "
            "something that doesn't require database or policy lookup. Respond "
            "warmly and briefly. Mention you can help with customer data lookups "
            "and policy questions if appropriate."
        )),
        HumanMessage(content=state["user_query"]),
    ])
    return {"sql_result": "", "rag_result": msg.content}


# ------------------------------- Synthesis (final) -------------------------------

SYNTHESIS_SYSTEM = """You are the final synthesizer for a customer support AI.
You receive (a) the user's original question, (b) factual customer data from
the SQL agent, and (c) policy text from the RAG agent. Your job is to
produce a clear, accurate, well-reasoned answer.

CRITICAL RULES FOR HYBRID QUESTIONS (questions that ask whether something is
allowed, eligible, valid, or compliant under our policy):

1. Apply the policy to the facts - don't just summarize both. The user wants
   a verdict, not a recap.

2. Walk through the reasoning explicitly:
   • State the relevant policy rule (cite the source PDF).
   • State the relevant customer facts (dates, amounts, statuses).
   • Apply the rule to the facts.
   • Give a clear verdict: ELIGIBLE / NOT ELIGIBLE / NEEDS REVIEW.

3. Do NOT confuse "this customer already received a refund in the past"
   with "this customer is eligible for a refund now." A past refund in the
   data is just a fact about history it does not establish present
   eligibility.

4. If the policy is restrictive by default (e.g. "payments are
   non-refundable except in specific cases"), the verdict should be NOT
   ELIGIBLE unless the customer's facts clearly meet a stated exception.

5. If the policy text doesn't directly answer the question, say so - don't
   invent rules. Suggest what additional info would be needed.

GENERAL FORMATTING:
- Lead with the verdict for hybrid questions, or the direct answer otherwise.
- For customer profiles: profile basics -> orders -> ticket history.
- For policy questions: cite the source PDF in parentheses.
- Use markdown sparingly. Be concise. No filler.
"""

def synthesis_node(state: AgentState) -> dict:
    """Compose the final user-facing answer from agent outputs."""
    # Chitchat path returns its message directly via rag_result
    if state["route"] == "chitchat":
        final = state.get("rag_result", "Hello! How can I help?")
        return {"messages": [AIMessage(content=final)]}

    parts = [f"USER QUESTION: {state['user_query']}\n"]
    if state.get("sql_result"):
        parts.append(f"\n--- STRUCTURED DATA FINDINGS (from SQL agent) ---\n{state['sql_result']}")
    if state.get("rag_result"):
        parts.append(f"\n--- POLICY DOCUMENT FINDINGS (from RAG agent) ---\n{state['rag_result']}")

    response = _llm(temperature=0.2).invoke([
        SystemMessage(content=SYNTHESIS_SYSTEM),
        HumanMessage(content="\n".join(parts)),
    ])
    return {"messages": [AIMessage(content=response.content)]}


# ------------------------------- Graph -------------------------------

def build_graph():
    """Wire every node into the LangGraph state machine."""
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("hybrid_agent", hybrid_agent_node)
    graph.add_node("chitchat_agent", chitchat_agent_node)
    graph.add_node("synthesis", synthesis_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_decision,
        {
            "sql_agent": "sql_agent",
            "rag_agent": "rag_agent",
            "hybrid_agent": "hybrid_agent",
            "chitchat_agent": "chitchat_agent",
        },
    )
    graph.add_edge("sql_agent", "synthesis")
    graph.add_edge("rag_agent", "synthesis")
    graph.add_edge("hybrid_agent", "synthesis")
    graph.add_edge("chitchat_agent", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


# Singleton compiled graph
_compiled = None

def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def run_query(user_message: str) -> dict:
    """Convenience runner. Returns answer + routing metadata."""
    graph = get_graph()
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_message)],
        "route": "",
        "sql_result": "",
        "rag_result": "",
        "user_query": user_message,
    }
    final_state = graph.invoke(initial_state)
    return {
        "answer": final_state["messages"][-1].content,
        "route": final_state.get("route", "unknown"),
        "sql_result": final_state.get("sql_result", ""),
        "rag_result": final_state.get("rag_result", ""),
    }


if __name__ == "__main__":
    # Quick smoke test
    test_queries = [
        "Hello!",
        "What is the refund policy?",
        "Give me an overview of customer Ema",
        "Is customer Ema eligible for a refund based on our policy?",
    ]
    for q in test_queries:
        print(f"\n{'=' * 70}\nQ: {q}")
        result = run_query(q)
        print(f"Route: {result['route']}")
        print(f"Answer:\n{result['answer']}")
