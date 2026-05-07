"""
LangChain tools used by the LangGraph agents.

These wrap the same business logic exposed by the MCP server, but call it
directly (no IPC) for speed inside the agent loop. The MCP server exposes
the same capabilities to external clients (Claude Desktop, IDEs, etc.).
"""
import json
from typing import Annotated

from langchain_core.tools import tool

from utils.sql_database import execute_query, get_customer_summary, get_schema_description
from utils.vector_store import get_vector_store


@tool
def query_customer_database(
    sql: Annotated[str, "A read-only SELECT statement against the customer support DB."]
) -> str:
    """
    Run a SQL SELECT query against the customer support database.
    The database has three tables: customers, orders, and support_tickets.
    Use this for analytics, lookups, ticket counts, plan distribution, etc.
    The agent should construct valid SQLite syntax. Use case-insensitive
    LIKE comparisons for names. Always JOIN on customer_id when retrieving
    cross-table customer details.
    """
    result = execute_query(sql)
    return json.dumps(result, default=str, indent=2)


# Append the live schema to the tool's description so the LLM always sees it
query_customer_database.description = (
    query_customer_database.description + "\n\n" + get_schema_description()
)

@tool
def get_customer_profile(
    identifier: Annotated[str, "Merchant name (first/last/full), business name, or email."]
) -> str:
    """
    Retrieve a complete profile for a merchant on our platform: their account
    details (status, KYC, plan, country), recent transactions, all chargebacks/
    disputes, support tickets, and aggregate stats including chargeback ratio.

    Use this when the user asks for an overview/summary/history of a specific
    merchant (e.g. "tell me about Ema", "show me Williams Coffee Co.",
    "history for ema@williamscoffee.com").
    """
    result = get_customer_summary(identifier)
    return json.dumps(result, default=str, indent=2)


@tool
def search_policy_documents(
    query: Annotated[str, "The natural-language question to search for in policy PDFs."]
) -> str:
    """
    Semantic search over indexed company policy PDFs (refund policy, privacy
    policy, shipping policy, etc.).

    Use this for any question about company policies, terms, procedures,
    eligibility rules, or anything answered by reading a policy document.

    Returns the top relevant chunks with source filenames so you can cite them.
    """
    try:
        vs = get_vector_store()
        docs = vs.search(query, k=4)
        if not docs:
            return json.dumps({
                "results": [],
                "message": "No policy documents have been indexed yet, or no chunks matched.",
            })
        payload = [
            {
                "content": d.page_content,
                "source": d.metadata.get("source_file", "unknown"),
                "page": d.metadata.get("page", "?"),
            }
            for d in docs
        ]
        return json.dumps({"results": payload}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "results": []})


# Tool collections grouped by which agent uses them
SQL_TOOLS = [query_customer_database, get_customer_profile]
RAG_TOOLS = [search_policy_documents]
ALL_TOOLS = SQL_TOOLS + RAG_TOOLS