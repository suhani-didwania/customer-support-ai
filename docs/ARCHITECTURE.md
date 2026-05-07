# Architecture

## The multi-agent graph

The graph lives in `agents/graph.py`. Six nodes:

```
START -> supervisor -+-> sql_agent ------+
                     +-> rag_agent ------+
                     +-> hybrid_agent ---+--> synthesis -> END
                     +-> chitchat_agent -+
```

### Why a supervisor pattern instead of one big agent?

The first version of this had a single agent with all the tools attached. It
worked, but had three problems that got worse as the toolset grew.

Specialization. Each specialist has a small, focused tool list. Shorter
prompts, sharper tool-call decisions, contained failure modes. A bad SQL
query doesn't poison a policy lookup.

Cost. Chitchat questions never touch the database or vector store. Pure
SQL questions never trigger an embedding call. The supervisor adds one extra
LLM call (~200ms with gpt-4o-mini), but that's more than offset by the
specialists running with smaller tool sets.

Observability. The routing decision is part of the state, so the UI can
show *why* a particular path was taken. The "agent trace" expander in the
chat panel reads this directly.

### The shared AgentState

Every node reads from and writes to a typed state dict:

```python
class AgentState(TypedDict):
    messages: list[BaseMessage]   # appended via reducer
    route: str                    # supervisor's decision
    sql_result: str               # SQL agent's findings
    rag_result: str               # RAG agent's findings
    user_query: str               # original user input
```

The `Annotated[list, add]` reducer pattern means each node can append to
`messages` without overwriting prior turns. That's what enables clean
multi-turn conversations later if a checkpointer is added.

## The MCP layer

The brief calls for an MCP Server, so `mcp_server/server.py` implements
one properly: stdio transport, tools registered via `@server.list_tools()`,
calls handled via `@server.call_tool()`. There's a working client demo at
`scripts/mcp_client_demo.py` that proves the server is operational and
standards-compliant.

### Internal vs external tool paths

The LangGraph agents call the same business logic the MCP server exposes,
but they call it directly (no IPC) because we control both ends and want
the agent loop to be fast. The MCP server is the "external" entry point -
useful when an MCP host (Claude Desktop, an IDE plugin, another agent
framework) wants to plug in.

| Caller | Path | Why |
|---|---|---|
| LangGraph agents (in-process) | Direct function calls in `agents/tools.py` | Latency. The agent loop runs many iterations; IPC per call would be wasteful. |
| External MCP clients | Server over stdio | Standard protocol. Lets external tools use our capabilities without code changes. |

Both paths share the same underlying logic in `utils/`. No duplication.

## Data layer

### Structured (SQL)

SQLite for zero-setup deployment. Three tables - `merchants`,
`transactions`, `disputes`, `support_tickets` - with foreign keys and
indexes on the columns we filter on.

All access goes through `utils/sql_database.py`. Every query is validated
as read-only before execution: `SELECT` and `WITH` are allowed, anything
else (`DROP`, `DELETE`, `UPDATE`, `ALTER`, etc.) is rejected. The check is
substring-based but uses padded matching so column names like `updated_at`
don't trigger a false positive on `UPDATE`.

### Unstructured (vector)

ChromaDB persisted to disk at `data/chroma_db/`.

Chunking: `RecursiveCharacterTextSplitter` with `chunk_size=1000` and
`chunk_overlap=200`. These values work well for policy text - small enough
that retrieval stays focused, large enough that paragraph context isn't
shredded.

Embeddings: OpenAI `text-embedding-3-small`. Cheap (~$0.02 per million
tokens), high quality, and 1536-dimensional which Chroma handles well.

Each chunk carries metadata: `source_file`, `page`, `doc_type`. The RAG
agent uses `source_file` to cite back to the original PDF in its answers.

## Prompt engineering

A few choices worth flagging:

The supervisor's classification prompt is closed-set. It's instructed to
return exactly one of four words. The code defensively defaults to
`chitchat` if the response is anything else, so the router can't crash on
a weird LLM response.

The synthesis prompt enforces a specific reasoning pattern for hybrid
questions: state the policy rule, state the customer facts, apply rule to
facts, give a verdict. Without this, the LLM tends to summarize both
findings instead of actually reasoning across them. Rule 3 in the prompt
("don't confuse a past refund with present eligibility") was added after
observing exactly that mistake during testing.

Temperatures: 0.0 for the supervisor and specialists (we want deterministic
routing and tool calls), 0.2 for synthesis (slightly more natural prose).

The hybrid agent issues separate sub-queries to the SQL and RAG agents that
explicitly tell each one *not* to evaluate the question - just retrieve the
facts. The synthesis node does the evaluation. This separation prevents
either specialist from pre-judging the answer based on partial info.

## Security

- **LLM cannot write to the DB.** The `is_safe_query()` whitelist rejects
  anything that isn't `SELECT`/`WITH`. No `INSERT`, `UPDATE`, `DELETE`,
  `DROP`, etc.
- **LLM cannot execute arbitrary code.** Agents can only invoke
  registered tools. There's no Python REPL tool, no shell tool.
- **Bounded ReAct loops.** Each specialist has `max_iterations=5`. A
  runaway agent can't burn through tokens or rate limits.
- **API keys live in `.env`.** Never committed. `.gitignore` excludes it.

## Performance notes

- Supervisor latency: ~200ms with `gpt-4o-mini`. Acceptable.
- ChromaDB similarity search: sub-100ms for our document corpus, all local.
- Embedding generation happens once per PDF at ingest time, not at query
  time. PDF uploads are the only expensive operation, and they're shown
  with a progress bar in the UI.
- Total round-trip for a typical question: 3-8 seconds end to end. Most of
  it is the synthesis LLM call. If we needed it faster, the obvious move
  would be streaming the synthesis response token-by-token to the UI.