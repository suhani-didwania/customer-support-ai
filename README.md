# Customer Support AI

A multi-agent system that lets a support agent query both structured merchant data and unstructured policy documents through one natural-language interface.

Built on LangGraph (supervisor pattern), an MCP server for external tool access, ChromaDB for vector search, and SQLite for structured data. Streamlit on top.

The use case: a support exec receives a question that mixes account-level facts ("what's this merchant's chargeback ratio?") with policy questions ("does our agreement allow us to suspend this account?"). They shouldn't have to query a database and read a PDF separately. The system routes each question to the right specialist, and for hybrid questions, applies the policy to the facts and returns a reasoned verdict.

## Demo

**Watch the demo:** https://www.loom.com/share/dcf0635e1929462284eb6214b382837d

The walkthrough covers:

1. Asking a policy question (RAG path)
2. Pulling a merchant's full profile (SQL path)
3. A hybrid compliance check ("does this merchant meet our standards?")
4. Uploading a new policy PDF live
5. Inspecting the agent trace to see *why* a particular path was taken
6. Running the MCP client demo from the terminal to prove the MCP server works

## What's in the data

**Structured (SQLite).** Synthetic merchant data modeling a Stripe-like B2B payments platform: 25 merchants, ~400 transactions, ~30 disputes, ~75 support tickets. The schema is `merchants`, `transactions`, `disputes`, `support_tickets` with foreign keys and indexes. Three named merchants are seeded so demo questions work without setup: Ema Williams (Williams Coffee Co.), John Smith (Smith Analytics Inc.), and Priya Sharma (Sharma Boutique).

**Unstructured (ChromaDB).** Stripe's publicly available [Services Agreement (General Terms)](https://stripe.com/legal/ssa). Loaded at first run; you can drop additional PDFs in `data/uploaded_pdfs/` or upload them through the UI.

The data and the policy are intentionally aligned. Real-world support questions reference both, and that alignment is what makes hybrid queries meaningful.

## Architecture

                    +-----------------------------------+
                    |          Streamlit UI             |
                    |        (chat + PDF upload)        |
                    +----------------+------------------+
                                     |
                                     v
                    +-----------------------------------+
                    |       LangGraph state machine     |
                    |                                   |
                    |             supervisor            |
                    |     (intent classifier / router)  |
                    |         |    |    |    |          |
                    |    +----+    |    |    +----+     |
                    |    v         v    v         v     |
                    |   sql      rag  hybrid   convers. |
                    |  agent    agent  agent    agent   |
                    |    |         |    |         |     |
                    |    +----+----+----+----+----+     |
                    |              v                    |
                    |          synthesis                |
                    +-----------+-----------------------+
                                      |
                             +--------+---------+
                             v                  v
                          +-------------+   +----------------+
                          | SQLite DB   |   | ChromaDB       |
                          | merchants   |   | policy chunks  |
                          | txns        |   | + embeddings   |
                          | disputes    |   +----------------+
                          | tickets     |
                          +-------------+
                                   ^                  ^
                                   +-- MCP server ----+
                               (same tools, exposed over the
                                Model Context Protocol for use
                                by external MCP-compatible hosts)

Why a supervisor pattern instead of one big agent with all the tools? Specialists have smaller, focused tool lists, which means shorter prompts, sharper tool-call decisions, and contained failure modes. Pure SQL questions never trigger embedding lookups; chitchat questions never touch the database. The routing decision is part of the state, so the UI can show *why* a particular path was taken.

More design detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Tech stack

| Layer | Choice |
|---|---|
| Agent framework | LangGraph (supervisor pattern, stateful graph) |
| LLM | OpenAI gpt-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector store | ChromaDB (persistent, local) |
| Structured store | SQLite |
| External tool protocol | MCP (Model Context Protocol) |
| UI | Streamlit |
| PDF parsing | PyPDF |
| Synthetic data | Faker |

## Project layout

```
customer-support-ai/
├── agents
│   ├── __init__.py
│   ├── graph.py
│   └── tools.py
├── docs
│   └── ARCHITECTURE.md
├── mcp_server
│   ├── __init__.py
│   └── server.py
├── scripts
│   ├── cli.py
│   ├── ingest_documents.py
│   ├── init_database.py
│   └── mcp_client_demo.py
├── ui
│   └── app.py
├── utils
│   ├── __init__.py
│   ├── sql_database.py
│   └── vector_store.py
├── data/                      # runtime data, gitignored
├── README.md
├── config.py
├── requirements.txt
└── setup.sh
```

## Quick start

Requirements: Python 3.10+ and an OpenAI API key.

```bash
git clone https://github.com/<your-username>/customer-support-ai.git
cd customer-support-ai

cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

./setup.sh
```

The script creates a venv, installs dependencies, seeds the database, and indexes any PDFs you've placed in `data/uploaded_pdfs/`. If you don't have any PDFs yet, the script will skip indexing with a clear message - drop them in and re-run `python scripts/ingest_documents.py`.

Then launch:

```bash
streamlit run ui/app.py
```

Open http://localhost:8501.

## Manual setup

If `setup.sh` doesn't fit your workflow (Windows, custom venv, etc.):

```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate            # Windows

pip install -r requirements.txt

cp .env.example .env               # add your OPENAI_API_KEY

python scripts/init_database.py    # seed merchants, transactions, etc.
python scripts/ingest_documents.py # index PDFs from data/uploaded_pdfs/

streamlit run ui/app.py
```

## Other entry points

**Terminal client.** Same multi-agent graph, no UI:

```bash
python scripts/cli.py
```

**MCP server (standalone).** Runs over stdio, the standard MCP transport:

```bash
python mcp_server/server.py
```

**MCP smoke test.** Verifies the server is operational by spawning it, listing the registered tools, and exercising each one:

```bash
python scripts/mcp_client_demo.py
```

**Plugging into Claude Desktop.** Add this to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "customer-support": {
      "command": "python",
      "args": ["/absolute/path/to/customer_support_ai/mcp_server/server.py"]
    }
  }
}
```

## How the multi-agent flow works

For the question *"Does Ema's account meet our standards under the Stripe terms?"*:

1. **Supervisor** classifies the intent as `hybrid` - needs both merchant data and policy text.

2. **Hybrid node** runs two specialists with explicit sub-queries that tell each one to *retrieve* the relevant facts, not to evaluate the question:
   - SQL agent calls `get_customer_profile("Ema")` and returns Ema's account status, KYC status, plan, monthly volume, transaction count, dispute count, and chargeback ratio.
   - RAG agent searches the indexed Stripe agreement and returns the sections about merchant standards, prohibited businesses, and account suspension grounds.

3. **Synthesis node** applies the policy to the facts. The prompt enforces a specific reasoning pattern: state the rule, state the facts, apply the rule, give a verdict (`ELIGIBLE` / `NOT ELIGIBLE` / `NEEDS REVIEW`). This is the part that took the most iteration - the first version would just summarize both findings instead of actually reasoning across them. Rule 3 in the synthesis prompt ("don't confuse a past refund with present eligibility") was added after observing exactly that mistake during testing.

Simpler questions skip the hybrid path:

- "What does our agreement say about non-refundable fees?" → `rag` only
- "How many merchants are under review?" → `sql` only
- "Hi" → `chitchat`

Each specialist runs a bounded ReAct loop (max 5 iterations) so a runaway agent can't burn through tokens or rate limits.

## The MCP server

`mcp_server/server.py` registers four tools:

| Tool | Purpose |
|---|---|
| `query_customer_database` | Run a safe SELECT against the merchant database |
| `get_customer_profile` | Full profile for one merchant: account, transactions, disputes, tickets |
| `search_policy_documents` | Semantic search over indexed policy PDFs |
| `list_indexed_documents` | Names of PDFs currently in the vector store |

The brief asks for an MCP server, so this is a real one - stdio transport, properly registered tools, runs against the same business logic the in-process LangGraph agents use. There's no duplication: both paths call the helpers in `utils/`. The MCP layer just makes those helpers available to external hosts (Claude Desktop, IDE plugins, other agent frameworks) without code changes on either side.

## Sample queries

| Query | Route | What happens |
|---|---|---|
| "What does our agreement say about non-refundable fees?" | rag | RAG agent searches the indexed Stripe agreement and quotes the relevant section |
| "Give me an overview of merchant Ema's account and recent activity" | sql | SQL agent calls `get_customer_profile("Ema")` and summarizes |
| "How many merchants are currently under review?" | sql | SQL agent runs a COUNT against `merchants` filtered on status |
| "Under what conditions can we suspend a merchant's account?" | rag | RAG agent finds the suspension/termination clauses |
| "Does Ema's account meet our risk standards?" | hybrid | Both specialists run; synthesis applies policy to facts and returns a verdict |
| "Hi" | chitchat | Plain greeting, no tools |

## Extending it

- **New data source.** Drop a helper in `utils/`, add a `@tool`-decorated wrapper in `agents/tools.py`, attach it to the relevant agent's tool list, and (optionally) register it on the MCP server.

- **Swap the LLM.** Change `LLM_MODEL` in `.env`. For Anthropic, swap `ChatOpenAI` for `ChatAnthropic` in `agents/graph.py`.

- **Local embeddings.** Replace `OpenAIEmbeddings` in `utils/vector_store.py` with `HuggingFaceEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2` is a solid default).

- **Multi-turn memory.** LangGraph supports checkpointers - add a `MemorySaver` and pass a `thread_id` through `run_query`.

- **Streaming responses.** The synthesis call is the longest part of the round-trip. Streaming it token-by-token to the UI is the obvious next move if interactivity matters more than throughput.

## Troubleshooting

**`OPENAI_API_KEY not set`** - copy `.env.example` to `.env` and add your key.

**`Database not initialized`** - run `python scripts/init_database.py`.

**No PDFs indexed** - drop PDFs in `data/uploaded_pdfs/` and run `python scripts/ingest_documents.py`. Or upload through the Streamlit UI.

**ChromaDB version mismatch after upgrades** - delete `data/chroma_db/` and re-run the ingest script.

**`Module not found`** - you forgot to activate the venv: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows).
