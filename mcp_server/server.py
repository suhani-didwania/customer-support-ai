"""
MCP (Model Context Protocol) server for the customer support assistant.

Exposes the same capabilities the LangGraph agents use internally as
standards-compliant MCP tools, so external MCP hosts (e.g. Claude Desktop)
can connect and query this system without any custom integration code.

Transport: stdio. To launch directly:
    python mcp_server/server.py

To verify it works end-to-end without a full client setup:
    python scripts/mcp_client_demo.py

The four tools registered here mirror what the in-process LangGraph agents
call. Both paths share the same business logic in utils/, so there's no
duplication or drift.
"""
import asyncio
import json
import sys
from pathlib import Path

# Make project imports resolve when running this file directly via `python`
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from utils.sql_database import (
    execute_query,
    get_customer_summary,
    get_schema_description,
)
from utils.vector_store import get_vector_store


server = Server("customer-support-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Advertise available tools to the MCP client."""
    return [
        Tool(
            name="query_customer_database",
            description=(
                "Execute a read-only SQL SELECT query against the merchant "
                "support database. Returns rows as JSON. Use this for "
                "analytics, lookups, and joining merchant / transaction / "
                "dispute / ticket data.\n\n" + get_schema_description()
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A SELECT statement to run. Must be read-only.",
                    },
                },
                "required": ["sql"],
            },
        ),
        Tool(
            name="get_customer_profile",
            description=(
                "Retrieve a complete profile for one merchant: account details "
                "(status, KYC, plan, country), recent transactions, all "
                "chargebacks/disputes, support tickets, and aggregate stats "
                "including chargeback ratio. Search by merchant name, business "
                "name, or email."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "Merchant name, business name, or email address.",
                    },
                },
                "required": ["identifier"],
            },
        ),
        Tool(
            name="search_policy_documents",
            description=(
                "Semantic search over indexed company policy PDFs. Returns the "
                "top relevant chunks with their source filenames so you can "
                "cite them. Use for any question about company policies, "
                "terms, procedures, or rules."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language question or topic.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of chunks to return (default 4).",
                        "default": 4,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_indexed_documents",
            description="List source filenames of all PDFs currently indexed in the vector DB.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch tool invocations from the MCP client."""

    if name == "query_customer_database":
        sql = arguments.get("sql", "")
        result = execute_query(sql)
        return [TextContent(type="text", text=json.dumps(result, default=str, indent=2))]

    if name == "get_customer_profile":
        identifier = arguments.get("identifier", "")
        result = get_customer_summary(identifier)
        return [TextContent(type="text", text=json.dumps(result, default=str, indent=2))]

    if name == "search_policy_documents":
        query = arguments.get("query", "")
        k = arguments.get("k", 4)
        try:
            vs = get_vector_store()
            docs = vs.search(query, k=k)
            payload = [
                {
                    "content": d.page_content,
                    "source": d.metadata.get("source_file", "unknown"),
                    "page": d.metadata.get("page", "?"),
                }
                for d in docs
            ]
            return [TextContent(type="text", text=json.dumps({"results": payload}, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e), "results": []}))]

    if name == "list_indexed_documents":
        try:
            vs = get_vector_store()
            docs = vs.list_documents()
            return [TextContent(type="text", text=json.dumps({"documents": docs}))]
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e), "documents": []}))]

    return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def main() -> None:
    """Run the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())