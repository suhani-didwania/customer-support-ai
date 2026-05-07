"""
End-to-end demo of the MCP server.

Spawns the server via stdio, lists the registered tools, and exercises each
one. Useful as both a sanity check and as a reference for how an external
MCP host (Claude Desktop, IDE plugin, another agent framework) would
integrate with this server.

Usage:
    python scripts/mcp_client_demo.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PATH = Path(__file__).parent.parent / "mcp_server" / "server.py"


def _section(title: str) -> None:
    print()
    print("-" * 60)
    print(title)
    print("-" * 60)


async def run_demo() -> None:
    """Connect to the MCP server, list tools, and call each one."""
    server_params = StdioServerParameters(
        command="python",
        args=[str(SERVER_PATH)],
    )

    print("Connecting to MCP server over stdio...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # --- registered tools ---
            tools_response = await session.list_tools()
            _section("Registered tools")
            for t in tools_response.tools:
                first_line = (t.description or "").split("\n")[0]
                print(f"  {t.name}")
                print(f"      {first_line}")

            # --- 1. indexed documents ---
            _section("list_indexed_documents()")
            r = await session.call_tool("list_indexed_documents", {})
            print(r.content[0].text)

            # --- 2. merchant profile lookup ---
            _section("get_customer_profile(identifier='Ema')")
            r = await session.call_tool("get_customer_profile", {"identifier": "Ema"})
            data = json.loads(r.content[0].text)
            if data.get("found"):
                m = data["merchant"]
                print(f"Merchant:        {m['first_name']} {m['last_name']} ({m['email']})")
                print(f"Business:        {m['business_name']}")
                print(f"Plan:            {m['subscription_plan']}")
                print(f"Account status:  {m['account_status']}")
                print(f"KYC status:      {m['kyc_status']}")
                print(f"Country:         {m['country']} (Stripe acct: {m['stripe_account_country']})")
                print(f"Monthly volume:  ${m['monthly_volume_usd']:,.2f} USD")
                print()
                print(f"Transactions:    {data['total_transaction_count']} "
                      f"(total ${data['total_transaction_volume_usd']:,.2f} USD)")
                print(f"Disputes:        {data['dispute_count']} "
                      f"(chargeback ratio: {data['chargeback_ratio']:.2%})")
                print(f"Support tickets: {data['ticket_count']}")
            else:
                print(data.get("message"))

            # --- 3. analytics query ---
            _section("query_customer_database (counts by plan)")
            sql = (
                "SELECT subscription_plan, COUNT(*) AS n "
                "FROM merchants GROUP BY subscription_plan ORDER BY n DESC"
            )
            r = await session.call_tool("query_customer_database", {"sql": sql})
            print(r.content[0].text)

            # --- 4. policy search ---
            _section("search_policy_documents('refunds and non-refundable fees', k=2)")
            r = await session.call_tool(
                "search_policy_documents",
                {"query": "refunds and non-refundable fees", "k": 2},
            )
            data = json.loads(r.content[0].text)
            for i, hit in enumerate(data.get("results", []), 1):
                print(f"\n[{i}] {hit['source']} (page {hit['page']})")
                snippet = hit["content"].strip().replace("\n", " ")
                print(f"    {snippet[:280]}...")

            print()
            print("Demo complete.\n")


if __name__ == "__main__":
    asyncio.run(run_demo())