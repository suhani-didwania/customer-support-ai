"""
Terminal client for the customer support assistant.

Useful for quick checks without launching the Streamlit UI, and for piping
output into other tools. Same multi-agent graph as the web UI underneath.

Usage:
    python scripts/cli.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.graph import run_query


BANNER = """
Customer Support Assistant (CLI)
--------------------------------
Ask about a merchant, a policy, or a compliance question.
Type 'exit' or Ctrl-D to quit.
"""

ROUTE_TAG = {
    "sql":      "[merchant data]",
    "rag":      "[policy]",
    "hybrid":   "[data + policy]",
    "chitchat": "[general]",
}


def main() -> None:
    print(BANNER)
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit", "bye"):
            break

        try:
            result = run_query(q)
            tag = ROUTE_TAG.get(result["route"], f"[{result['route']}]")
            print(f"\n{tag}\n")
            print(result["answer"])
            print()
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()