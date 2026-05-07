"""
Read-only SQL access for the customer support agent.

Two design decisions worth flagging:

1. The agent layer never sees raw cursor handles or builds SQL through string
   concatenation. It calls the functions in this file, which run with a
   safety check on the way in.

2. Every query goes through `is_safe_query()`, which rejects anything that
   isn't a SELECT or WITH (CTE) read. This is belt-and-suspenders since the
   tool definitions only ask for SELECT, but it means even a misbehaving LLM
   can't issue an UPDATE or DROP through the database tool.
"""
import sqlite3
from pathlib import Path
from typing import Any

from config import SQL_DB_PATH


# Only SELECT and CTE-based reads are allowed
_ALLOWED_PREFIXES = ("SELECT", "WITH")
_FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "REPLACE", "ATTACH", "DETACH", "PRAGMA",
)


def get_schema_description() -> str:
    """Schema description fed into LLM tool descriptions and agent prompts."""
    return """
DATABASE SCHEMA (Stripe-like payments platform):

Table: merchants
  - merchant_id (INTEGER, PRIMARY KEY)
  - first_name, last_name (TEXT) - the primary contact at the business
  - business_name (TEXT)
  - email (TEXT, UNIQUE)
  - phone, country (TEXT)
  - stripe_account_country (TEXT) - 2-letter code (US, CA, IN, etc.)
  - account_status (TEXT: 'Active', 'Restricted', 'Under Review', 'Suspended', 'Closed')
  - kyc_status (TEXT: 'Verified', 'Pending', 'Failed', 'Not Started')
  - subscription_plan (TEXT: 'Standard', 'Custom', 'Enterprise')
  - mcc_category (TEXT) - merchant category, e.g. 'Software/SaaS', 'E-commerce - Apparel'
  - signup_date (DATE)
  - monthly_volume_usd (REAL) - average monthly processing volume

Table: transactions
  - transaction_id (INTEGER, PRIMARY KEY)
  - merchant_id (INTEGER, FK -> merchants)
  - amount_usd (REAL)
  - currency (TEXT, default 'USD')
  - payment_method (TEXT: e.g. 'card_visa', 'ach_debit', 'apple_pay')
  - transaction_date (DATETIME)
  - status (TEXT: 'Succeeded', 'Pending', 'Failed', 'Refunded', 'Partially Refunded')
  - fee_usd (REAL) - the processing fee for the transaction

Table: disputes
  - dispute_id (INTEGER, PRIMARY KEY)
  - merchant_id (INTEGER, FK -> merchants)
  - transaction_id (INTEGER, FK -> transactions, nullable)
  - amount_usd (REAL)
  - reason (TEXT: 'fraudulent', 'product_not_received', 'duplicate',
            'subscription_canceled', 'credit_not_processed', 'general')
  - status (TEXT: 'needs_response', 'under_review', 'won', 'lost', 'warning_closed')
  - opened_at, resolved_at (DATETIME)
  - evidence_due_by (DATETIME) - deadline for merchant to submit evidence

Table: support_tickets
  - ticket_id (INTEGER, PRIMARY KEY)
  - merchant_id (INTEGER, FK -> merchants)
  - subject, description (TEXT)
  - category (TEXT: 'Billing', 'Disputes', 'API/Integration', 'Account',
             'Payouts', 'Compliance', 'General')
  - priority (TEXT: 'Low', 'Medium', 'High', 'Urgent')
  - status (TEXT: 'Open', 'In Progress', 'Resolved', 'Closed')
  - created_at, resolved_at (DATETIME)
  - agent_notes (TEXT)

QUERYING NOTES:
- Use case-insensitive comparisons for names: LOWER(first_name) LIKE LOWER('%ema%')
- Always JOIN on merchant_id when retrieving cross-table merchant details
- The merchant 'Ema Williams' (Williams Coffee Co.) exists in the seed data
- A merchant's chargeback ratio = count(disputes) / count(transactions)
"""


def is_safe_query(sql: str) -> tuple[bool, str]:
    """Reject anything that isn't a read-only SELECT/WITH query."""
    sql_clean = sql.strip().upper()
    if not sql_clean:
        return False, "Empty query"

    if not any(sql_clean.startswith(p) for p in _ALLOWED_PREFIXES):
        return False, f"Only {_ALLOWED_PREFIXES} queries are allowed"

    # Match forbidden keywords as whole words. Padding both sides with a
    # space avoids false positives like 'updated_at' triggering on UPDATE.
    padded = f" {sql_clean} "
    for kw in _FORBIDDEN_KEYWORDS:
        if f" {kw} " in padded or sql_clean.startswith(f"{kw} "):
            return False, f"Forbidden keyword detected: {kw}"

    return True, "OK"


def execute_query(sql: str, max_rows: int = 50) -> dict[str, Any]:
    """Run a read-only SQL query. Returns rows + columns, or an error dict."""
    is_safe, msg = is_safe_query(sql)
    if not is_safe:
        return {"error": f"Query rejected: {msg}", "rows": [], "columns": []}

    try:
        conn = sqlite3.connect(SQL_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchmany(max_rows)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()
        return {
            "columns": columns,
            "rows": [dict(r) for r in rows],
            "row_count": len(rows),
            "error": None,
        }
    except sqlite3.Error as e:
        return {"error": str(e), "rows": [], "columns": []}


def get_customer_summary(identifier: str) -> dict[str, Any]:
    """Return a full profile for one merchant: account details, recent
    transactions, disputes, support tickets, and a few aggregate stats
    (transaction volume, chargeback ratio).

    Looks up the merchant by name, business name, or email. Case-insensitive
    on all of them. Returns {"found": False, "message": ...} if no match.

    The function is named `get_customer_summary` rather than
    `get_merchant_summary` to keep the public tool name stable - the LLM
    tool is registered as `get_customer_profile` and it would be confusing
    to have helpers with mismatched names. Internally everything is a
    merchant.
    """
    if not Path(SQL_DB_PATH).exists():
        return {"found": False, "message": "Database not initialized."}

    conn = sqlite3.connect(SQL_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """SELECT * FROM merchants
           WHERE LOWER(email) = LOWER(?)
              OR LOWER(first_name) LIKE LOWER(?)
              OR LOWER(last_name) LIKE LOWER(?)
              OR LOWER(first_name || ' ' || last_name) LIKE LOWER(?)
              OR LOWER(business_name) LIKE LOWER(?)
           LIMIT 1""",
        (identifier, f"%{identifier}%", f"%{identifier}%",
         f"%{identifier}%", f"%{identifier}%"),
    )
    merchant = cursor.fetchone()

    if not merchant:
        conn.close()
        return {"found": False, "message": f"No merchant found matching '{identifier}'"}

    mid = merchant["merchant_id"]

    cursor.execute(
        "SELECT * FROM support_tickets WHERE merchant_id = ? ORDER BY created_at DESC",
        (mid,),
    )
    tickets = [dict(t) for t in cursor.fetchall()]

    cursor.execute(
        """SELECT * FROM transactions WHERE merchant_id = ?
           ORDER BY transaction_date DESC LIMIT 20""",
        (mid,),
    )
    transactions = [dict(t) for t in cursor.fetchall()]

    cursor.execute(
        "SELECT * FROM disputes WHERE merchant_id = ? ORDER BY opened_at DESC",
        (mid,),
    )
    disputes = [dict(d) for d in cursor.fetchall()]

    # Aggregate stats. We send these to the LLM separately so it doesn't
    # have to count rows itself - more reliable, and cheaper in tokens.
    cursor.execute(
        "SELECT COUNT(*) AS n, COALESCE(SUM(amount_usd), 0) AS total "
        "FROM transactions WHERE merchant_id = ?",
        (mid,),
    )
    txn_agg = cursor.fetchone()

    cursor.execute(
        "SELECT COUNT(*) AS n FROM disputes WHERE merchant_id = ?",
        (mid,),
    )
    dispute_count = cursor.fetchone()["n"]

    chargeback_ratio = (dispute_count / txn_agg["n"]) if txn_agg["n"] > 0 else 0.0

    conn.close()
    return {
        "found": True,
        "merchant": dict(merchant),
        "tickets": tickets,
        "recent_transactions": transactions,
        "disputes": disputes,
        "ticket_count": len(tickets),
        "total_transaction_count": txn_agg["n"],
        "total_transaction_volume_usd": round(txn_agg["total"], 2),
        "dispute_count": dispute_count,
        "chargeback_ratio": round(chargeback_ratio, 4),
    }