"""
Seed the SQLite database with synthetic merchant data.

The model approximates a B2B payments platform: merchants who process
transactions through us, with disputes (chargebacks) and support tickets.
This pairs cleanly with the Stripe Services Agreement loaded into the
vector store, so hybrid questions ("is this merchant compliant with our
terms?") have meaningful inputs on both sides.

Three named merchants are seeded so the demo questions referenced in the
brief work without setup: Ema Williams (Williams Coffee Co.), John Smith
(Smith Analytics Inc.), and Priya Sharma (Sharma Boutique).

Usage:
    python scripts/init_database.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
import random
from datetime import date, datetime, timedelta
from faker import Faker

from config import SQL_DB_PATH

fake = Faker()
Faker.seed(42)
random.seed(42)

# Store dates as ISO-8601 strings; silences Python 3.12 sqlite3 deprecations
sqlite3.register_adapter(date, lambda d: d.isoformat())
sqlite3.register_adapter(datetime, lambda dt: dt.isoformat(sep=" "))


def create_schema(conn: sqlite3.Connection) -> None:
    """Drop existing tables and recreate the schema from scratch."""
    cursor = conn.cursor()

    cursor.executescript("""
        DROP TABLE IF EXISTS disputes;
        DROP TABLE IF EXISTS transactions;
        DROP TABLE IF EXISTS support_tickets;
        DROP TABLE IF EXISTS merchants;
    """)

    cursor.executescript("""
        CREATE TABLE merchants (
            merchant_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            business_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            country TEXT,
            stripe_account_country TEXT,
            account_status TEXT CHECK(account_status IN ('Active', 'Restricted', 'Under Review', 'Suspended', 'Closed')),
            kyc_status TEXT CHECK(kyc_status IN ('Verified', 'Pending', 'Failed', 'Not Started')),
            subscription_plan TEXT CHECK(subscription_plan IN ('Standard', 'Custom', 'Enterprise')),
            mcc_category TEXT,
            signup_date DATE,
            monthly_volume_usd REAL DEFAULT 0
        );

        CREATE TABLE transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            merchant_id INTEGER NOT NULL,
            amount_usd REAL NOT NULL,
            currency TEXT DEFAULT 'USD',
            payment_method TEXT,
            transaction_date DATETIME NOT NULL,
            status TEXT CHECK(status IN ('Succeeded', 'Pending', 'Failed', 'Refunded', 'Partially Refunded')),
            fee_usd REAL,
            FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id)
        );

        CREATE TABLE disputes (
            dispute_id INTEGER PRIMARY KEY AUTOINCREMENT,
            merchant_id INTEGER NOT NULL,
            transaction_id INTEGER,
            amount_usd REAL NOT NULL,
            reason TEXT CHECK(reason IN ('fraudulent', 'product_not_received', 'duplicate', 'subscription_canceled', 'credit_not_processed', 'general')),
            status TEXT CHECK(status IN ('needs_response', 'under_review', 'won', 'lost', 'warning_closed')),
            opened_at DATETIME NOT NULL,
            resolved_at DATETIME,
            evidence_due_by DATETIME,
            FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id),
            FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
        );

        CREATE TABLE support_tickets (
            ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
            merchant_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            description TEXT,
            category TEXT CHECK(category IN ('Billing', 'Disputes', 'API/Integration', 'Account', 'Payouts', 'Compliance', 'General')),
            priority TEXT CHECK(priority IN ('Low', 'Medium', 'High', 'Urgent')),
            status TEXT CHECK(status IN ('Open', 'In Progress', 'Resolved', 'Closed')),
            created_at DATETIME NOT NULL,
            resolved_at DATETIME,
            agent_notes TEXT,
            FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id)
        );

        CREATE INDEX idx_merchant_email ON merchants(email);
        CREATE INDEX idx_ticket_merchant ON support_tickets(merchant_id);
        CREATE INDEX idx_txn_merchant ON transactions(merchant_id);
        CREATE INDEX idx_dispute_merchant ON disputes(merchant_id);
    """)
    conn.commit()
    print("  schema created")


def generate_merchants(conn: sqlite3.Connection, count: int = 25) -> list[int]:
    """Insert merchants. Three named seeds + (count - 3) random ones."""
    cursor = conn.cursor()

    mccs = [
        'Software/SaaS', 'E-commerce - Apparel', 'E-commerce - Electronics',
        'Food & Beverage', 'Professional Services', 'Subscription Box',
        'Digital Goods', 'Marketplace', 'Non-profit', 'Education',
    ]
    plans = ['Standard', 'Custom', 'Enterprise']
    plan_weights = [0.70, 0.20, 0.10]
    statuses = ['Active', 'Restricted', 'Under Review', 'Suspended', 'Closed']
    status_weights = [0.75, 0.08, 0.07, 0.05, 0.05]
    kyc_statuses = ['Verified', 'Pending', 'Failed', 'Not Started']
    kyc_weights = [0.80, 0.10, 0.05, 0.05]

    # Named seeds. Ema is referenced in the brief.
    showcase = [
        # (first, last, biz_name, email, phone, country, account_country,
        #  status, kyc, plan, mcc)
        ('Ema', 'Williams', 'Williams Coffee Co.', 'ema@williamscoffee.com',
         '+1-555-0142', 'Canada', 'CA', 'Active', 'Verified', 'Standard',
         'Food & Beverage'),
        ('John', 'Smith', 'Smith Analytics Inc.', 'john@smithanalytics.com',
         '+1-555-0123', 'USA', 'US', 'Active', 'Verified', 'Enterprise',
         'Software/SaaS'),
        ('Priya', 'Sharma', 'Sharma Boutique', 'priya@sharmaboutique.com',
         '+91-555-0234', 'India', 'IN', 'Restricted', 'Pending', 'Standard',
         'E-commerce - Apparel'),
    ]

    merchant_ids = []
    for row in showcase:
        first, last, biz, email, phone, country, acc_country, status, kyc, plan, mcc = row
        signup = fake.date_between(start_date='-2y', end_date='-30d')
        # Volume scales loosely with plan tier so analytics queries make sense
        volume = round(random.uniform(5_000, 15_000), 2) if plan == 'Standard' else \
                 round(random.uniform(50_000, 200_000), 2) if plan == 'Custom' else \
                 round(random.uniform(500_000, 2_000_000), 2)
        cursor.execute(
            """INSERT INTO merchants (first_name, last_name, business_name, email,
               phone, country, stripe_account_country, account_status, kyc_status,
               subscription_plan, mcc_category, signup_date, monthly_volume_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (first, last, biz, email, phone, country, acc_country, status, kyc,
             plan, mcc, signup, volume),
        )
        merchant_ids.append(cursor.lastrowid)

    for _ in range(count - len(showcase)):
        first = fake.first_name()
        last = fake.last_name()
        biz = fake.company()
        email = f"{first.lower()}.{last.lower()}{random.randint(1, 999)}@{fake.domain_name()}"
        country = fake.country()
        acc_country = fake.country_code()
        plan = random.choices(plans, weights=plan_weights)[0]
        volume = round(random.uniform(1_000, 8_000), 2) if plan == 'Standard' else \
                 round(random.uniform(20_000, 150_000), 2) if plan == 'Custom' else \
                 round(random.uniform(300_000, 3_000_000), 2)

        cursor.execute(
            """INSERT INTO merchants (first_name, last_name, business_name, email,
               phone, country, stripe_account_country, account_status, kyc_status,
               subscription_plan, mcc_category, signup_date, monthly_volume_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (first, last, biz, email, fake.phone_number(), country, acc_country,
             random.choices(statuses, weights=status_weights)[0],
             random.choices(kyc_statuses, weights=kyc_weights)[0],
             plan, random.choice(mccs),
             fake.date_between(start_date='-2y', end_date='-30d'), volume),
        )
        merchant_ids.append(cursor.lastrowid)

    conn.commit()
    print(f"  inserted {len(merchant_ids)} merchants")
    return merchant_ids


def generate_transactions(conn: sqlite3.Connection, merchant_ids: list[int]) -> dict[int, list[int]]:
    """Insert sample transactions for each merchant. Returns merchant_id -> txn_ids."""
    cursor = conn.cursor()
    payment_methods = ['card_visa', 'card_mastercard', 'card_amex', 'ach_debit', 'apple_pay', 'google_pay']
    statuses = ['Succeeded', 'Pending', 'Failed', 'Refunded', 'Partially Refunded']
    status_weights = [0.85, 0.03, 0.05, 0.05, 0.02]

    txn_map: dict[int, list[int]] = {}
    total = 0
    for mid in merchant_ids:
        n_txns = random.randint(8, 25)
        ids = []
        for _ in range(n_txns):
            amount = round(random.choice([
                random.uniform(5, 50),       # small purchases
                random.uniform(50, 500),     # mid
                random.uniform(500, 5000),   # large
            ]), 2)
            fee = round(amount * 0.029 + 0.30, 2)  # Stripe's standard 2.9% + 30 cents
            cursor.execute(
                """INSERT INTO transactions (merchant_id, amount_usd, currency,
                   payment_method, transaction_date, status, fee_usd)
                   VALUES (?, ?, 'USD', ?, ?, ?, ?)""",
                (mid, amount,
                 random.choice(payment_methods),
                 fake.date_time_between(start_date='-180d', end_date='now'),
                 random.choices(statuses, weights=status_weights)[0],
                 fee),
            )
            ids.append(cursor.lastrowid)
            total += 1
        txn_map[mid] = ids

    conn.commit()
    print(f"  inserted {total} transactions")
    return txn_map


def generate_disputes(conn: sqlite3.Connection, merchant_ids: list[int],
                      txn_map: dict[int, list[int]]) -> None:
    """Insert chargeback disputes against merchant transactions."""
    cursor = conn.cursor()
    reasons = ['fraudulent', 'product_not_received', 'duplicate',
               'subscription_canceled', 'credit_not_processed', 'general']
    reason_weights = [0.40, 0.20, 0.05, 0.15, 0.10, 0.10]
    statuses = ['needs_response', 'under_review', 'won', 'lost', 'warning_closed']
    status_weights = [0.10, 0.15, 0.30, 0.35, 0.10]

    total = 0
    for mid in merchant_ids:
        # Most merchants have 0-2 disputes; a long tail of higher-risk ones
        n_disputes = random.choices(
            [0, 1, 2, 3, 5, 8],
            weights=[0.40, 0.30, 0.15, 0.08, 0.05, 0.02],
        )[0]

        for _ in range(n_disputes):
            txn_id = random.choice(txn_map[mid]) if txn_map[mid] else None
            amount = round(random.uniform(10, 2000), 2)
            opened = fake.date_time_between(start_date='-180d', end_date='-7d')
            status = random.choices(statuses, weights=status_weights)[0]

            resolved = None
            evidence_due = opened + timedelta(days=7)  # standard evidence window
            if status in ('won', 'lost', 'warning_closed'):
                resolved = opened + timedelta(days=random.randint(7, 60))

            cursor.execute(
                """INSERT INTO disputes (merchant_id, transaction_id, amount_usd,
                   reason, status, opened_at, resolved_at, evidence_due_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (mid, txn_id, amount,
                 random.choices(reasons, weights=reason_weights)[0],
                 status, opened, resolved, evidence_due),
            )
            total += 1

    conn.commit()
    print(f"  inserted {total} disputes")


def generate_tickets(conn: sqlite3.Connection, merchant_ids: list[int]) -> None:
    """Insert support tickets. Categories cover the realistic surface area
    of a payments platform's inbound queue."""
    cursor = conn.cursor()

    templates = [
        ('Billing', 'Question about Stripe processing fees',
         'Merchant is asking why their effective fee rate seems higher than expected.'),
        ('Billing', 'Disputed monthly subscription charge',
         'Merchant claims they cancelled their Stripe Billing subscription but were still charged.'),
        ('Disputes', 'How do I respond to a chargeback?',
         'Merchant received their first chargeback notification and needs guidance on submitting evidence.'),
        ('Disputes', 'Excessive chargeback ratio warning',
         'Merchant received warning that their chargeback ratio exceeds acceptable thresholds.'),
        ('API/Integration', 'Webhook signatures failing verification',
         'Webhook signatures are not validating in their staging environment.'),
        ('API/Integration', 'PaymentIntent confirmation returning errors',
         'Intermittent errors when calling /v1/payment_intents/confirm.'),
        ('Account', 'KYC documents rejected',
         'Merchant submitted business verification docs but they were marked as failed.'),
        ('Account', 'Account under review - need clarification',
         'Account status changed to Under Review without explanation.'),
        ('Account', 'Request to close Stripe account',
         'Merchant wants to close their Stripe account and asking what happens to remaining balance.'),
        ('Payouts', 'Payout delayed beyond standard schedule',
         'Expected payout did not arrive on standard 2-business-day schedule.'),
        ('Payouts', 'Payout to bank account failed',
         'Bank account rejected the payout - needs to update banking info.'),
        ('Compliance', 'Question about Prohibited Businesses list',
         'Merchant is expanding into new product category and wants to confirm compliance.'),
        ('General', 'Documentation feedback',
         'Suggestions for improving the API documentation around Connect.'),
    ]

    resolutions = [
        'Walked merchant through fee structure. Confirmed accurate billing.',
        'Refunded duplicate subscription charge. Investigation found billing system glitch.',
        'Provided dispute evidence template and submission walkthrough.',
        'Escalated to Risk team. Merchant placed on monitoring plan.',
        'Identified webhook secret mismatch. Issue resolved after secret rotation.',
        'Helped merchant re-submit KYC documents with required clarifications.',
        'Account review completed by Compliance. Returned to Active status.',
        'Confirmed banking info update. Next payout queued for re-attempt.',
        'Reviewed merchant business model against Prohibited Businesses list. Cleared.',
        'Account closure processed per merchant request.',
    ]

    priorities = ['Low', 'Medium', 'High', 'Urgent']
    priority_weights = [0.20, 0.40, 0.30, 0.10]
    statuses = ['Open', 'In Progress', 'Resolved', 'Closed']
    status_weights = [0.15, 0.20, 0.40, 0.25]

    total = 0
    for mid in merchant_ids:
        for _ in range(random.randint(1, 5)):
            category, subject, description = random.choice(templates)
            priority = random.choices(priorities, weights=priority_weights)[0]
            status = random.choices(statuses, weights=status_weights)[0]

            created = fake.date_time_between(start_date='-180d', end_date='now')
            resolved = None
            notes = None
            if status in ('Resolved', 'Closed'):
                resolved = created + timedelta(hours=random.randint(2, 168))
                notes = random.choice(resolutions)

            cursor.execute(
                """INSERT INTO support_tickets
                   (merchant_id, subject, description, category, priority,
                    status, created_at, resolved_at, agent_notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (mid, subject, description, category, priority, status,
                 created, resolved, notes),
            )
            total += 1

    conn.commit()
    print(f"  inserted {total} support tickets")


def main() -> None:
    print(f"\nSeeding database at {SQL_DB_PATH}")
    Path(SQL_DB_PATH).parent.mkdir(exist_ok=True, parents=True)

    conn = sqlite3.connect(SQL_DB_PATH)
    try:
        create_schema(conn)
        merchant_ids = generate_merchants(conn, count=25)
        txn_map = generate_transactions(conn, merchant_ids)
        generate_disputes(conn, merchant_ids, txn_map)
        generate_tickets(conn, merchant_ids)

        cursor = conn.cursor()
        counts = {}
        for table in ("merchants", "transactions", "disputes", "support_tickets"):
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]

        print("\nTotals:")
        for table, n in counts.items():
            print(f"  {table:20s} {n}")
        print()
    finally:
        conn.close()


if __name__ == "__main__":
    main()