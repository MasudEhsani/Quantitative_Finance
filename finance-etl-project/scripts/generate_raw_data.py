"""
generate_raw_data.py

Generates realistic, INTENTIONALLY MESSY raw source extracts for the
finance ETL project, simulating what you'd actually get from a bank's
core system / card processor / CRM export:

    data/raw/customers_raw.csv
    data/raw/accounts_raw.csv
    data/raw/merchants_raw.csv
    data/raw/transactions_raw.csv

Messiness injected on purpose (this is the whole point of the project --
the transform layer in sql/04_transform_load.sql has to clean this up):
    - inconsistent date formats
    - inconsistent text casing
    - stray whitespace
    - currency symbols / thousands separators / parens-negatives in amounts
    - null / blank fields
    - duplicate rows (exact re-extract dupes)
    - orphan foreign keys (account with no customer, txn with no account)

Deterministic: re-running with the same SEED reproduces the same files.
"""

import csv
import random
from datetime import date, timedelta

SEED = 42
random.seed(SEED)

OUT_DIR = "data/raw"

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Paul", "Ashley",
    "Steven", "Kimberly", "Andrew", "Emily", "Kenneth", "Donna", "Joshua", "Michelle",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
]
CITIES_STATES = [
    ("Austin", "TX"), ("Dallas", "TX"), ("Houston", "TX"), ("Denver", "CO"),
    ("Seattle", "WA"), ("Portland", "OR"), ("Chicago", "IL"), ("Boston", "MA"),
    ("Atlanta", "GA"), ("Miami", "FL"), ("Phoenix", "AZ"), ("San Diego", "CA"),
    ("Sacramento", "CA"), ("Columbus", "OH"), ("Charlotte", "NC"), ("Nashville", "TN"),
]

MERCHANTS = [
    ("Whole Foods Market", "Groceries"), ("Trader Joes", "Groceries"),
    ("Kroger", "Groceries"), ("Safeway", "Groceries"),
    ("Shell Oil", "Gas & Fuel"), ("Chevron", "Gas & Fuel"), ("Exxon", "Gas & Fuel"),
    ("Netflix", "Entertainment"), ("Spotify", "Entertainment"), ("AMC Theatres", "Entertainment"),
    ("Amazon", "Shopping"), ("Target", "Shopping"), ("Walmart", "Shopping"), ("Best Buy", "Shopping"),
    ("Starbucks", "Dining"), ("Chipotle", "Dining"), ("McDonalds", "Dining"), ("Olive Garden", "Dining"),
    ("Delta Air Lines", "Travel"), ("Marriott", "Travel"), ("Uber", "Travel"), ("Airbnb", "Travel"),
    ("AT&T", "Utilities"), ("Comcast Xfinity", "Utilities"), ("Pacific Gas & Electric", "Utilities"),
    ("Blue Cross Blue Shield", "Healthcare"), ("CVS Pharmacy", "Healthcare"), ("Walgreens", "Healthcare"),
    ("Planet Fitness", "Fitness"), ("24 Hour Fitness", "Fitness"),
    ("Employer Payroll Inc", "Income"),
]

ACCOUNT_TYPES = ["checking", "savings", "credit_card"]

DATE_FORMAT_FUNCS = [
    lambda d: d.strftime("%Y-%m-%d"),        # 2024-03-05
    lambda d: d.strftime("%m/%d/%Y"),        # 03/05/2024
    lambda d: d.strftime("%d-%b-%Y"),        # 05-Mar-2024
    lambda d: d.strftime("%m/%d/%y"),        # 03/05/24
]


def messy_case(s):
    r = random.random()
    if r < 0.25:
        return s.upper()
    if r < 0.5:
        return s.lower()
    return s


def maybe_pad(s):
    if random.random() < 0.15:
        return f"  {s}  "
    return s


def random_date(start, end):
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def fmt_date(d, allow_blank=True, blank_rate=0.03):
    if allow_blank and random.random() < blank_rate:
        return ""
    fmt = random.choice(DATE_FORMAT_FUNCS)
    return fmt(d)


def gen_customers(n=60):
    rows = []
    for i in range(1, n + 1):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        name = f"{first} {last}"
        city, state = random.choice(CITIES_STATES)
        signup = random_date(date(2019, 1, 1), date(2024, 6, 30))
        email = f"{first.lower()}.{last.lower()}{i}@example.com"
        if random.random() < 0.06:
            email = ""  # missing email
        is_active = random.choices(["Y", "N", "1", "0", "true", "false"], weights=[30, 3, 5, 1, 5, 1])[0]
        row = {
            "cust_id": f"C{i:04d}",
            "full_name": maybe_pad(messy_case(name)),
            "email": email,
            "signup_date": fmt_date(signup, blank_rate=0.02),
            "city": messy_case(city) if random.random() > 0.05 else "",
            "state": state,
            "is_active": is_active,
        }
        rows.append(row)

    # Inject exact duplicate rows (simulates a re-run extract landing twice)
    dupes = random.sample(rows, k=max(1, n // 15))
    rows.extend(dupes)
    random.shuffle(rows)
    return rows


def gen_accounts(customers, n=90):
    cust_ids = [c["cust_id"] for c in customers]
    rows = []
    for i in range(1, n + 1):
        cust_id = random.choice(cust_ids)
        open_d = random_date(date(2019, 1, 1), date(2024, 12, 1))
        status = random.choices(
            ["Active", "active", "ACTIVE", "Closed", "closed", "CLOSED"],
            weights=[40, 10, 5, 8, 3, 2],
        )[0]
        rows.append({
            "account_id": f"A{i:05d}",
            "cust_id": cust_id,
            "account_type": messy_case(random.choice(ACCOUNT_TYPES)),
            "open_date": fmt_date(open_d, blank_rate=0.02),
            "status": status,
        })

    # A few orphan accounts referencing a customer that doesn't exist
    # (simulates late-arriving / deleted customer records upstream)
    for i in range(n + 1, n + 4):
        rows.append({
            "account_id": f"A{i:05d}",
            "cust_id": f"C{9000 + i}",
            "account_type": random.choice(ACCOUNT_TYPES),
            "open_date": fmt_date(random_date(date(2020, 1, 1), date(2024, 1, 1))),
            "status": "Active",
        })

    dupes = random.sample(rows[:n], k=max(1, n // 20))
    rows.extend(dupes)
    random.shuffle(rows)
    return rows


def gen_merchants():
    rows = []
    for i, (name, category) in enumerate(MERCHANTS, start=1):
        rows.append({
            "merchant_id": f"M{i:03d}",
            "merchant_name": maybe_pad(messy_case(name)),
            "category": category,
        })
    return rows


def fmt_amount(amount, is_income):
    """Return amount as a messy string, matching real-world exports."""
    r = random.random()
    val = f"{abs(amount):,.2f}"
    if is_income:
        s = f"{val}"
    else:
        if r < 0.2:
            s = f"({val})"       # accounting-style negative
        elif r < 0.35:
            s = f"-{val}"
        else:
            s = f"-{val}"
    if random.random() < 0.3:
        s = f"${s}"
    if random.random() < 0.1:
        s = f" {s} "
    return s


def gen_transactions(accounts, merchants, n=3200):
    active_accounts = [a["account_id"] for a in accounts]
    merchant_ids = [m["merchant_id"] for m in merchants]
    income_merchant_id = next(m["merchant_id"] for m in merchants if m["category"] == "Income")

    rows = []
    for i in range(1, n + 1):
        account_id = random.choice(active_accounts)
        txn_date = random_date(date(2024, 1, 1), date(2025, 6, 30))
        is_income = random.random() < 0.06
        if is_income:
            merchant_id = income_merchant_id
            amount_val = round(random.uniform(1800, 5200), 2)
        else:
            merchant_id = random.choice(merchant_ids)
            amount_val = round(random.uniform(3, 420), 2)

        if random.random() < 0.02:
            merchant_id = ""  # missing merchant (e.g. ATM withdrawal / misc debit)

        amount_str = fmt_amount(amount_val, is_income)
        if random.random() < 0.01:
            amount_str = ""  # missing amount -> must be quarantined downstream

        desc = f"{'PAYROLL DEP' if is_income else 'PURCHASE'} {random.randint(100000,999999)}"
        rows.append({
            "txn_id": f"T{i:07d}",
            "account_id": account_id,
            "merchant_id": merchant_id,
            "txn_date": fmt_date(txn_date, blank_rate=0.015),
            "amount": amount_str,
            "description": maybe_pad(desc),
        })

    # Orphan transactions referencing a non-existent account
    for i in range(n + 1, n + 15):
        rows.append({
            "txn_id": f"T{i:07d}",
            "account_id": "A99999",
            "merchant_id": random.choice(merchant_ids),
            "txn_date": fmt_date(random_date(date(2024, 1, 1), date(2025, 6, 30))),
            "amount": fmt_amount(round(random.uniform(5, 100), 2), False),
            "description": "PURCHASE ORPHAN",
        })

    # Duplicate re-extract rows (exact duplicates of real transactions)
    dupes = random.sample(rows[:n], k=max(1, n // 25))
    rows.extend(dupes)
    random.shuffle(rows)
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows):>6} rows -> {path}")


def main():
    customers = gen_customers()
    accounts = gen_accounts(customers)
    merchants = gen_merchants()
    transactions = gen_transactions(accounts, merchants)

    write_csv(f"{OUT_DIR}/customers_raw.csv", customers,
              ["cust_id", "full_name", "email", "signup_date", "city", "state", "is_active"])
    write_csv(f"{OUT_DIR}/accounts_raw.csv", accounts,
              ["account_id", "cust_id", "account_type", "open_date", "status"])
    write_csv(f"{OUT_DIR}/merchants_raw.csv", merchants,
              ["merchant_id", "merchant_name", "category"])
    write_csv(f"{OUT_DIR}/transactions_raw.csv", transactions,
              ["txn_id", "account_id", "merchant_id", "txn_date", "amount", "description"])


if __name__ == "__main__":
    main()
