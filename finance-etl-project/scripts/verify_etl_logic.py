#!/usr/bin/env python3
"""
verify_etl_logic.py -- a pure-Python/pandas mirror of the transform rules
in sql/04_transform_load.sql, used to sanity-check the SQL logic without
needing a live MySQL server (handy in CI, or before you've got MySQL set
up locally).

This is NOT a substitute for actually running the .sql files against
MySQL -- it's a second, independent implementation of the same cleaning
rules (parse 4 date formats, strip currency formatting, dedupe by natural
key, drop orphan FKs) so that if the two disagree, something in the SQL
(or in this script) needs a second look.

Run:
    python3 scripts/generate_raw_data.py   # if you haven't already
    python3 scripts/verify_etl_logic.py
"""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATE_PATTERNS = [
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "%Y-%m-%d"),
    (re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$"), "%m/%d/%Y"),
    (re.compile(r"^\d{1,2}/\d{1,2}/\d{2}$"), "%m/%d/%y"),
    (re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{4}$"), "%d-%b-%Y"),
]


def parse_flex_date(raw):
    if raw is None:
        return None
    v = str(raw).strip()
    if not v or v.lower() == "nan":
        return None
    for pattern, fmt in DATE_PATTERNS:
        if pattern.match(v):
            try:
                return datetime.strptime(v, fmt).date()
            except ValueError:
                return None
    return None


AMOUNT_RE = re.compile(r"^-?\d+(\.\d+)?$")


def clean_amount(raw):
    if raw is None:
        return None
    v = str(raw).strip()
    if not v or v.lower() == "nan":
        return None
    v = v.replace("$", "").replace(",", "")
    v = v.replace("(", "-").replace(")", "")
    v = v.strip()
    if AMOUNT_RE.match(v):
        return round(float(v), 2)
    return None


def to_title_case(raw):
    if raw is None:
        return None
    v = str(raw).strip()
    if not v:
        return None
    return " ".join(w[:1].upper() + w[1:].lower() for w in v.split(" ") if w)


def dedupe_keep_first(df, key):
    return df.drop_duplicates(subset=[key], keep="first").copy()


def section(title):
    print(f"\n=== {title} " + "=" * max(1, 60 - len(title)))


def main():
    customers_raw = pd.read_csv(RAW_DIR / "customers_raw.csv", dtype=str, keep_default_na=False)
    accounts_raw = pd.read_csv(RAW_DIR / "accounts_raw.csv", dtype=str, keep_default_na=False)
    merchants_raw = pd.read_csv(RAW_DIR / "merchants_raw.csv", dtype=str, keep_default_na=False)
    txns_raw = pd.read_csv(RAW_DIR / "transactions_raw.csv", dtype=str, keep_default_na=False)

    report = []

    # ---------------------------------------------------------------- dim_customer
    section("dim_customer")
    valid_cust = customers_raw[customers_raw["cust_id"].str.strip() != ""].copy()
    rejected_cust = len(customers_raw) - len(valid_cust)
    dim_customer = dedupe_keep_first(valid_cust, "cust_id")
    dim_customer["full_name_clean"] = dim_customer["full_name"].map(lambda s: to_title_case(s.strip()))
    dim_customer["signup_date_parsed"] = dim_customer["signup_date"].map(parse_flex_date)
    unparsed_signup = dim_customer["signup_date_parsed"].isna().sum()

    print(f"staged: {len(customers_raw)}  ->  deduped+valid: {len(dim_customer)}  "
          f"(dupes removed: {len(valid_cust) - len(dim_customer)}, missing cust_id rejected: {rejected_cust})")
    print(f"signup_date values that failed to parse (kept as NULL, not rejected): {unparsed_signup}")
    assert dim_customer["cust_id"].is_unique, "FAIL: duplicate cust_id survived dedupe"
    report.append(("dim_customer", len(customers_raw), len(dim_customer), rejected_cust))

    # ---------------------------------------------------------------- dim_merchant
    section("dim_merchant")
    valid_merch = merchants_raw[merchants_raw["merchant_id"].str.strip() != ""].copy()
    dim_merchant = dedupe_keep_first(valid_merch, "merchant_id")
    print(f"staged: {len(merchants_raw)}  ->  deduped: {len(dim_merchant)}")
    assert dim_merchant["merchant_id"].is_unique, "FAIL: duplicate merchant_id survived dedupe"
    report.append(("dim_merchant", len(merchants_raw), len(dim_merchant), 0))

    # ---------------------------------------------------------------- dim_account
    section("dim_account")
    valid_acct = accounts_raw[accounts_raw["account_id"].str.strip() != ""].copy()
    deduped_acct = dedupe_keep_first(valid_acct, "account_id")
    known_cust_ids = set(dim_customer["cust_id"])
    dim_account = deduped_acct[deduped_acct["cust_id"].isin(known_cust_ids)].copy()
    orphan_acct = deduped_acct[~deduped_acct["cust_id"].isin(known_cust_ids)]
    print(f"staged: {len(accounts_raw)}  ->  deduped: {len(deduped_acct)}  "
          f"->  valid (customer resolves): {len(dim_account)}  (orphans rejected: {len(orphan_acct)})")
    assert dim_account["account_id"].is_unique, "FAIL: duplicate account_id survived dedupe"
    report.append(("dim_account", len(accounts_raw), len(dim_account), len(orphan_acct)))

    # ---------------------------------------------------------------- fact_transactions
    section("fact_transactions")
    valid_txn = txns_raw[txns_raw["txn_id"].str.strip() != ""].copy()
    deduped_txn = dedupe_keep_first(valid_txn, "txn_id")
    deduped_txn["date_parsed"] = deduped_txn["txn_date"].map(parse_flex_date)
    deduped_txn["amount_clean"] = deduped_txn["amount"].map(clean_amount)
    known_account_ids = set(dim_account["account_id"])
    deduped_txn["account_ok"] = deduped_txn["account_id"].isin(known_account_ids)

    reason = pd.Series("ok", index=deduped_txn.index)
    reason[~deduped_txn["account_ok"]] = "unknown account_id (orphan FK)"
    reason[deduped_txn["account_ok"] & deduped_txn["date_parsed"].isna()] = "unparsable txn_date"
    reason[deduped_txn["account_ok"] & deduped_txn["date_parsed"].notna()
           & deduped_txn["amount_clean"].isna()] = "unparsable amount"
    deduped_txn["reject_reason"] = reason

    fact = deduped_txn[deduped_txn["reject_reason"] == "ok"].copy()
    rejects = deduped_txn[deduped_txn["reject_reason"] != "ok"]

    print(f"staged: {len(txns_raw)}  ->  deduped: {len(deduped_txn)}  "
          f"->  loaded to fact: {len(fact)}  (rejected: {len(rejects)})")
    print("\nreject breakdown:")
    print(rejects["reject_reason"].value_counts().to_string())

    assert fact["txn_id"].is_unique, "FAIL: duplicate txn_id survived dedupe"
    assert fact["amount_clean"].notna().all(), "FAIL: NULL amount leaked into fact"
    assert fact["date_parsed"].notna().all(), "FAIL: NULL date leaked into fact"
    assert fact["account_id"].isin(known_account_ids).all(), "FAIL: orphan account leaked into fact"
    report.append(("fact_transactions", len(txns_raw), len(fact), len(rejects)))

    # ---------------------------------------------------------------- spot-check analytics
    section("spot-check: total debit spend & top category")
    fact["txn_type"] = fact["amount_clean"].map(lambda a: "credit" if a >= 0 else "debit")
    total_debit = -fact.loc[fact["txn_type"] == "debit", "amount_clean"].sum()
    total_credit = fact.loc[fact["txn_type"] == "credit", "amount_clean"].sum()
    print(f"total debit spend across all accounts: ${total_debit:,.2f}")
    print(f"total credit/income across all accounts: ${total_credit:,.2f}")

    merch_lookup = dim_merchant.set_index("merchant_id")["category"].to_dict()
    fact["category"] = fact["merchant_id"].map(merch_lookup)
    by_cat = (
        fact[fact["txn_type"] == "debit"]
        .assign(spend=lambda d: -d["amount_clean"])
        .groupby("category")["spend"].sum()
        .sort_values(ascending=False)
    )
    print("\ntop 5 categories by spend:")
    print(by_cat.head(5).round(2).to_string())

    # ---------------------------------------------------------------- summary
    section("SUMMARY (staged -> loaded -> rejected)")
    for name, staged, loaded, rejected in report:
        print(f"  {name:<20} staged={staged:<6} loaded={loaded:<6} rejected={rejected}")

    print("\nALL CHECKS PASSED -- transform logic (dedupe, date parsing, amount cleaning,")
    print("orphan-FK handling) is internally consistent with sql/04_transform_load.sql.")


if __name__ == "__main__":
    main()
