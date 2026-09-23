# Finance ETL Project (MySQL)

A complete, runnable ETL pipeline built entirely in MySQL: messy raw
financial extracts land in a **staging** layer, get cleaned and validated
in a **transform** layer, and are loaded into a small **star schema**
ready for analytics -- with a quarantine table for anything that fails
validation and a run log for every step.

```
data/raw/*.csv  --(LOAD DATA)-->  staging tables  --(transform)-->  star schema
   (messy)                        (loose VARCHAR)                  (clean, typed)
                                                        \--> etl_rejects (quarantine)
                                                        \--> etl_batch_log (run log)
```

## Project layout

```
finance-etl-project/
  data/
    raw/                    generated messy source CSVs (customers, accounts, merchants, transactions)
  sql/
    00_functions.sql        reusable SQL cleaning functions (date parsing, amount cleaning, title-casing)
    01_staging_schema.sql   loosely-typed landing tables, one per source file
    02_load_staging.sql     LOAD DATA LOCAL INFILE -> staging
    03_warehouse_schema.sql the target star schema + etl_rejects / etl_batch_log
    04_transform_load.sql   clean, validate, dedupe, and load staging -> star schema
    05_data_quality_checks.sql   post-load validation queries (each should return 0 rows / pass)
    06_analytics_queries.sql     example business queries against the finished warehouse
  scripts/
    generate_raw_data.py    regenerates the raw CSVs (deterministic, seeded)
    run_etl.py               orchestrator: runs all the .sql files in order against MySQL
    verify_etl_logic.py       pure-Python/pandas mirror of the transform rules, for sanity-checking without a live DB
  README.md
```

## Why the data is messy on purpose

The raw CSVs are generated (not hand-typed) but deliberately reproduce the
kind of mess a real bank export or card-processor feed actually contains:

- **Inconsistent date formats** in the same column: `2024-03-05`, `03/05/2024`, `05-Mar-2024`, `03/05/24`
- **Inconsistent casing**: `ACTIVE`, `active`, `Active`
- **Currency formatting noise** in amounts: `$1,234.56`, `(45.00)` (accounting-style negative), stray whitespace
- **Missing values**: blank emails, blank dates, blank amounts, blank merchant IDs
- **Duplicate rows**: the same record landing twice (simulates a re-run extract)
- **Orphan foreign keys**: an account referencing a customer that doesn't exist upstream, a transaction referencing an unknown account

Cleaning all of that is the actual point of the project -- `sql/04_transform_load.sql`
is where it happens, and every rule there is commented.

## The warehouse schema (star schema)

```
        dim_customer                dim_merchant
              |                            |
              |                            |
        dim_account -------- fact_transactions -------- dim_date
```

- **dim_customer** -- one row per customer (`cust_id` natural key)
- **dim_account** -- one row per account, FK to `dim_customer`
- **dim_merchant** -- one row per merchant, with a spend `category`
- **dim_date** -- a standard calendar dimension (2019-01-01 .. 2026-12-31), built with a recursive CTE
- **fact_transactions** -- one row per cleaned, deduplicated transaction; `amount` is signed (negative = debit, positive = credit/income), `customer_key` is denormalized onto the fact for convenience
- **etl_rejects** -- every row that failed validation, with a `reject_reason` and the offending raw values as JSON
- **etl_batch_log** -- one row per pipeline step per run, with row counts and timing

## Running it

You need a MySQL 8.0+ server (this was written and reviewed against 8.0
syntax: window functions, recursive CTEs, stored functions, `JSON_OBJECT`).

**1. Generate the raw data** (only needed once, or whenever you want a fresh seed):

```bash
python3 scripts/generate_raw_data.py
```

**2. Run the pipeline.** Two ways:

**Option A -- the `mysql` CLI, step by step** (run from the project root, so the
relative CSV paths in `02_load_staging.sql` resolve):

```bash
mysql --local-infile=1 -u root -p < sql/00_functions.sql
mysql --local-infile=1 -u root -p < sql/01_staging_schema.sql
mysql --local-infile=1 -u root -p < sql/02_load_staging.sql
mysql --local-infile=1 -u root -p < sql/03_warehouse_schema.sql
mysql --local-infile=1 -u root -p < sql/04_transform_load.sql
mysql --local-infile=1 -u root -p < sql/05_data_quality_checks.sql
```

> **Common first-run error:** `ERROR 2068 (HY000): LOAD DATA LOCAL INFILE file
> request rejected`. `local_infile` has to be enabled on both the client
> (`--local-infile=1`, above) **and** the server. On the server, run once:
> `SET GLOBAL local_infile = 1;` (or set `local-infile=1` in `my.cnf`).

**Option B -- the Python orchestrator** (does the same thing, plus prints a
run log with timing and row counts for every step):

```bash
pip install mysql-connector-python
python3 scripts/run_etl.py --host 127.0.0.1 --user root --password yourpassword
# or: MYSQL_HOST=... MYSQL_USER=... MYSQL_PASSWORD=... python3 scripts/run_etl.py
```

**3. Explore the results:**

```bash
mysql -u root -p finance_dw < sql/06_analytics_queries.sql
```

Re-running the whole pipeline is safe -- every load step is idempotent
(`ON DUPLICATE KEY UPDATE`, or a `NOT IN` guard for `dim_date`).

## Verifying without a MySQL server

If you just want to sanity-check the cleaning *logic* (date parsing across
4 formats, amount normalization, dedup, orphan handling) without standing
up MySQL yet, there's a pure-Python/pandas mirror of the same rules:

```bash
python3 scripts/verify_etl_logic.py
```

It loads the same raw CSVs, applies the same rules independently in
Python, asserts there are no duplicate keys / orphans / nulls in the
result, and prints a staged -> loaded -> rejected reconciliation per
table, plus a couple of spot-check numbers (total spend, top categories).
This was run during development against the generated data and passed
cleanly (e.g. `fact_transactions`: 3,342 staged -> 3,036 loaded -> 178
rejected, split across "unknown account_id", "unparsable txn_date", and
"unparsable amount"). It's not a replacement for running the actual `.sql`
files against MySQL -- it's a second, independent check that the rules
agree with themselves.

## Data quality checks (`sql/05_data_quality_checks.sql`)

Run after every load. Each check is written so an **empty result / zero
count means pass**:

1. Reconciliation -- staged vs. loaded vs. rejected row counts per table
2. No duplicate natural keys survived into the warehouse
3. No orphan foreign keys in `fact_transactions`
4. No NULLs in required fact columns
5. No transaction dates outside the expected extract window (catches, e.g., a 2-digit-year rollover bug)
6. Reject summary -- counts by table and reason
7. Latest run's step-by-step log from `etl_batch_log`

## Example analytics (`sql/06_analytics_queries.sql`)

1. Monthly spend by category
2. Top 10 merchants by total spend
3. Running account balance over time (window function)
4. Month-over-month spend growth % per customer (`LAG()`)
5. Signup-month cohort activity
6. Spend-spike detector (current month vs. trailing 3-month average)
7. Net cash flow (income - spend) per customer per month

## Extending this project

Natural next steps if you want to take this further: add a `dim_customer`
**SCD Type 2** history (track address/status changes over time instead of
overwriting), partition `fact_transactions` by month, add a
`fct_daily_account_balance` snapshot table instead of computing running
balance on the fly, or swap `generate_raw_data.py` for a real bank/card
export and adjust the format-detection rules in `sql/00_functions.sql` to
match what actually shows up.
