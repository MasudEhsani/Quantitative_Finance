-- ============================================================================
-- 04_transform_load.sql
--
-- The transform-and-load step: reads from staging, cleans/validates/dedupes,
-- and loads into the star schema. Every row that fails validation is
-- quarantined into etl_rejects with a reason instead of silently dropped
-- or (worse) crashing the whole load. Every step is timed and row-counted
-- into etl_batch_log so a run has an audit trail.
--
-- Idempotent: safe to re-run against the same staging data (dimension
-- loads use ON DUPLICATE KEY UPDATE; the fact load does too, keyed on
-- txn_id).
--
-- Run after 01_staging_schema.sql, 02_load_staging.sql and 00_functions.sql:
--   mysql -u root -p finance_dw < sql/04_transform_load.sql
-- ============================================================================

USE finance_dw;
SET SESSION cte_max_recursion_depth = 5000;
SET @batch_run_id = DATE_FORMAT(NOW(), '%Y%m%d%H%i%s');


-- ----------------------------------------------------------------------------
-- STEP 0: dim_date -- populate the calendar dimension once (idempotent: only
-- inserts dates not already present). Covers 2019-01-01 .. 2026-12-31, well
-- beyond the transaction date range, so late-arriving data never breaks a join.
-- ----------------------------------------------------------------------------
SET @dim_date_step_start = NOW();

INSERT INTO dim_date (date_key, full_date, year, quarter, month, month_name,
                       day, day_of_week, day_name, is_weekend)
WITH RECURSIVE seq AS (
    SELECT DATE('2019-01-01') AS d
    UNION ALL
    SELECT d + INTERVAL 1 DAY FROM seq WHERE d < '2026-12-31'
)
SELECT
    CAST(DATE_FORMAT(d, '%Y%m%d') AS UNSIGNED),
    d,
    YEAR(d), QUARTER(d), MONTH(d), MONTHNAME(d),
    DAY(d), DAYOFWEEK(d), DAYNAME(d),
    IF(DAYOFWEEK(d) IN (1, 7), 1, 0)
FROM seq
WHERE d NOT IN (SELECT full_date FROM dim_date);

INSERT INTO etl_batch_log (batch_run_id, step_name, started_at, finished_at, rows_out, status)
VALUES (@batch_run_id, 'build_dim_date', @dim_date_step_start, NOW(), ROW_COUNT(), 'SUCCESS');


-- ----------------------------------------------------------------------------
-- STEP 1: dim_customer -- trim/title-case names & cities, normalize the
-- is_active flag from its six different raw spellings, parse signup_date
-- from any of the four raw date formats, and collapse duplicate rows
-- (same cust_id appearing more than once in the extract) to one.
-- ----------------------------------------------------------------------------
SET @step_start = NOW();

INSERT INTO dim_customer (cust_id, full_name, email, signup_date, city, state, is_active)
SELECT
    cust_id,
    to_title_case(TRIM(full_name)),
    NULLIF(TRIM(email), ''),
    parse_flex_date(signup_date),
    NULLIF(to_title_case(TRIM(city)), ''),
    NULLIF(UPPER(TRIM(state)), ''),
    CASE
        WHEN LOWER(TRIM(is_active)) IN ('y', '1', 'true')  THEN 1
        WHEN LOWER(TRIM(is_active)) IN ('n', '0', 'false') THEN 0
        ELSE 1
    END
FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY cust_id ORDER BY _loaded_at) AS rn
    FROM stg_customers_raw
    WHERE cust_id IS NOT NULL AND TRIM(cust_id) <> ''
) d
WHERE rn = 1
ON DUPLICATE KEY UPDATE
    full_name   = VALUES(full_name),
    email       = VALUES(email),
    signup_date = VALUES(signup_date),
    city        = VALUES(city),
    state       = VALUES(state),
    is_active   = VALUES(is_active);

INSERT INTO etl_batch_log (batch_run_id, step_name, started_at, finished_at, rows_out, status)
VALUES (@batch_run_id, 'load_dim_customer', @step_start, NOW(), ROW_COUNT(), 'SUCCESS');

-- quarantine rows with no usable cust_id
INSERT INTO etl_rejects (batch_run_id, source_table, source_pk, reject_reason, raw_payload)
SELECT @batch_run_id, 'stg_customers_raw', COALESCE(cust_id, '(null)'), 'missing cust_id',
       JSON_OBJECT('full_name', full_name, 'email', email)
FROM stg_customers_raw
WHERE cust_id IS NULL OR TRIM(cust_id) = '';

INSERT INTO etl_batch_log (batch_run_id, step_name, started_at, finished_at, rows_out, status)
VALUES (@batch_run_id, 'reject_customers_missing_id', @step_start, NOW(), ROW_COUNT(), 'SUCCESS');


-- ----------------------------------------------------------------------------
-- STEP 2: dim_merchant -- trim names, dedupe by merchant_id.
-- ----------------------------------------------------------------------------
SET @step_start = NOW();

INSERT INTO dim_merchant (merchant_id, merchant_name, category)
SELECT merchant_id, TRIM(merchant_name), TRIM(category)
FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY merchant_id ORDER BY _loaded_at) AS rn
    FROM stg_merchants_raw
    WHERE merchant_id IS NOT NULL AND TRIM(merchant_id) <> ''
) m
WHERE rn = 1
ON DUPLICATE KEY UPDATE
    merchant_name = VALUES(merchant_name),
    category      = VALUES(category);

INSERT INTO etl_batch_log (batch_run_id, step_name, started_at, finished_at, rows_out, status)
VALUES (@batch_run_id, 'load_dim_merchant', @step_start, NOW(), ROW_COUNT(), 'SUCCESS');


-- ----------------------------------------------------------------------------
-- STEP 3: dim_account -- normalize account_type/status casing, parse
-- open_date, dedupe by account_id, and drop (quarantine) any account whose
-- cust_id doesn't resolve to a known customer -- an orphan FK we can't
-- safely attach to the star schema.
-- ----------------------------------------------------------------------------
SET @step_start = NOW();

INSERT INTO dim_account (account_id, customer_key, account_type, open_date, status)
SELECT
    a.account_id,
    c.customer_key,
    LOWER(TRIM(a.account_type)),
    parse_flex_date(a.open_date),
    CASE WHEN LOWER(TRIM(a.status)) = 'closed' THEN 'closed' ELSE 'active' END
FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY _loaded_at) AS rn
    FROM stg_accounts_raw
    WHERE account_id IS NOT NULL AND TRIM(account_id) <> ''
) a
JOIN dim_customer c ON c.cust_id = a.cust_id
WHERE a.rn = 1
ON DUPLICATE KEY UPDATE
    customer_key = VALUES(customer_key),
    account_type = VALUES(account_type),
    open_date    = VALUES(open_date),
    status       = VALUES(status);

INSERT INTO etl_batch_log (batch_run_id, step_name, started_at, finished_at, rows_out, status)
VALUES (@batch_run_id, 'load_dim_account', @step_start, NOW(), ROW_COUNT(), 'SUCCESS');

-- quarantine orphan accounts (cust_id not found in dim_customer)
INSERT INTO etl_rejects (batch_run_id, source_table, source_pk, reject_reason, raw_payload)
SELECT DISTINCT @batch_run_id, 'stg_accounts_raw', a.account_id, 'unknown cust_id (orphan FK)',
       JSON_OBJECT('cust_id', a.cust_id, 'account_type', a.account_type, 'status', a.status)
FROM stg_accounts_raw a
LEFT JOIN dim_customer c ON c.cust_id = a.cust_id
WHERE c.customer_key IS NULL;

INSERT INTO etl_batch_log (batch_run_id, step_name, started_at, finished_at, rows_out, status)
VALUES (@batch_run_id, 'reject_accounts_orphan_customer', @step_start, NOW(), ROW_COUNT(), 'SUCCESS');


-- ----------------------------------------------------------------------------
-- STEP 4: fact_transactions -- the big one. Dedupe by txn_id, clean the
-- amount string (currency symbols / commas / accounting parens), parse the
-- date, then require a resolvable account and a valid date + amount before
-- a row is allowed into the fact table. merchant_id is allowed to be
-- missing/unmatched (merchant_key stays NULL) since not every real-world
-- debit has a merchant (ATM withdrawals, misc fees, etc).
-- ----------------------------------------------------------------------------
SET @step_start = NOW();

DROP TEMPORARY TABLE IF EXISTS tmp_txn_clean;
CREATE TEMPORARY TABLE tmp_txn_clean AS
SELECT
    t.txn_id,
    t.account_id,
    NULLIF(TRIM(t.merchant_id), '') AS merchant_id,
    parse_flex_date(t.txn_date)     AS txn_date_parsed,
    clean_amount(t.amount)          AS amount_clean,
    NULLIF(TRIM(t.description), '') AS description_clean
FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY txn_id ORDER BY _loaded_at) AS rn
    FROM stg_transactions_raw
    WHERE txn_id IS NOT NULL AND TRIM(txn_id) <> ''
) t
WHERE t.rn = 1;

INSERT INTO fact_transactions
    (txn_id, account_key, customer_key, merchant_key, date_key, amount, txn_type, description)
SELECT
    tc.txn_id,
    da.account_key,
    da.customer_key,
    dm.merchant_key,
    dd.date_key,
    tc.amount_clean,
    IF(tc.amount_clean >= 0, 'credit', 'debit'),
    tc.description_clean
FROM tmp_txn_clean tc
JOIN dim_account da ON da.account_id = tc.account_id
JOIN dim_date dd    ON dd.full_date  = tc.txn_date_parsed
LEFT JOIN dim_merchant dm ON dm.merchant_id = tc.merchant_id
WHERE tc.txn_date_parsed IS NOT NULL
  AND tc.amount_clean IS NOT NULL
ON DUPLICATE KEY UPDATE
    account_key  = VALUES(account_key),
    customer_key = VALUES(customer_key),
    merchant_key = VALUES(merchant_key),
    date_key     = VALUES(date_key),
    amount       = VALUES(amount),
    txn_type     = VALUES(txn_type),
    description  = VALUES(description);

INSERT INTO etl_batch_log (batch_run_id, step_name, started_at, finished_at, rows_out, status)
VALUES (@batch_run_id, 'load_fact_transactions', @step_start, NOW(), ROW_COUNT(), 'SUCCESS');

-- quarantine everything that couldn't make it into the fact table, with the reason
INSERT INTO etl_rejects (batch_run_id, source_table, source_pk, reject_reason, raw_payload)
SELECT
    @batch_run_id,
    'stg_transactions_raw',
    tc.txn_id,
    CASE
        WHEN da.account_key IS NULL       THEN 'unknown account_id (orphan FK)'
        WHEN tc.txn_date_parsed IS NULL   THEN 'unparsable txn_date'
        WHEN tc.amount_clean IS NULL      THEN 'unparsable amount'
    END,
    JSON_OBJECT('account_id', tc.account_id, 'merchant_id', tc.merchant_id)
FROM tmp_txn_clean tc
LEFT JOIN dim_account da ON da.account_id = tc.account_id
WHERE da.account_key IS NULL
   OR tc.txn_date_parsed IS NULL
   OR tc.amount_clean IS NULL;

INSERT INTO etl_batch_log (batch_run_id, step_name, started_at, finished_at, rows_out, status)
VALUES (@batch_run_id, 'reject_transactions_invalid', @step_start, NOW(), ROW_COUNT(), 'SUCCESS');

DROP TEMPORARY TABLE IF EXISTS tmp_txn_clean;

SELECT @batch_run_id AS batch_run_id, step_name, started_at, finished_at, rows_out, status
FROM etl_batch_log
WHERE batch_run_id = @batch_run_id
ORDER BY log_id;
