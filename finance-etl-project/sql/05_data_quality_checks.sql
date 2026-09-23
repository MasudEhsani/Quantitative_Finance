-- ============================================================================
-- 05_data_quality_checks.sql
--
-- Post-load validation. Every query is written so that an EMPTY result set
-- (or a count of 0) means "pass". Run this after every load; wire it into
-- the orchestrator (scripts/run_etl.py) so a failing check can stop a
-- pipeline before bad data reaches anything downstream.
-- ============================================================================

USE finance_dw;

-- 1. Reconciliation: how many raw rows became warehouse rows vs. rejects?
--    rows_staged should equal rows_loaded + rows_rejected + rows_deduped_away.
SELECT 'customers' AS entity,
       (SELECT COUNT(*) FROM stg_customers_raw) AS rows_staged,
       (SELECT COUNT(*) FROM dim_customer) AS rows_loaded,
       (SELECT COUNT(*) FROM etl_rejects WHERE source_table = 'stg_customers_raw') AS rows_rejected
UNION ALL
SELECT 'accounts',
       (SELECT COUNT(*) FROM stg_accounts_raw),
       (SELECT COUNT(*) FROM dim_account),
       (SELECT COUNT(*) FROM etl_rejects WHERE source_table = 'stg_accounts_raw')
UNION ALL
SELECT 'merchants',
       (SELECT COUNT(*) FROM stg_merchants_raw),
       (SELECT COUNT(*) FROM dim_merchant),
       0
UNION ALL
SELECT 'transactions',
       (SELECT COUNT(*) FROM stg_transactions_raw),
       (SELECT COUNT(*) FROM fact_transactions),
       (SELECT COUNT(*) FROM etl_rejects WHERE source_table = 'stg_transactions_raw');

-- 2. PASS/FAIL: duplicate natural keys in the warehouse (should be 0 rows each)
SELECT 'dup_cust_id' AS check_name, cust_id AS offending_value, COUNT(*) AS occurrences
FROM dim_customer GROUP BY cust_id HAVING COUNT(*) > 1;

SELECT 'dup_account_id' AS check_name, account_id AS offending_value, COUNT(*) AS occurrences
FROM dim_account GROUP BY account_id HAVING COUNT(*) > 1;

SELECT 'dup_txn_id' AS check_name, txn_id AS offending_value, COUNT(*) AS occurrences
FROM fact_transactions GROUP BY txn_id HAVING COUNT(*) > 1;

-- 3. PASS/FAIL: orphan foreign keys in the fact table (should be 0 rows)
SELECT 'fact_orphan_account' AS check_name, f.txn_id
FROM fact_transactions f
LEFT JOIN dim_account a ON a.account_key = f.account_key
WHERE a.account_key IS NULL;

SELECT 'fact_orphan_customer' AS check_name, f.txn_id
FROM fact_transactions f
LEFT JOIN dim_customer c ON c.customer_key = f.customer_key
WHERE c.customer_key IS NULL;

SELECT 'fact_orphan_date' AS check_name, f.txn_id
FROM fact_transactions f
LEFT JOIN dim_date d ON d.date_key = f.date_key
WHERE d.date_key IS NULL;

-- 4. PASS/FAIL: required fields that must never be NULL in the fact table
SELECT 'fact_null_amount' AS check_name, COUNT(*) AS offending_rows
FROM fact_transactions WHERE amount IS NULL
HAVING COUNT(*) > 0;

SELECT 'fact_null_date_key' AS check_name, COUNT(*) AS offending_rows
FROM fact_transactions WHERE date_key IS NULL
HAVING COUNT(*) > 0;

-- 5. Sanity range check: any transaction dated outside the expected extract
--    window is worth a second look (could be a parsing bug, e.g. 2-digit
--    year rolling to the wrong century).
SELECT 'txn_date_out_of_range' AS check_name, f.txn_id, d.full_date
FROM fact_transactions f
JOIN dim_date d ON d.date_key = f.date_key
WHERE d.full_date NOT BETWEEN '2024-01-01' AND '2025-06-30';

-- 6. Reject summary -- what got quarantined, and why, grouped and counted.
--    Read this to decide whether upstream data quality is degrading over time.
SELECT source_table, reject_reason, COUNT(*) AS reject_count
FROM etl_rejects
GROUP BY source_table, reject_reason
ORDER BY source_table, reject_count DESC;

-- 7. Latest run summary from the batch log (row counts per step, most recent run)
SELECT step_name, started_at, finished_at,
       TIMESTAMPDIFF(SECOND, started_at, finished_at) AS duration_seconds,
       rows_out, status
FROM etl_batch_log
WHERE batch_run_id = (SELECT MAX(batch_run_id) FROM etl_batch_log)
ORDER BY log_id;
