-- ============================================================================
-- 02_load_staging.sql
--
-- Bulk-loads the raw CSV extracts into the staging tables. Truncates first
-- so the script is idempotent (safe to re-run on a fresh drop of the same
-- files, which is exactly what happens on every ETL run in real life).
--
-- Paths are relative to the project root -- run mysql from there, e.g.:
--   mysql --local-infile=1 -u root -p finance_dw < sql/02_load_staging.sql
--
-- ============================================================================

USE finance_dw;

TRUNCATE TABLE stg_customers_raw;
LOAD DATA LOCAL INFILE 'data/raw/customers_raw.csv'
INTO TABLE stg_customers_raw
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(cust_id, full_name, email, signup_date, city, state, is_active)
SET _source_file = 'customers_raw.csv';

TRUNCATE TABLE stg_accounts_raw;
LOAD DATA LOCAL INFILE 'data/raw/accounts_raw.csv'
INTO TABLE stg_accounts_raw
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(account_id, cust_id, account_type, open_date, status)
SET _source_file = 'accounts_raw.csv';

TRUNCATE TABLE stg_merchants_raw;
LOAD DATA LOCAL INFILE 'data/raw/merchants_raw.csv'
INTO TABLE stg_merchants_raw
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(merchant_id, merchant_name, category)
SET _source_file = 'merchants_raw.csv';

TRUNCATE TABLE stg_transactions_raw;
LOAD DATA LOCAL INFILE 'data/raw/transactions_raw.csv'
INTO TABLE stg_transactions_raw
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(txn_id, account_id, merchant_id, txn_date, amount, description)
SET _source_file = 'transactions_raw.csv';

SELECT 'stg_customers_raw'    AS staging_table, COUNT(*) AS row_count FROM stg_customers_raw
UNION ALL
SELECT 'stg_accounts_raw',     COUNT(*) FROM stg_accounts_raw
UNION ALL
SELECT 'stg_merchants_raw',    COUNT(*) FROM stg_merchants_raw
UNION ALL
SELECT 'stg_transactions_raw', COUNT(*) FROM stg_transactions_raw;
