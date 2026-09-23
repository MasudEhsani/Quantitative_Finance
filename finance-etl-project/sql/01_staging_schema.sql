-- ============================================================================
-- 01_staging_schema.sql
--
-- "Landing zone" tables. Every column is a loosely-typed VARCHAR that
-- mirrors the raw CSV exactly -- no parsing, casting, or validation
-- happens here.
-- Run after 00_functions.sql:
--   mysql -u root -p finance_dw < sql/01_staging_schema.sql
-- ============================================================================

CREATE DATABASE IF NOT EXISTS finance_dw CHARACTER SET utf8mb4;
USE finance_dw;

DROP TABLE IF EXISTS stg_customers_raw;
CREATE TABLE stg_customers_raw (
    cust_id      VARCHAR(20),
    full_name    VARCHAR(200),
    email        VARCHAR(200),
    signup_date  VARCHAR(50),
    city         VARCHAR(100),
    state        VARCHAR(10),
    is_active    VARCHAR(10),
    _source_file VARCHAR(200)  DEFAULT NULL,
    _loaded_at   TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
) ENGINE = InnoDB;

DROP TABLE IF EXISTS stg_accounts_raw;
CREATE TABLE stg_accounts_raw (
    account_id   VARCHAR(20),
    cust_id      VARCHAR(20),
    account_type VARCHAR(50),
    open_date    VARCHAR(50),
    status       VARCHAR(20),
    _source_file VARCHAR(200)  DEFAULT NULL,
    _loaded_at   TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
) ENGINE = InnoDB;

DROP TABLE IF EXISTS stg_merchants_raw;
CREATE TABLE stg_merchants_raw (
    merchant_id   VARCHAR(20),
    merchant_name VARCHAR(200),
    category      VARCHAR(100),
    _source_file  VARCHAR(200) DEFAULT NULL,
    _loaded_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
) ENGINE = InnoDB;

DROP TABLE IF EXISTS stg_transactions_raw;
CREATE TABLE stg_transactions_raw (
    txn_id       VARCHAR(20),
    account_id   VARCHAR(20),
    merchant_id  VARCHAR(20),
    txn_date     VARCHAR(50),
    amount       VARCHAR(50),
    description  VARCHAR(300),
    _source_file VARCHAR(200)  DEFAULT NULL,
    _loaded_at   TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
) ENGINE = InnoDB;
