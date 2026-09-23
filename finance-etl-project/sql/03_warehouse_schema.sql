-- ============================================================================
-- 03_warehouse_schema.sql
--
-- The target model: a small star schema plus two ETL bookkeeping tables.
--
--            dim_customer        dim_merchant
--                   \                /
--                    \              /
--   dim_account -- fact_transactions -- dim_date
--
-- fact_transactions is one row per cleaned, deduplicated transaction.
-- customer_key is denormalized onto the fact table (in addition to going
-- through account_key) because "spend by customer" is such a common query
-- that forcing a join through dim_account every time isn't worth it here.
--

-- ============================================================================

USE finance_dw;

SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS fact_transactions;
DROP TABLE IF EXISTS dim_customer;
DROP TABLE IF EXISTS dim_account;
DROP TABLE IF EXISTS dim_merchant;
DROP TABLE IF EXISTS dim_date;
DROP TABLE IF EXISTS etl_rejects;
DROP TABLE IF EXISTS etl_batch_log;

CREATE TABLE dim_customer (
    customer_key  INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    cust_id       VARCHAR(20)  NOT NULL,
    full_name     VARCHAR(200) NOT NULL,
    email         VARCHAR(200) NULL,
    signup_date   DATE         NULL,
    city          VARCHAR(100) NULL,
    state         VARCHAR(10)  NULL,
    is_active     TINYINT(1)   NOT NULL DEFAULT 1,
    created_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_dim_customer_cust_id (cust_id)
) ENGINE = InnoDB;

CREATE TABLE dim_account (
    account_key   INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    account_id    VARCHAR(20) NOT NULL,
    customer_key  INT UNSIGNED NOT NULL,
    account_type  ENUM('checking', 'savings', 'credit_card') NOT NULL,
    open_date     DATE NULL,
    status        ENUM('active', 'closed') NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_dim_account_account_id (account_id),
    KEY idx_dim_account_customer (customer_key),
    CONSTRAINT fk_account_customer FOREIGN KEY (customer_key)
        REFERENCES dim_customer (customer_key)
) ENGINE = InnoDB;

CREATE TABLE dim_merchant (
    merchant_key   INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    merchant_id    VARCHAR(20)  NOT NULL,
    merchant_name  VARCHAR(200) NOT NULL,
    category       VARCHAR(100) NOT NULL,
    UNIQUE KEY uq_dim_merchant_merchant_id (merchant_id)
) ENGINE = InnoDB;

CREATE TABLE dim_date (
    date_key      INT UNSIGNED PRIMARY KEY,  -- YYYYMMDD
    full_date     DATE NOT NULL,
    year          SMALLINT NOT NULL,
    quarter       TINYINT NOT NULL,
    month         TINYINT NOT NULL,
    month_name    VARCHAR(10) NOT NULL,
    day           TINYINT NOT NULL,
    day_of_week   TINYINT NOT NULL,          -- MySQL DAYOFWEEK(): 1=Sunday .. 7=Saturday
    day_name      VARCHAR(10) NOT NULL,
    is_weekend    TINYINT(1) NOT NULL,
    UNIQUE KEY uq_dim_date_full_date (full_date)
) ENGINE = InnoDB;

CREATE TABLE fact_transactions (
    txn_key       BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    txn_id        VARCHAR(20) NOT NULL,
    account_key   INT UNSIGNED NOT NULL,
    customer_key  INT UNSIGNED NOT NULL,
    merchant_key  INT UNSIGNED NULL,          -- NULL = unknown / unmatched merchant (e.g. ATM)
    date_key      INT UNSIGNED NOT NULL,
    amount        DECIMAL(12, 2) NOT NULL,    -- negative = debit, positive = credit/income
    txn_type      ENUM('debit', 'credit') NOT NULL,
    description   VARCHAR(300) NULL,
    loaded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_fact_txn_id (txn_id),
    KEY idx_fact_account (account_key),
    KEY idx_fact_customer (customer_key),
    KEY idx_fact_merchant (merchant_key),
    KEY idx_fact_date (date_key),
    CONSTRAINT fk_fact_account  FOREIGN KEY (account_key)  REFERENCES dim_account (account_key),
    CONSTRAINT fk_fact_customer FOREIGN KEY (customer_key) REFERENCES dim_customer (customer_key),
    CONSTRAINT fk_fact_merchant FOREIGN KEY (merchant_key) REFERENCES dim_merchant (merchant_key),
    CONSTRAINT fk_fact_date     FOREIGN KEY (date_key)     REFERENCES dim_date (date_key)
) ENGINE = InnoDB;

CREATE TABLE etl_rejects (
    reject_id      BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    batch_run_id    VARCHAR(50)  NULL,
    source_table    VARCHAR(50)  NOT NULL,
    source_pk       VARCHAR(50)  NULL,
    reject_reason   VARCHAR(200) NOT NULL,
    raw_payload      JSON         NULL,
    rejected_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
) ENGINE = InnoDB;

CREATE TABLE etl_batch_log (
    log_id        BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    batch_run_id  VARCHAR(50)  NOT NULL,
    step_name     VARCHAR(100) NOT NULL,
    started_at    TIMESTAMP    NOT NULL,
    finished_at   TIMESTAMP    NULL,
    rows_out      INT          NULL,
    status        VARCHAR(20)  NOT NULL DEFAULT 'SUCCESS'
) ENGINE = InnoDB;

SET FOREIGN_KEY_CHECKS = 1;
