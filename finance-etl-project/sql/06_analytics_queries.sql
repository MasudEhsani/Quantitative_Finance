-- ============================================================================
-- 06_analytics_queries.sql
--
-- Example queries against the finished warehouse -- proof the star schema
-- is actually usable, and a good tour of window functions / CTEs on top of
-- the fact table. Each query is independent; run whichever you like.
-- ============================================================================

USE finance_dw;

-- ----------------------------------------------------------------------------
-- 1. Monthly spend by category (pivoted debits only, i.e. money going out)
-- ----------------------------------------------------------------------------
SELECT
    d.year,
    d.month,
    m.category,
    ROUND(SUM(-f.amount), 2) AS total_spend
FROM fact_transactions f
JOIN dim_date d      ON d.date_key = f.date_key
LEFT JOIN dim_merchant m ON m.merchant_key = f.merchant_key
WHERE f.txn_type = 'debit'
GROUP BY d.year, d.month, m.category
ORDER BY d.year, d.month, total_spend DESC;


-- ----------------------------------------------------------------------------
-- 2. Top 10 merchants by total spend across the whole period
-- ----------------------------------------------------------------------------
SELECT
    m.merchant_name,
    m.category,
    COUNT(*) AS txn_count,
    ROUND(SUM(-f.amount), 2) AS total_spend,
    ROUND(AVG(-f.amount), 2) AS avg_txn_amount
FROM fact_transactions f
JOIN dim_merchant m ON m.merchant_key = f.merchant_key
WHERE f.txn_type = 'debit'
GROUP BY m.merchant_key, m.merchant_name, m.category
ORDER BY total_spend DESC
LIMIT 10;


-- ----------------------------------------------------------------------------
-- 3. Running account balance over time (assumes each account starts at 0
--    and every fact row is a movement -- a simplified ledger view)
-- ----------------------------------------------------------------------------
SELECT
    a.account_id,
    d.full_date,
    f.amount,
    ROUND(
        SUM(f.amount) OVER (
            PARTITION BY a.account_id
            ORDER BY d.full_date, f.txn_key
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ), 2
    ) AS running_balance
FROM fact_transactions f
JOIN dim_account a ON a.account_key = f.account_key
JOIN dim_date d    ON d.date_key    = f.date_key
WHERE a.account_id = 'A00001'   -- swap in any account_id
ORDER BY d.full_date, f.txn_key;


-- ----------------------------------------------------------------------------
-- 4. Month-over-month spend growth (%) per customer, using LAG()
-- ----------------------------------------------------------------------------
WITH monthly_spend AS (
    SELECT
        c.cust_id,
        c.full_name,
        d.year,
        d.month,
        SUM(-f.amount) AS spend
    FROM fact_transactions f
    JOIN dim_customer c ON c.customer_key = f.customer_key
    JOIN dim_date d     ON d.date_key     = f.date_key
    WHERE f.txn_type = 'debit'
    GROUP BY c.cust_id, c.full_name, d.year, d.month
)
SELECT
    cust_id,
    full_name,
    year,
    month,
    ROUND(spend, 2) AS spend,
    ROUND(LAG(spend) OVER (PARTITION BY cust_id ORDER BY year, month), 2) AS prev_month_spend,
    ROUND(
        100 * (spend - LAG(spend) OVER (PARTITION BY cust_id ORDER BY year, month))
        / NULLIF(LAG(spend) OVER (PARTITION BY cust_id ORDER BY year, month), 0)
    , 1) AS pct_change
FROM monthly_spend
ORDER BY cust_id, year, month;


-- ----------------------------------------------------------------------------
-- 5. Signup-month cohorts: average monthly transaction count per customer,
--    by the month they signed up (a simple cohort activity view)
-- ----------------------------------------------------------------------------
WITH cohorts AS (
    SELECT
        customer_key,
        DATE_FORMAT(signup_date, '%Y-%m') AS cohort_month
    FROM dim_customer
    WHERE signup_date IS NOT NULL
),
customer_activity AS (
    SELECT
        customer_key,
        COUNT(*) AS txn_count,
        COUNT(DISTINCT DATE_FORMAT(d.full_date, '%Y-%m')) AS active_months
    FROM fact_transactions f
    JOIN dim_date d ON d.date_key = f.date_key
    GROUP BY customer_key
)
SELECT
    co.cohort_month,
    COUNT(DISTINCT co.customer_key) AS customers_in_cohort,
    ROUND(AVG(ca.txn_count), 1) AS avg_txn_per_customer,
    ROUND(AVG(ca.active_months), 1) AS avg_active_months
FROM cohorts co
LEFT JOIN customer_activity ca ON ca.customer_key = co.customer_key
GROUP BY co.cohort_month
ORDER BY co.cohort_month;


-- ----------------------------------------------------------------------------
-- 6. Spend spike detector: months where a customer's spend was more than
--    2x their trailing 3-month average (flag for review)
-- ----------------------------------------------------------------------------
WITH monthly AS (
    SELECT
        c.cust_id,
        d.year,
        d.month,
        SUM(-f.amount) AS spend
    FROM fact_transactions f
    JOIN dim_customer c ON c.customer_key = f.customer_key
    JOIN dim_date d     ON d.date_key     = f.date_key
    WHERE f.txn_type = 'debit'
    GROUP BY c.cust_id, d.year, d.month
),
with_trailing_avg AS (
    SELECT
        cust_id, year, month, spend,
        AVG(spend) OVER (
            PARTITION BY cust_id
            ORDER BY year, month
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS trailing_3mo_avg
    FROM monthly
)
SELECT cust_id, year, month, ROUND(spend, 2) AS spend, ROUND(trailing_3mo_avg, 2) AS trailing_3mo_avg
FROM with_trailing_avg
WHERE trailing_3mo_avg IS NOT NULL
  AND spend > 2 * trailing_3mo_avg
ORDER BY cust_id, year, month;


-- ----------------------------------------------------------------------------
-- 7. Net cash flow (income - spend) per customer per month
-- ----------------------------------------------------------------------------
SELECT
    c.cust_id,
    c.full_name,
    d.year,
    d.month,
    ROUND(SUM(CASE WHEN f.txn_type = 'credit' THEN f.amount ELSE 0 END), 2) AS income,
    ROUND(SUM(CASE WHEN f.txn_type = 'debit' THEN -f.amount ELSE 0 END), 2) AS spend,
    ROUND(SUM(f.amount), 2) AS net_cash_flow
FROM fact_transactions f
JOIN dim_customer c ON c.customer_key = f.customer_key
JOIN dim_date d      ON d.date_key    = f.date_key
GROUP BY c.cust_id, c.full_name, d.year, d.month
ORDER BY c.cust_id, d.year, d.month;
