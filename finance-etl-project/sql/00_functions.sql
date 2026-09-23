-- ============================================================================
-- 00_functions.sql
-- Run this file FIRST, e.g.:
--   mysql -u root -p finance_dw < sql/00_functions.sql
-- ============================================================================

CREATE DATABASE IF NOT EXISTS finance_dw CHARACTER SET utf8mb4;
USE finance_dw;

DROP FUNCTION IF EXISTS parse_flex_date;
DROP FUNCTION IF EXISTS clean_amount;
DROP FUNCTION IF EXISTS to_title_case;

DELIMITER $$

-- Parses a date string that may arrive in any of the formats we see from
-- upstream extracts: 2024-03-05 | 03/05/2024 | 05-Mar-2024 | 03/05/24

CREATE FUNCTION parse_flex_date(raw_value VARCHAR(50))
RETURNS DATE
DETERMINISTIC
BEGIN
    DECLARE v VARCHAR(50);
    SET v = TRIM(raw_value);

    IF v IS NULL OR v = '' THEN
        RETURN NULL;
    ELSEIF v REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' THEN
        RETURN STR_TO_DATE(v, '%Y-%m-%d');
    ELSEIF v REGEXP '^[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}$' THEN
        RETURN STR_TO_DATE(v, '%m/%d/%Y');
    ELSEIF v REGEXP '^[0-9]{1,2}/[0-9]{1,2}/[0-9]{2}$' THEN
        RETURN STR_TO_DATE(v, '%m/%d/%y');
    ELSEIF v REGEXP '^[0-9]{1,2}-[A-Za-z]{3}-[0-9]{4}$' THEN
        RETURN STR_TO_DATE(v, '%d-%b-%Y');
    ELSE
        RETURN NULL;
    END IF;
END$$

-- Strips currency symbols, thousands separators, stray whitespace, and

CREATE FUNCTION clean_amount(raw_value VARCHAR(50))
RETURNS DECIMAL(12, 2)
DETERMINISTIC
BEGIN
    DECLARE v VARCHAR(50);
    SET v = TRIM(raw_value);

    IF v IS NULL OR v = '' THEN
        RETURN NULL;
    END IF;

    SET v = REPLACE(v, '$', '');
    SET v = REPLACE(v, ',', '');
    SET v = REPLACE(v, '(', '-');
    SET v = REPLACE(v, ')', '');
    SET v = TRIM(v);

    IF v REGEXP '^-?[0-9]+(\\.[0-9]+)?$' THEN
        RETURN CAST(v AS DECIMAL(12, 2));
    ELSE
        RETURN NULL;
    END IF;
END$$

-- Normalizes ragged casing ("JOHN SMITH", "john smith") to Title Case,

CREATE FUNCTION to_title_case(raw_value VARCHAR(200))
RETURNS VARCHAR(200)
DETERMINISTIC
BEGIN
    DECLARE result VARCHAR(200) DEFAULT '';
    DECLARE remaining VARCHAR(200);
    DECLARE word VARCHAR(200);
    DECLARE space_pos INT;

    SET remaining = TRIM(raw_value);
    IF remaining IS NULL OR remaining = '' THEN
        RETURN remaining;
    END IF;

    WHILE LENGTH(remaining) > 0 DO
        SET space_pos = LOCATE(' ', remaining);
        IF space_pos = 0 THEN
            SET word = remaining;
            SET remaining = '';
        ELSE
            SET word = LEFT(remaining, space_pos - 1);
            SET remaining = SUBSTRING(remaining, space_pos + 1);
        END IF;

        IF LENGTH(word) > 0 THEN
            SET result = CONCAT(result, IF(result = '', '', ' '),
                                 UPPER(LEFT(word, 1)), LOWER(SUBSTRING(word, 2)));
        END IF;
    END WHILE;

    RETURN result;
END$$

DELIMITER ;
