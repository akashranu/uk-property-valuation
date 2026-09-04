
CREATE INDEX idx_postcode ON raw_property_sales(postcode);
CREATE INDEX idx_town_city ON raw_property_sales(town_city);
CREATE INDEX idx_transfer_date ON raw_property_sales(transfer_date);
CREATE INDEX idx_type_date ON raw_property_sales(property_type, transfer_date);

CREATE OR REPLACE VIEW vw_property_features AS
SELECT 
    transaction_id,
    price,
    transfer_date,
    YEAR(transfer_date) AS transfer_year,
    QUARTER(transfer_date) AS transfer_quarter,
    MONTH(transfer_date) AS transfer_month,
    postcode,
    TRIM(SUBSTRING_INDEX(postcode, ' ', 1)) AS postcode_district,
    property_type,
    old_new,
    duration,
    town_city,
    district,
    county
FROM raw_property_sales
WHERE 
    price >= 50000    -- filter non-market nominal transfers
    AND price <= 10000000   -- filter ultra-luxury commercial outliers
    AND postcode IS NOT NULL 
    AND postcode != '';