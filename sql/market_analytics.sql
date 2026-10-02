-- Quarterly Sales Momentum with LAG Window Function
SELECT 
    town_city,
    transfer_year,
    transfer_quarter,
    COUNT(*) AS quarterly_sales_volume,
    ROUND(AVG(price), 2) AS current_qtr_avg_price,
    LAG(ROUND(AVG(price), 2), 1) OVER (
        PARTITION BY town_city 
        ORDER BY transfer_year, transfer_quarter
    ) AS prev_qtr_avg_price,
    ROUND(
        (AVG(price) - LAG(AVG(price), 1) OVER (
            PARTITION BY town_city 
            ORDER BY transfer_year, transfer_quarter
        )) / LAG(AVG(price), 1) OVER (
            PARTITION BY town_city 
            ORDER BY transfer_year, transfer_quarter
        ) * 100, 
        2
    ) AS qoq_growth_pct
FROM vw_property_features
WHERE town_city IN ('LONDON', 'MANCHESTER', 'BIRMINGHAM', 'BRISTOL', 'READING')
GROUP BY town_city, transfer_year, transfer_quarter
ORDER BY town_city, transfer_year, transfer_quarter;

-- Price Quartile Distribution using NTILE
WITH price_buckets AS (
    SELECT 
        town_city,
        price,
        NTILE(4) OVER (PARTITION BY town_city ORDER BY price) AS price_quartile
    FROM vw_property_features
    WHERE town_city IN ('LONDON', 'MANCHESTER', 'BIRMINGHAM', 'BRISTOL', 'READING')
)
SELECT 
    town_city,
    price_quartile,
    COUNT(*) AS total_sales,
    MIN(price) AS min_price,
    MAX(price) AS max_price,
    ROUND(AVG(price), 2) AS avg_quartile_price
FROM price_buckets
GROUP BY town_city, price_quartile
ORDER BY town_city, price_quartile;