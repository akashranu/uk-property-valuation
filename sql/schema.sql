CREATE TABLE raw_property_sales (
    transaction_id VARCHAR(45) NOT NULL,
    price BIGINT NOT NULL,
    transfer_date DATE NOT NULL,
    postcode VARCHAR(10),
    property_type CHAR(1),
    old_new CHAR(1),
    duration CHAR(1),
    paon VARCHAR(100),
    saon VARCHAR(100),
    street VARCHAR(100),
    locality VARCHAR(100),
    town_city VARCHAR(100),
    district VARCHAR(100),
    county VARCHAR(100),
    ppd_category_type CHAR(1),
    record_status CHAR(1),
    PRIMARY KEY (transaction_id)
);