from pathlib import Path
import duckdb

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "mart" / "olist.duckdb"

conn = duckdb.connect(str(DB_PATH))

query = """
SELECT
    COUNT(*) AS total_rows,
    SUM(
        CASE
            WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1
            ELSE 0
        END
    ) AS late_deliveries,
    ROUND(
        100.0 * SUM(
            CASE
                WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS late_delivery_rate_pct
FROM mart_order_items
WHERE order_delivered_customer_date IS NOT NULL
  AND order_estimated_delivery_date IS NOT NULL
"""

result = conn.execute(query).fetchdf()
print(result)

conn.close()