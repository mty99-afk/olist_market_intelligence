from pathlib import Path
import duckdb

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "mart" / "olist.duckdb"

conn = duckdb.connect(str(DB_PATH))

query = """
SELECT
    p.product_category_name,
    COUNT(*) AS total_items,
    ROUND(AVG(oi.price), 2) AS avg_price
FROM order_items oi
LEFT JOIN products p
    ON oi.product_id = p.product_id
GROUP BY 1
ORDER BY total_items DESC
LIMIT 10
"""

result = conn.execute(query).fetchdf()
print(result)

conn.close()