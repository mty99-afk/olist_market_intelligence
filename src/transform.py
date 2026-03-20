from pathlib import Path
import duckdb

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "mart" / "olist.duckdb"
SQL_PATH = BASE_DIR / "sql" / "marts" / "mart_orders.sql"

def create_mart():
    conn = duckdb.connect(str(DB_PATH))

    with open(SQL_PATH, "r", encoding="utf-8") as f:
        sql_query = f.read()

    conn.execute(sql_query)
    conn.close()

    print("Tabla mart_order_items creada correctamente.")

if __name__ == "__main__":
    create_mart()