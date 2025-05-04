# File: database.py
from dotenv import load_dotenv
from pathlib import Path
import os, datetime, logging
from functools import lru_cache
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

# Load .env
env_path = Path(__file__).parent / ".env"
if env_path.exists(): load_dotenv(env_path)

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_engine():
    srv = os.getenv("DB_SERVER"); db = os.getenv("DB_NAME")
    usr = os.getenv("DB_USER"); pwd = os.getenv("DB_PASS")
    if not all([srv,db,usr,pwd]):
        raise RuntimeError("🚨 DB creds missing! Define DB_SERVER, DB_NAME, DB_USER, DB_PASS.")
    conn = f"mssql+pymssql://{usr}:{pwd}@{srv}/{db}"
    try:
        eng = create_engine(conn, pool_pre_ping=True, pool_size=5, max_overflow=10)
        with eng.connect() as conn: conn.execute(text("SELECT 1"))
        logger.info("✅ Database connected")
        return eng
    except OperationalError as e:
        logger.exception("❌ DB operational error")
        raise RuntimeError(f"🚨 Cannot connect to DB: {e}")

# Chunked fetch helper
def fetch_table(qry, params):
    eng = get_engine()
    chunks = []
    for chunk in pd.read_sql(qry, eng, params=params, chunksize=100_000):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

@lru_cache(maxsize=1)
def fetch_raw_tables(start_date: str = "2020-01-01", end_date: str = None) -> dict:
    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    params = {"start": start_date, "end": end_date}
    queries = {
        "orders": text("""
            SELECT OrderId, CustomerId, SalesRepId,
                   CreatedAt AS CreatedAt, DateOrdered,
                   DateExpected, DateShipped, ShippingMethodRequested
              FROM dbo.Orders
             WHERE OrderStatus='packed'
               AND CreatedAt BETWEEN :start AND :end
        """),
        "order_lines": text("""
            SELECT OrderLineId, OrderId, ProductId, ShipperId,
                   QuantityShipped, Price AS SalePrice,
                   CostPrice AS CostPrice, DateShipped
              FROM dbo.OrderLines
             WHERE CreatedAt BETWEEN :start AND :end
        """),
        # ... other queries unchanged ...
    }
    raw ={}
    for name, qry in queries.items():
        try:
            df = fetch_table(qry, params)
            logger.debug(f"Fetched {name}: {len(df)} rows")
            raw[name] = df
        except SQLAlchemyError as e:
            logger.error(f"Error fetching {name}: {e}")
            raw[name] = pd.DataFrame()
    return raw
