# File: database.py

from dotenv import load_dotenv
from pathlib import Path
import os
import datetime
import logging
from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError

# ─── Load .env ────────────────────────────────────────────────────────────────
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# ─── Logging setup ────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Engine factory ────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_engine():
    server   = os.getenv("DB_SERVER")
    database = os.getenv("DB_NAME")
    user     = os.getenv("DB_USER")
    pwd      = os.getenv("DB_PASS")

    if not all([server, database, user, pwd]):
        raise RuntimeError(
            "🚨 Database credentials not fully set! "
            "Please define DB_SERVER, DB_NAME, DB_USER & DB_PASS in your environment or .env"
        )

    conn_str = f"mssql+pymssql://{user}:{pwd}@{server}/{database}"

    try:
        engine = create_engine(
            conn_str,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        # quick smoke-test
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection successful")
        return engine

    except OperationalError as oe:
        logger.exception("❌ OperationalError during DB connect")
        raise RuntimeError(f"🚨 Cannot connect to the database: {oe}") from oe

    except Exception as e:
        logger.exception("❌ Unexpected error creating DB engine")
        raise RuntimeError(f"🚨 Unexpected error initializing the DB engine: {e}") from e


# ─── Data fetcher ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=32)
def fetch_raw_tables(start_date: str = "2020-01-01", end_date: str = None) -> dict:
    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")

    engine = get_engine()
    params = {"start": start_date, "end": end_date}

    queries = {
        "orders": text("""
            SELECT OrderId, CustomerId, SalesRepId,
                   CreatedAt AS CreatedAt_order, DateOrdered,
                   DateExpected, DateShipped AS ShipDate,
                   ShippingMethodRequested
              FROM dbo.Orders
             WHERE OrderStatus='packed'
               AND CreatedAt BETWEEN :start AND :end
        """),
        "order_lines": text("""
            SELECT OrderLineId, OrderId, ProductId, ShipperId,
                   QuantityShipped, Price AS SalePrice,
                   CostPrice AS UnitCost, DateShipped
              FROM dbo.OrderLines
             WHERE CreatedAt BETWEEN :start AND :end
        """),
        "customers": text("SELECT CustomerId, Name AS CustomerName, RegionId, IsRetail FROM dbo.Customers"),
        "products": text("""
            SELECT ProductId, SKU, Description AS ProductName,
                   UnitOfBillingId, SupplierId,
                   ListPrice AS ProductListPrice, CostPrice
              FROM dbo.Products
        """),
        "regions": text("SELECT RegionId, Name AS RegionName FROM dbo.Regions"),
        "shippers": text("SELECT ShipperId, Name AS Carrier FROM dbo.Shippers"),
        "shipping_methods": text("SELECT ShippingMethodId AS SMId, Name AS ShippingMethodName FROM dbo.ShippingMethods"),
        "suppliers": text("SELECT SupplierId, Name AS SupplierName FROM dbo.Suppliers"),
        "packs": text("""
            WITH ol AS (
                SELECT OrderLineId
                  FROM dbo.OrderLines
                 WHERE CreatedAt BETWEEN :start AND :end
            )
            SELECT p.PickedForOrderLine, p.WeightLb, p.ItemCount,
                   p.ShippedAt AS DeliveryDate
              FROM dbo.Packs p
              JOIN ol ON p.PickedForOrderLine = ol.OrderLineId
        """),
    }

    # initialize every key to an empty DataFrame
    raw = {name: pd.DataFrame() for name in queries.keys()}

    for name, qry in queries.items():
        try:
            df = pd.read_sql(qry, engine, params=params)
            logger.debug(f"Fetched '{name}': {len(df)} rows")
            raw[name] = df
        except SQLAlchemyError as e:
            logger.error(f"Error fetching '{name}': {e}")
            # raw[name] stays as empty DataFrame

    return raw