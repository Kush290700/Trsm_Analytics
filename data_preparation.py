# File: data_preparation.py
import logging
import pandas as pd
import numpy as np
import os

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD & PREPROCESS CSV TABLES WITH DTYPE & DATE OPTIMIZATIONS
# ──────────────────────────────────────────────────────────────────────────────
def load_csv_tables(csv_dir="data") -> dict:
    table_names = [
        "orders", "order_lines", "products", "customers",
        "regions", "shippers", "suppliers", "shipping_methods", "packs"
    ]
    raw = {}
    for name in table_names:
        file_path = os.path.join(csv_dir, f"{name}.csv")
        if not os.path.exists(file_path):
            raw[name] = pd.DataFrame()
            logger.warning(f"⚠️ Missing table: {name}.csv")
            continue

        # Optimize heavy tables by selecting only needed cols, types, and parsing dates
        if name == "orders":
            dtype_map = {
                "OrderId": "string", "CustomerId": "string",
                "SalesRepId": "string", "ShippingMethodRequested": "string"
            }
            parse_dates = ["CreatedAt", "DateExpected", "DateShipped"]
            usecols = [
                "OrderId", "CustomerId", "SalesRepId",
                "CreatedAt", "DateOrdered", "DateExpected",
                "DateShipped", "ShippingMethodRequested", "OrderStatus"
            ]
            raw[name] = pd.read_csv(
                file_path,
                usecols=usecols,
                dtype=dtype_map,
                parse_dates=parse_dates,
                low_memory=False,
            )
        elif name == "order_lines":
            dtype_map = {
                "OrderLineId": "string", "OrderId": "string",
                "ProductId": "string", "ShipperId": "string"
            }
            parse_dates = ["DateShipped"]
            usecols = [
                "OrderLineId", "OrderId", "ProductId", "ShipperId",
                "QuantityShipped", "Price", "CostPrice", "DateShipped"
            ]
            raw[name] = pd.read_csv(
                file_path,
                usecols=usecols,
                dtype=dtype_map,
                parse_dates=parse_dates,
                low_memory=False,
            )
        else:
            raw[name] = pd.read_csv(file_path, low_memory=False)

    return raw


def prepare_full_data(raw: dict) -> pd.DataFrame:
    # Validate core tables
    if raw.get("orders") is None or raw["orders"].empty:
        raise RuntimeError("Missing or empty 'orders'. Cannot continue.")
    if raw.get("order_lines") is None or raw["order_lines"].empty:
        raise RuntimeError("Missing or empty 'order_lines'. Cannot continue.")

    # Cast key ID columns to categorical to save memory
    orders_df = raw["orders"].astype({
        "OrderId": "category", "CustomerId": "category",
        "SalesRepId": "category", "ShippingMethodRequested": "category"
    })
    lines_df = raw["order_lines"].astype({
        "OrderLineId": "category", "OrderId": "category",
        "ProductId": "category", "ShipperId": "category"
    })

    # Merge orders & lines
    df = lines_df.merge(
        orders_df,
        on="OrderId",
        how="inner",
        suffixes=("", "_ord")
    )
    logger.info(f"After joining orders+lines: {len(df):,} rows")

    # Lookup merges for dimension tables
    lookups = {
        "customers": ("CustomerId", ["CustomerId","RegionId","CustomerName"], raw.get("customers")),
        "products": ("ProductId", ["ProductId","SupplierId","UnitOfBillingId","ProductName","ProductListPrice","CostPrice","IsProduction","SKU"], raw.get("products")),
        "regions": ("RegionId", ["RegionId","RegionName"], raw.get("regions")),
        "shippers": ("ShipperId", ["ShipperId","Carrier"], raw.get("shippers")),
        "suppliers": ("SupplierId", ["SupplierId","SupplierName"], raw.get("suppliers")),
        "smethods": ("ShippingMethodRequested", ["ShippingMethodRequested","ShippingMethodName"], raw.get("shipping_methods")),
    }
    for name, (keycol, cols, lookup_df) in lookups.items():
        if lookup_df is None or lookup_df.empty:
            logger.warning(f"Lookup '{name}' missing or empty—skipping.")
            continue
        if name == "smethods" and "ShippingMethodId" in lookup_df.columns:
            lookup_df = lookup_df.rename(columns={"ShippingMethodId":"ShippingMethodRequested"})
        # cast lookup keys & cols
        lookup_df = lookup_df.astype({col: "string" for col in cols if col in lookup_df.columns})
        df = df.merge(lookup_df[cols], on=keycol, how="left")
        logger.info(f"After merging '{name}': {len(df):,} rows")

    # Packs -> aggregated measures
    packs = raw.get("packs")
    if packs is not None and not packs.empty:
        packs = packs.astype({"PickedForOrderLine": "string"})
        packs["OrderLineId"] = packs["PickedForOrderLine"]
        psum = (
            packs
            .groupby("OrderLineId", as_index=False)
            .agg(
                WeightLb=("WeightLb","sum"),
                ItemCount=("ItemCount","sum"),
                DeliveryDate=("DeliveryDate","max")
            )
            .astype({"OrderLineId":"category"})
        )
        df = df.merge(psum, on="OrderLineId", how="left")
        df["WeightLb"] = df["WeightLb"].fillna(0.0)
        df["ItemCount"] = df["ItemCount"].fillna(0.0)
    else:
        df["WeightLb"] = 0.0
        df["ItemCount"] = 0.0
        df["DeliveryDate"] = pd.NaT

    # Numeric downcasting
    for col in ["QuantityShipped","Price","CostPrice","WeightLb","ItemCount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="float").fillna(0.0)

    # Business logic
    df["ShippedWeightLb"] = np.where(df["UnitOfBillingId"] == "3", df["WeightLb"], df["ItemCount"])
    df["Revenue"] = np.where(df.get("IsProduction","0") != "1", df["ShippedWeightLb"] * df["Price"], 0.0)
    df["Cost"]    = np.where(df.get("IsProduction","0") != "1", df["ShippedWeightLb"] * df["CostPrice"], 0.0)
    df["Profit"]  = df["Revenue"] - df["Cost"]

    # Dates & derived
    df["Date"]          = pd.to_datetime(df.get("CreatedAt"), errors="coerce").dt.normalize()
    df["ShipDate"]      = pd.to_datetime(df.get("DateShipped"), errors="coerce")
    df["DeliveryDate"]  = pd.to_datetime(df.get("DeliveryDate"), errors="coerce")
    df["DateExpected"]  = pd.to_datetime(df.get("DateExpected"), errors="coerce")
    df["TransitDays"]   = (df["DeliveryDate"] - df["ShipDate"]).dt.days.clip(lower=0).astype("Int32")
    df["DeliveryStatus"] = np.where(df["DeliveryDate"] <= df["DateExpected"], "On Time", "Late").astype("category")

    # Final downcast
    for col in ["Revenue","Cost","Profit"]:
        df[col] = pd.to_numeric(df[col], downcast="float")

    logger.info(f"✅ Prepared full data: {len(df):,} rows")
    return df
