# ✅ Final version of data_preparation.py with SKU fix and IsProduction logic

import logging
import pandas as pd
import numpy as np
import os

logger = logging.getLogger(__name__)

def load_csv_tables(csv_dir="data") -> dict:
    table_names = [
        "orders", "order_lines", "products", "customers",
        "regions", "shippers", "suppliers", "shipping_methods", "packs"
    ]
    raw = {}
    for name in table_names:
        file_path = os.path.join(csv_dir, f"{name}.csv")
        if os.path.exists(file_path):
            raw[name] = pd.read_csv(file_path, low_memory=False)
        else:
            raw[name] = pd.DataFrame()
            logger.warning(f"⚠️ Missing table: {name}.csv")
    return raw

def prepare_full_data(raw: dict) -> pd.DataFrame:
    if "orders" not in raw or raw["orders"].empty:
        raise RuntimeError("Missing or empty 'orders'. Cannot continue.")
    if "order_lines" not in raw or raw["order_lines"].empty:
        raise RuntimeError("Missing or empty 'order_lines'. Cannot continue.")

    orders_df = raw["orders"]
    lines_df  = raw["order_lines"]

    def cast(df, col):
        if col not in df.columns:
            raise RuntimeError(f"Expected '{col}' in DataFrame but got {df.columns.tolist()}")
        df[col] = df[col].astype(str)

    for df, cols in [
        (orders_df, ["OrderId", "CustomerId", "SalesRepId", "ShippingMethodRequested"]),
        (lines_df,  ["OrderLineId", "OrderId", "ProductId", "ShipperId"])
    ]:
        for c in cols:
            cast(df, c)

    df = lines_df.merge(orders_df, on="OrderId", how="inner", suffixes=("", "_ord"))
    logger.info(f"After joining orders+lines: {len(df):,} rows")

    lookups = {
        "customers": ("CustomerId", ["CustomerId", "RegionId", "CustomerName"], raw.get("customers")),
        "products": ("ProductId", ["ProductId", "SupplierId", "UnitOfBillingId", "ProductName", "ProductListPrice", "CostPrice", "IsProduction", "SKU"], raw.get("products")),
        "regions": ("RegionId", ["RegionId", "RegionName"], raw.get("regions")),
        "shippers": ("ShipperId", ["ShipperId", "Carrier"], raw.get("shippers")),
        "suppliers": ("SupplierId", ["SupplierId", "SupplierName"], raw.get("suppliers")),
        "smethods": ("ShippingMethodRequested", ["ShippingMethodRequested", "ShippingMethodName"], raw.get("shipping_methods")),
    }

    for name, (keycol, required_cols, lookup_df) in lookups.items():
        if lookup_df is None or lookup_df.empty:
            logger.warning(f"Lookup table '{name}' is missing or empty—skipping merge.")
            continue
        if name == "smethods" and "SMId" in lookup_df.columns:
            lookup_df = lookup_df.rename(columns={"SMId": "ShippingMethodRequested"})
        for col in required_cols:
            cast(lookup_df, col)
        df = df.merge(lookup_df, on=keycol, how="left")
        logger.info(f"After merging '{name}': {len(df):,} rows")

    if "UnitOfBillingId" in df.columns:
        df["UnitOfBillingId"] = df["UnitOfBillingId"].astype(str)
    if "IsProduction" in df.columns:
        df["IsProduction"] = df["IsProduction"].astype(str)
    else:
        df["IsProduction"] = "0"

    packs = raw.get("packs")
    if packs is not None and not packs.empty:
        packs = packs.copy()
        cast(packs, "PickedForOrderLine")
        packs["OrderLineId"] = packs["PickedForOrderLine"]
        psum = packs.groupby("OrderLineId", as_index=False).agg(
            WeightLb=("WeightLb", "sum"),
            ItemCount=("ItemCount", "sum"),
            DeliveryDate=("DeliveryDate", "max")
        )
        cast(psum, "OrderLineId")
        df = df.merge(psum, on="OrderLineId", how="left")
        df[["WeightLb", "ItemCount"]] = df[["WeightLb", "ItemCount"]].fillna(0)
        logger.info(f"After merging 'packs': {len(df):,} rows")
    else:
        df["WeightLb"] = 0.0
        df["ItemCount"] = 0.0
        df["DeliveryDate"] = pd.NaT

    numeric_cols = ["QuantityShipped", "SalePrice", "UnitCost", "WeightLb", "ItemCount"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    per_item = df["WeightLb"] / df["ItemCount"].replace(0, np.nan)
    is_wt = (df["UnitOfBillingId"] == "3") & (df["WeightLb"] > 0)
    df["ShippedWeightLb"] = np.where(is_wt, df["WeightLb"], df["ItemCount"] * per_item.fillna(0))

    df["Revenue"] = np.where(
        df["IsProduction"] != "1",
        df["ShippedWeightLb"] * df["SalePrice"],
        0.0
    )
    df["Cost"] = np.where(
        df["IsProduction"] != "1",
        df["ShippedWeightLb"] * df["UnitCost"],
        0.0
    )
    df["Profit"] = df["Revenue"] - df["Cost"]

    df["Date"] = pd.to_datetime(df["CreatedAt_order"], errors="coerce").dt.normalize()
    df["ShipDate"] = pd.to_datetime(df["ShipDate"], errors="coerce")
    df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"], errors="coerce")
    df["DateExpected"] = pd.to_datetime(df.get("DateExpected"), errors="coerce")
    df["TransitDays"] = (df["DeliveryDate"] - df["ShipDate"]).dt.days.clip(lower=0)
    df["DeliveryStatus"] = np.where(df["DeliveryDate"] <= df["DateExpected"], "On Time", "Late")

    logger.info(f"✅ Prepared full data: {len(df):,} rows")
    return df
