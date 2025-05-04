# data_preparation.py

import logging
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Desired dtypes for each table
CSV_DTYPES = {
    "orders": {
        "OrderId": "category",
        "CustomerId": "category",
        "SalesRepId": "category",
        "ShippingMethodRequested": "category",
        "OrderStatus": "category",
    },
    "order_lines": {
        "OrderLineId": "category",
        "OrderId": "category",
        "ProductId": "category",
        "ShipperId": "category",
        "QuantityShipped": "float32",
        "Price": "float32",
        "CostPrice": "float32",
    },
    "products": {
        "ProductId": "category",
        "SupplierId": "category",
        "UnitOfBillingId": "category",
        "ProductName": "category",
        "ProductListPrice": "float32",
        "CostPrice": "float32",
        "IsProduction": "category",
        "SKU": "category",
    },
    "customers": {
        "CustomerId": "category",
        "RegionId": "category",
        "CustomerName": "category",
    },
    "regions": {
        "RegionId": "category",
        "RegionName": "category",
    },
    "shippers": {
        "ShipperId": "category",
        "Carrier": "category",
    },
    "suppliers": {
        "SupplierId": "category",
        "SupplierName": "category",
    },
    "shipping_methods": {
        "ShippingMethodId": "category",
        "ShippingMethodName": "category",
    },
    "packs": {
        "PickedForOrderLine": "category",
        "WeightLb": "float32",
        "ItemCount": "float32",
    },
}

# Which columns to parse as dates, if present
DATE_COLS = {
    "orders": ["CreatedAt", "DateOrdered", "DateExpected", "DateShipped"],
    "order_lines": ["DateShipped", "CreatedAt"],
    "packs": ["ShippedAt"],
}


def load_csv_tables(csv_dir="data") -> dict[str, pd.DataFrame]:
    table_names = [
        "orders", "order_lines", "products", "customers",
        "regions", "shippers", "suppliers", "shipping_methods", "packs"
    ]
    raw = {}
    for name in table_names:
        path = os.path.join(csv_dir, f"{name}.csv")
        if not os.path.exists(path):
            logger.warning(f"⚠️ Missing table: {name}.csv")
            raw[name] = pd.DataFrame()
            continue

        # 1) Read full CSV
        df = pd.read_csv(path, low_memory=False)

        # 2) Cast dtypes for any columns that exist
        for col, dtype in CSV_DTYPES.get(name, {}).items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"Could not cast {name}.{col} to {dtype}: {e}")

        # 3) Parse dates for any date columns that exist
        for dtcol in DATE_COLS.get(name, []):
            if dtcol in df.columns:
                df[dtcol] = pd.to_datetime(df[dtcol], errors="coerce")

        raw[name] = df

    return raw


def prepare_full_data(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Validate required tables
    if raw.get("orders", pd.DataFrame()).empty:
        raise RuntimeError("Missing or empty 'orders'.")
    if raw.get("order_lines", pd.DataFrame()).empty:
        raise RuntimeError("Missing or empty 'order_lines'.")

    orders = raw["orders"].rename(columns={"CreatedAt": "CreatedAt_order"})
    lines  = raw["order_lines"]

    # Merge orders + lines
    df = pd.merge(
        lines, orders,
        on="OrderId", how="inner", suffixes=("", "_ord")
    )
    logger.info(f"After joining orders+lines: {len(df):,} rows")

    # Lookup tables
    lookups = {
        "customers": ("CustomerId", ["CustomerId", "RegionId", "CustomerName"], raw.get("customers")),
        "products":  ("ProductId", ["ProductId", "SupplierId", "UnitOfBillingId", "ProductName", "SKU", "IsProduction", "ProductListPrice", "CostPrice"], raw.get("products")),
        "regions":   ("RegionId", ["RegionId", "RegionName"], raw.get("regions")),
        "shippers":  ("ShipperId", ["ShipperId", "Carrier"], raw.get("shippers")),
        "suppliers": ("SupplierId", ["SupplierId", "SupplierName"], raw.get("suppliers")),
        "smethods":  ("ShippingMethodId", ["ShippingMethodId", "ShippingMethodName"], raw.get("shipping_methods")),
    }

    for lname, (keycol, cols, lookup_df) in lookups.items():
        if lookup_df is None or lookup_df.empty:
            logger.warning(f"Lookup table '{lname}' missing or empty; skipping.")
            continue

        # Rename shipping_methods key
        if lname == "smethods":
            lookup_df = lookup_df.rename(columns={"ShippingMethodId": "ShippingMethodRequested"})

        df = pd.merge(df, lookup_df[cols], left_on=keycol, right_on=cols[0], how="left")
        logger.info(f"After merging '{lname}': {len(df):,} rows")

    # Packs aggregation
    packs = raw.get("packs", pd.DataFrame())
    if not packs.empty:
        packs = packs.rename(columns={"PickedForOrderLine": "OrderLineId"})
        agg = packs.groupby("OrderLineId", as_index=False).agg(
            WeightLb=("WeightLb", "sum"),
            ItemCount=("ItemCount", "sum"),
            DeliveryDate=("ShippedAt", "max")
        )
        agg["DeliveryDate"] = pd.to_datetime(agg["DeliveryDate"], errors="coerce")
        df = pd.merge(df, agg, on="OrderLineId", how="left")
    else:
        df["WeightLb"] = 0.0
        df["ItemCount"] = 0.0
        df["DeliveryDate"] = pd.NaT

    # Numeric conversions & business logic
    for col in ["QuantityShipped", "Price", "CostPrice", "WeightLb", "ItemCount"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")

    df["ShippedWeightLb"] = np.where(
        df.get("UnitOfBillingId") == "3",
        df["WeightLb"], df["ItemCount"]
    )
    df["Revenue"] = np.where(
        df.get("IsProduction") != "1",
        df["ShippedWeightLb"] * df.get("Price", 0.0),
        0.0
    )
    df["Cost"] = np.where(
        df.get("IsProduction") != "1",
        df["ShippedWeightLb"] * df.get("CostPrice", 0.0),
        0.0
    )
    df["Profit"] = df["Revenue"] - df["Cost"]

    # Final date columns
    df["Date"] = pd.to_datetime(df["CreatedAt_order"], errors="coerce").dt.normalize()
    df["ShipDate"] = pd.to_datetime(df.get("DateShipped"), errors="coerce")
    df["DateExpected"] = pd.to_datetime(df.get("DateExpected"), errors="coerce")

    df["TransitDays"] = (df["DeliveryDate"] - df["ShipDate"]).dt.days.clip(lower=0).fillna(0).astype("int32")
    df["DeliveryStatus"] = np.where(
        df["DeliveryDate"] <= df["DateExpected"], "On Time", "Late"
    ).astype("category")

    logger.info(f"✅ Prepared full data: {len(df):,} rows")
    return df
