# File: data_preparation.py

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
        path = os.path.join(csv_dir, f"{name}.csv")
        if os.path.exists(path):
            raw[name] = pd.read_csv(path, low_memory=False)
        else:
            raw[name] = pd.DataFrame()
            logger.warning(f"⚠️ Missing table: {name}.csv")
    return raw

def prepare_full_data(raw: dict) -> pd.DataFrame:
    # 1) Basic existence checks
    if raw.get("orders", pd.DataFrame()).empty:
        raise RuntimeError("Missing or empty 'orders'.")
    if raw.get("order_lines", pd.DataFrame()).empty:
        raise RuntimeError("Missing or empty 'order_lines'.")

    orders_df   = raw["orders"].copy()
    lines_df    = raw["order_lines"].copy()

    # Helper: cast column to string (or create empty if missing)
    def cast_str(df, col):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)

    # 2) Ensure key columns exist and are strings
    for df_, cols in [
        (orders_df,    ["OrderId", "CustomerId", "SalesRepId", "ShippingMethodRequested"]),
        (lines_df,     ["OrderLineId", "OrderId", "ProductId", "ShipperId"])
    ]:
        for c in cols:
            cast_str(df_, c)

    # 3) Merge orders + lines
    df = lines_df.merge(
        orders_df,
        on="OrderId",
        how="inner",
        suffixes=("", "_ord")
    )
    logger.info(f"After joining orders+lines: {len(df):,} rows")

    # 4) Lookup tables
    lookups = {
        "customers":         ("CustomerId",    ["RegionId","CustomerName"], raw.get("customers", pd.DataFrame())),
        "products":          ("ProductId",     ["SKU","UnitOfBillingId","SupplierId","ProductName","ProductListPrice","CostPrice","IsProduction"], raw.get("products", pd.DataFrame())),
        "regions":           ("RegionId",      ["RegionName"], raw.get("regions", pd.DataFrame())),
        "shippers":          ("ShipperId",     ["Carrier"], raw.get("shippers", pd.DataFrame())),
        "suppliers":         ("SupplierId",    ["SupplierName"], raw.get("suppliers", pd.DataFrame())),
        "shipping_methods":  ("ShippingMethodRequested", ["ShippingMethodName"], raw.get("shipping_methods", pd.DataFrame())),
    }

    for name, (keycol, cols, lkdf) in lookups.items():
        if lkdf is None or lkdf.empty:
            logger.warning(f"Skipping merge for '{name}'—table missing or empty.")
            continue

        # special rename for shipping_methods
        if name == "shipping_methods" and "SMId" in lkdf.columns:
            lkdf = lkdf.rename(columns={"SMId": "ShippingMethodRequested"})

        # only keep those required columns that actually exist
        valid = [c for c in cols if c in lkdf.columns]
        if keycol not in lkdf.columns or not valid:
            logger.warning(f"Skipping merge for '{name}'—key '{keycol}' or required cols not found.")
            continue

        # cast them to string
        cast_str(lkdf, keycol)
        for c in valid:
            cast_str(lkdf, c)

        df = df.merge(
            lkdf[[keycol] + valid],
            on=keycol,
            how="left"
        )
        logger.info(f"After merging '{name}': {len(df):,} rows")

    # 5) Packs aggregation
    packs = raw.get("packs", pd.DataFrame())
    if not packs.empty:
        packs = packs.copy()
        cast_str(packs, "PickedForOrderLine")
        packs["OrderLineId"] = packs["PickedForOrderLine"]
        # only aggregate if columns exist
        agg_kwargs = {}
        if "WeightLb" in packs.columns:
            agg_kwargs["WeightLb"] = ("WeightLb", "sum")
        if "ItemCount" in packs.columns:
            agg_kwargs["ItemCount"] = ("ItemCount", "sum")
        if "DeliveryDate" in packs.columns:
            agg_kwargs["DeliveryDate"] = ("DeliveryDate", "max")

        if agg_kwargs:
            psum = packs.groupby("OrderLineId", as_index=False).agg(**agg_kwargs)
            cast_str(psum, "OrderLineId")
            df = df.merge(psum, on="OrderLineId", how="left")
            # fill defaults
            for col in ["WeightLb","ItemCount"]:
                if col in df.columns:
                    df[col] = df[col].fillna(0.0)
            logger.info(f"After merging 'packs': {len(df):,} rows")
    else:
        df["WeightLb"]    = 0.0
        df["ItemCount"]   = 0.0
        df["DeliveryDate"]= pd.NaT

    # 6) Numeric conversions (fix for numpy.ndarray.fillna error)
    num_cols = ["QuantityShipped","SalePrice","UnitCost","WeightLb","ItemCount"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = pd.Series(df[col]).fillna(0.0)

    # 7) Compute shipped weight & margins
    # fallback weight per item = 1 lb
    df["UnitOfBillingId"] = df.get("UnitOfBillingId", "").astype(str)
    is_wt = (df["UnitOfBillingId"] == "3") & (df["WeightLb"] > 0) & (df["ItemCount"] > 0)
    # average fallback
    if is_wt.any():
        avg_wt = (df.loc[is_wt, "WeightLb"] / df.loc[is_wt, "ItemCount"]).mean()
    else:
        avg_wt = 1.0
    df["ShippedWeightLb"] = np.where(
        is_wt,
        df["WeightLb"],
        df["ItemCount"] * avg_wt
    ).fillna(0.0)

    df["Revenue"] = df["ShippedWeightLb"] * df["SalePrice"]
    df["Cost"]    = df["ShippedWeightLb"] * df["UnitCost"]
    df["Profit"]  = df["Revenue"] - df["Cost"]

    # exclude production items
    if "IsProduction" in df.columns:
        df["IsProduction"] = pd.to_numeric(df["IsProduction"], errors="coerce").fillna(0).astype(int)
        mask = df["IsProduction"] == 1
        df.loc[mask, ["Revenue","Cost","Profit"]] = 0.0
        logger.info(f"Excluded {mask.sum():,} production rows from margin")

    # 8) Final dates and transit/delivery status
    df["Date"]         = pd.to_datetime(df.get("CreatedAt_order"), errors="coerce").dt.normalize()
    df["ShipDate"]     = pd.to_datetime(df.get("ShipDate"), errors="coerce")
    df["DeliveryDate"] = pd.to_datetime(df.get("DeliveryDate"), errors="coerce")
    df["DateExpected"] = pd.to_datetime(df.get("DateExpected"), errors="coerce")
    df["TransitDays"]  = (df["DeliveryDate"] - df["ShipDate"]).dt.days.clip(lower=0)
    df["DeliveryStatus"] = np.where(
        df["DeliveryDate"] <= df["DateExpected"],
        "On Time", "Late"
    )

    logger.info(f"✅ Prepared full data: {len(df):,} rows")
    return df
