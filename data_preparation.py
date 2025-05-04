# File: data_preparation.py

import logging
import pandas as pd
import numpy as np
import os

logger = logging.getLogger(__name__)

def load_csv_tables(csv_dir="data") -> dict:
    """Load all CSVs into a dict of DataFrames (empty if missing)."""
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
    """Join, clean, and engineer all tables into one analytics DataFrame."""
    # --- 1) require orders + order_lines ---
    orders_df = raw.get("orders", pd.DataFrame()).copy()
    lines_df  = raw.get("order_lines", pd.DataFrame()).copy()
    if orders_df.empty:
        raise RuntimeError("Missing or empty 'orders'.")
    if lines_df.empty:
        raise RuntimeError("Missing or empty 'order_lines'.")

    def cast_str(df: pd.DataFrame, col: str):
        """Ensure df[col] exists and is string dtype."""
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)

    # cast keys to string
    for df_, cols in [
        (orders_df, ["OrderId","CustomerId","SalesRepId","ShippingMethodRequested"]),
        (lines_df,  ["OrderLineId","OrderId","ProductId","ShipperId"])
    ]:
        for c in cols:
            cast_str(df_, c)

    # --- 2) merge orders + lines ---
    df = lines_df.merge(orders_df, on="OrderId", how="inner", suffixes=("","_ord"))
    logger.info(f"After orders+lines: {len(df):,} rows")

    # --- 3) lookup tables ---
    lookups = {
        "customers":        ("CustomerId",    ["RegionId","CustomerName"], raw.get("customers")),
        "products":         ("ProductId",     ["SKU","UnitOfBillingId","SupplierId","ProductName","ProductListPrice","CostPrice","IsProduction"], raw.get("products")),
        "regions":          ("RegionId",      ["RegionName"], raw.get("regions")),
        "shippers":         ("ShipperId",     ["Carrier"], raw.get("shippers")),
        "suppliers":        ("SupplierId",    ["SupplierName"], raw.get("suppliers")),
        "shipping_methods": ("ShippingMethodRequested", ["ShippingMethodName"], raw.get("shipping_methods")),
    }

    for name, (keycol, cols, lkdf) in lookups.items():
        if lkdf is None or lkdf.empty:
            logger.warning(f"Skipping '{name}'—table empty.")
            continue

        # rename SMId → ShippingMethodRequested if present
        if name=="shipping_methods" and "SMId" in lkdf.columns:
            lkdf = lkdf.rename(columns={"SMId":"ShippingMethodRequested"})

        # restrict to only columns actually present
        valid = [c for c in cols if c in lkdf.columns]
        if keycol not in lkdf.columns or not valid:
            logger.warning(f"Skipping '{name}'—missing key or columns.")
            continue

        cast_str(lkdf, keycol)
        for c in valid:
            cast_str(lkdf, c)

        df = df.merge(lkdf[[keycol] + valid], on=keycol, how="left")
        logger.info(f"After merging '{name}': {len(df):,} rows")

    # --- 4) packs aggregation ---
    packs = raw.get("packs", pd.DataFrame())
    if not packs.empty:
        packs = packs.copy()
        cast_str(packs, "PickedForOrderLine")
        packs["OrderLineId"] = packs["PickedForOrderLine"]

        agg_kwargs = {}
        if "WeightLb" in packs.columns:
            agg_kwargs["WeightLb"] = ("WeightLb","sum")
        if "ItemCount" in packs.columns:
            agg_kwargs["ItemCount"] = ("ItemCount","sum")
        if "DeliveryDate" in packs.columns:
            agg_kwargs["DeliveryDate"] = ("DeliveryDate","max")

        if agg_kwargs:
            psum = packs.groupby("OrderLineId", as_index=False).agg(**agg_kwargs)
            cast_str(psum, "OrderLineId")
            df = df.merge(psum, on="OrderLineId", how="left")
            for col in ["WeightLb","ItemCount"]:
                if col in df.columns:
                    df[col] = df[col].fillna(0.0)
            logger.info(f"After 'packs': {len(df):,} rows")
    else:
        df["WeightLb"]    = 0.0
        df["ItemCount"]   = 0.0
        df["DeliveryDate"]= pd.NaT

    # --- 5) numeric casts ---
    for col in ["QuantityShipped","SalePrice","UnitCost","WeightLb","ItemCount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = pd.Series(df[col]).fillna(0.0)

    # --- 6) shipped weight & margin ---
    df["UnitOfBillingId"] = df.get("UnitOfBillingId","").astype(str)
    is_wt = (df["UnitOfBillingId"]=="3") & (df["WeightLb"]>0) & (df["ItemCount"]>0)

    if is_wt.any():
        avg_wt = (df.loc[is_wt,"WeightLb"] / df.loc[is_wt,"ItemCount"]).mean()
    else:
        avg_wt = 1.0

    # use Series.where so we can fillna()
    df["ShippedWeightLb"] = (
        df["WeightLb"]
          .where(is_wt, df["ItemCount"] * avg_wt)
          .fillna(0.0)
    )

    df["Revenue"] = df["ShippedWeightLb"] * df["SalePrice"]
    df["Cost"]    = df["ShippedWeightLb"] * df["UnitCost"]
    df["Profit"]  = df["Revenue"] - df["Cost"]

    # exclude production items
    if "IsProduction" in df.columns:
        df["IsProduction"] = pd.to_numeric(df["IsProduction"], errors="coerce").fillna(0).astype(int)
        mask = df["IsProduction"] == 1
        df.loc[mask, ["Revenue","Cost","Profit"]] = 0.0
        logger.info(f"Excluded {mask.sum():,} production rows")

    # --- 7) final dates & delivery metrics ---
    df["Date"]         = pd.to_datetime(df.get("CreatedAt_order"), errors="coerce").dt.normalize()
    df["ShipDate"]     = pd.to_datetime(df.get("ShipDate"), errors="coerce")
    df["DeliveryDate"] = pd.to_datetime(df.get("DeliveryDate"), errors="coerce")
    df["DateExpected"] = pd.to_datetime(df.get("DateExpected"), errors="coerce")
    df["TransitDays"]  = (df["DeliveryDate"] - df["ShipDate"]).dt.days.clip(lower=0)
    df["DeliveryStatus"] = np.where(
        df["DeliveryDate"] <= df["DateExpected"], "On Time", "Late"
    )

    logger.info(f"✅ Final prepared data: {len(df):,} rows")
    return df
