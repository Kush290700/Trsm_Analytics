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
        file_path = os.path.join(csv_dir, f"{name}.csv")
        if os.path.exists(file_path):
            raw[name] = pd.read_csv(file_path, low_memory=False)
        else:
            raw[name] = pd.DataFrame()
            logger.warning(f"⚠️ Missing table: {name}.csv")
    return raw

def prepare_full_data(raw: dict) -> pd.DataFrame:
    # 1) Basic existence checks
    if raw.get("orders", pd.DataFrame()).empty:
        raise RuntimeError("Missing or empty 'orders'. Cannot continue.")
    if raw.get("order_lines", pd.DataFrame()).empty:
        raise RuntimeError("Missing or empty 'order_lines'. Cannot continue.")

    orders_df = raw["orders"].copy()
    lines_df  = raw["order_lines"].copy()

    # 2) Ensure key cols are strings
    def cast_str(df, col):
        if col in df.columns:
            df[col] = df[col].astype(str)
        else:
            df[col] = ""
            logger.warning(f"⚠️ '{col}' not in DataFrame; filling with empty strings.")

    for df_, cols in [
        (orders_df,    ["OrderId","CustomerId","SalesRepId","ShippingMethodRequested"]),
        (lines_df,     ["OrderLineId","OrderId","ProductId","ShipperId"])
    ]:
        for c in cols:
            cast_str(df_, c)

    # 3) Join order_lines + orders
    df = lines_df.merge(
        orders_df,
        on="OrderId",
        how="inner",
        suffixes=("", "_ord")
    )
    logger.info(f"After joining orders+lines: {len(df):,} rows")

    # 4) Lookup merges, only if both sides have the key
    lookups = {
        "customers": ("CustomerId", ["CustomerId","RegionId","CustomerName"], raw.get("customers")),
        "products":  ("ProductId",  [
                           "ProductId",
                           "SKU",                # <-- bring SKU in
                           "SupplierId",
                           "UnitOfBillingId",
                           "ProductName",
                           "ProductListPrice",
                           "CostPrice"
                       ], raw.get("products")),
        "regions":   ("RegionId",   ["RegionId","RegionName"], raw.get("regions")),
        "shippers":  ("ShipperId",  ["ShipperId","Carrier"], raw.get("shippers")),
        "suppliers": ("SupplierId", ["SupplierId","SupplierName"], raw.get("suppliers")),
        "smethods":  ("ShippingMethodRequested", ["ShippingMethodRequested","ShippingMethodName"], raw.get("shipping_methods")),
    }

    for name, (keycol, cols, lookup_df) in lookups.items():
        if lookup_df is None or lookup_df.empty:
            logger.warning(f"Skipping merge '{name}'—lookup table empty.")
            continue
        # rename SMId to ShippingMethodRequested if needed
        if name=="smethods" and "SMId" in lookup_df.columns:
            lookup_df = lookup_df.rename(columns={"SMId":"ShippingMethodRequested"})

        # only merge if df has keycol and lookup_df has keycol
        if keycol not in df.columns or keycol not in lookup_df.columns:
            logger.warning(f"Skipping merge '{name}'—key '{keycol}' missing.")
            continue

        # cast all required cols to string to avoid dtypes mismatch
        for c in cols:
            cast_str(lookup_df, c)

        valid = [c for c in cols if c in lookup_df.columns]
        df = df.merge(
            lookup_df[[keycol] + valid],
            on=keycol,
            how="left"
        )
        logger.info(f"After merging '{name}': {len(df):,} rows")

    # 5) Packs aggregation
    packs = raw.get("packs")
    if packs is not None and not packs.empty:
        packs = packs.copy()
        cast_str(packs, "PickedForOrderLine")
        packs["OrderLineId"] = packs["PickedForOrderLine"]
        # only aggregate existing numeric cols
        agg_dict = {}
        for src, fn in [("WeightLb","sum"), ("ItemCount","sum")]:
            if src in packs.columns:
                agg_dict[src] = (src, fn)
        # use DeliveryDate if present
        if "DeliveryDate" in packs.columns:
            agg_dict["DeliveryDate"] = ("DeliveryDate", "max")

        if agg_dict:
            psum = packs.groupby("OrderLineId", as_index=False).agg(**agg_dict)
            cast_str(psum, "OrderLineId")
            df = df.merge(psum, on="OrderLineId", how="left")
            # fill zeros
            for num in ["WeightLb","ItemCount"]:
                if num in df.columns:
                    df[num] = df[num].fillna(0.0)
            logger.info(f"After merging 'packs': {len(df):,} rows")
    else:
        df["WeightLb"]    = 0.0
        df["ItemCount"]   = 0.0
        df["DeliveryDate"]= pd.NaT

    # 6) Numeric conversion
    num_cols = ["QuantityShipped","SalePrice","UnitCost","WeightLb","ItemCount"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 7) Compute shipped weight & margins
    # fallback weight/item = 1 lb/item
    df["UnitOfBillingId"] = df.get("UnitOfBillingId","").astype(str)
    is_wt = (df["UnitOfBillingId"]=="3") & (df["WeightLb"]>0) & (df["ItemCount"]>0)
    fallback = 1.0
    if is_wt.any():
        fallback = (df.loc[is_wt, "WeightLb"] / df.loc[is_wt, "ItemCount"]).mean()
    df["ShippedWeightLb"] = np.where(
        is_wt,
        df["WeightLb"],
        df["ItemCount"] * fallback
    ).fillna(0.0)

    df["Revenue"] = df["ShippedWeightLb"] * df["SalePrice"]
    df["Cost"]    = df["ShippedWeightLb"] * df["UnitCost"]
    df["Profit"]  = df["Revenue"] - df["Cost"]

    # exclude production items if present
    if "IsProduction" in df.columns:
        prod_mask = pd.to_numeric(df["IsProduction"], errors="coerce").fillna(0).astype(int) == 1
        df.loc[prod_mask, ["Revenue","Cost","Profit"]] = 0.0

    # 8) Final dates & delivery metrics
    df["Date"]         = pd.to_datetime(df.get("CreatedAt_order"), errors="coerce").dt.normalize()
    df["ShipDate"]     = pd.to_datetime(df.get("ShipDate"), errors="coerce")
    df["DeliveryDate"] = pd.to_datetime(df.get("DeliveryDate"), errors="coerce")
    df["DateExpected"] = pd.to_datetime(df.get("DateExpected"), errors="coerce")

    # transit days & status
    df["TransitDays"] = (
        df["DeliveryDate"] - df["ShipDate"]
    ).dt.days.clip(lower=0)
    df["DeliveryStatus"] = np.where(
        df["DeliveryDate"] <= df["DateExpected"],
        "On Time", "Late"
    )

    logger.info(f"✅ Prepared full data: {len(df):,} rows")
    return df
