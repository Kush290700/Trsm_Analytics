import logging
import os
import pandas as pd
import numpy as np

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
    # 1) Sanity checks
    if raw.get("orders", pd.DataFrame()).empty:
        raise RuntimeError("Missing or empty 'orders'.")
    if raw.get("order_lines", pd.DataFrame()).empty:
        raise RuntimeError("Missing or empty 'order_lines'.")

    orders_df = raw["orders"].copy()
    lines_df  = raw["order_lines"].copy()

    # 2) Ensure key columns exist and cast to string
    def safe_cast(df, col):
        if col not in df.columns:
            df[col] = ""
            logger.warning(f"⚠️ Added missing column '{col}'.")
        df[col] = df[col].astype(str)
    for df, cols in [
        (orders_df,     ["OrderId","CustomerId","SalesRepId","ShippingMethodRequested"]),
        (lines_df,      ["OrderLineId","OrderId","ProductId","ShipperId"])
    ]:
        for c in cols:
            safe_cast(df, c)

    # 3) Join orders + order_lines
    df = lines_df.merge(
        orders_df,
        on="OrderId",
        how="inner",
        suffixes=("", "_ord")
    )
    logger.info(f"After joining orders+lines: {len(df):,} rows")

    # 4) Lookup merges: customers, products, regions, shippers, suppliers, shipping_methods
    lookups = {
        "customers":        ("CustomerId", ["CustomerId","RegionId","CustomerName"], raw.get("customers")),
        "products":         ("ProductId",  ["ProductId","SupplierId","UnitOfBillingId","ProductName","ProductListPrice","CostPrice","IsProduction","SKU"], raw.get("products")),
        "regions":          ("RegionId",   ["RegionId","RegionName"], raw.get("regions")),
        "shippers":         ("ShipperId",  ["ShipperId","Carrier"], raw.get("shippers")),
        "suppliers":        ("SupplierId", ["SupplierId","SupplierName"], raw.get("suppliers")),
        "shipping_methods": ("ShippingMethodRequested", ["ShippingMethodName"], raw.get("shipping_methods"), {"ShippingMethodId":"ShippingMethodRequested"}),
    }

    for name, (keycol, cols, lookup_df, *rename) in lookups.items():
        lookup = lookup_df.copy() if lookup_df is not None else pd.DataFrame()
        if lookup.empty:
            logger.warning(f"Lookup table '{name}' missing or empty; skipping.")
            continue

        # rename columns if provided
        rename_map = rename[0] if rename else {}
        for old, new in rename_map.items():
            if old in lookup.columns:
                lookup = lookup.rename(columns={old:new})

        # ensure key exists on both sides
        if keycol not in lookup.columns or keycol not in df.columns:
            logger.warning(f"Skipping merge '{name}'—key '{keycol}' missing.")
            continue

        # cast and pick only existing columns, never repeating the key twice
        subcols = [keycol] + [c for c in cols if c!=keycol and c in lookup.columns]
        for c in subcols:
            lookup[c] = lookup[c].astype(str)
        df = df.merge(lookup[subcols], on=keycol, how="left")
        logger.info(f"After merging '{name}': {len(df):,} rows")

    # 5) Packs aggregation
    packs = raw.get("packs", pd.DataFrame()).copy()
    if not packs.empty and "PickedForOrderLine" in packs.columns:
        # prepare
        packs["OrderLineId"] = packs["PickedForOrderLine"].astype(str)
        for c in ["WeightLb","ItemCount"]:
            packs[c] = pd.to_numeric(packs.get(c,0), errors="coerce").fillna(0)
        # choose date column if present
        date_col = None
        if "ShippedAt" in packs.columns:
            date_col = "ShippedAt"
        elif "DeliveryDate" in packs.columns:
            date_col = "DeliveryDate"
        if date_col:
            packs[date_col] = pd.to_datetime(packs[date_col], errors="coerce")

        # group
        agg_args = {"WeightLb":"sum","ItemCount":"sum"}
        if date_col:
            agg_args["DeliveryDate"] = (date_col,"max")
        psum = packs.groupby("OrderLineId",as_index=False).agg(**agg_args)
        if "DeliveryDate" in psum.columns:
            psum["DeliveryDate"] = pd.to_datetime(psum["DeliveryDate"], errors="coerce")
        df = df.merge(psum, on="OrderLineId", how="left")
        df["WeightLb"]  = df["WeightLb"].fillna(0)
        df["ItemCount"] = df["ItemCount"].fillna(0)
        logger.info(f"After merging 'packs': {len(df):,} rows")
    else:
        df["WeightLb"]     = 0.0
        df["ItemCount"]    = 0.0
        df["DeliveryDate"] = pd.NaT

    # 6) Numeric casts
    for col in ["QuantityShipped","SalePrice","UnitCost","WeightLb","ItemCount"]:
        if col not in df.columns and col=="SalePrice" and "Price" in df.columns:
            df["SalePrice"] = df["Price"]
        if col not in df.columns and col=="UnitCost" and "CostPrice" in df.columns:
            df["UnitCost"] = df["CostPrice"]
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 7) Compute shipped weight, revenue, cost, profit
    df["ShippedWeightLb"] = np.where(df.get("UnitOfBillingId")=="3", df["WeightLb"], df["ItemCount"])
    df["Revenue"]         = df["ShippedWeightLb"] * df["SalePrice"]
    df["Cost"]            = df["ShippedWeightLb"] * df["UnitCost"]
    df["Profit"]          = df["Revenue"] - df["Cost"]

    # 8) Exclude production items if flagged
    if "IsProduction" in df.columns:
        df["IsProduction"] = pd.to_numeric(df["IsProduction"], errors="coerce").fillna(0).astype(int)
        prod_mask = df["IsProduction"] == 1
        df.loc[prod_mask, ["Revenue","Cost","Profit"]] = 0.0
        logger.info(f"Excluded {prod_mask.sum():,} production items from margin")

    # 9) Final date fields & delivery metrics
    df["Date"]         = pd.to_datetime(df.get("CreatedAt_order"), errors="coerce").dt.normalize()
    df["ShipDate"]     = pd.to_datetime(df.get("ShipDate") or df.get("DateShipped"), errors="coerce")
    df["DeliveryDate"] = pd.to_datetime(df.get("DeliveryDate"), errors="coerce")
    df["DateExpected"] = pd.to_datetime(df.get("DateExpected"), errors="coerce")
    df["TransitDays"]  = (df["DeliveryDate"] - df["ShipDate"]).dt.days.clip(lower=0).fillna(0)
    df["DeliveryStatus"] = np.where(df["DeliveryDate"]<=df["DateExpected"], "On Time", "Late")

    logger.info(f"✅ Prepared full data: {len(df):,} rows")
    return df
