import logging
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def load_csv_tables(csv_dir: str = "data") -> dict[str, pd.DataFrame]:
    table_names = [
        "orders", "order_lines", "products", "customers",
        "regions", "shippers", "suppliers", "shipping_methods", "packs"
    ]
    raw: dict[str, pd.DataFrame] = {}
    for name in table_names:
        path = os.path.join(csv_dir, f"{name}.csv")
        if os.path.exists(path):
            raw[name] = pd.read_csv(path, low_memory=False)
        else:
            raw[name] = pd.DataFrame()
            logger.warning(f"⚠️ Missing table: {name}.csv")
    return raw

def prepare_full_data(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # 1) Validate core tables
    orders = raw.get("orders", pd.DataFrame()).copy()
    lines  = raw.get("order_lines", pd.DataFrame()).copy()
    if orders.empty:
        raise RuntimeError("Missing or empty 'orders.csv'")
    if lines.empty:
        raise RuntimeError("Missing or empty 'order_lines.csv'")

    # 2) Cast the key columns to str (so merges always line up)
    for df, cols in [
        (orders, ["OrderId","CustomerId","SalesRepId","ShippingMethodRequested"]),
        (lines,  ["OrderLineId","OrderId","ProductId","ShipperId"])
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype(str)
            else:
                raise RuntimeError(f"Expected '{c}' in {df}")

    # 3) Join order_lines ⇄ orders
    df = lines.merge(
        orders,
        on="OrderId",
        how="inner",
        suffixes=("", "_order")
    )
    logger.info(f"After joining orders+lines: {len(df):,} rows")

    # 4) Bring in all of the little lookup tables if they exist
    lookups = {
        "customers":       ("CustomerId",            ["RegionId","CustomerName","IsRetail"], raw.get("customers")),
        "products":        ("ProductId",             ["SKU","ProductName","UnitOfBillingId","SupplierId"], raw.get("products")),
        "regions":         ("RegionId",              ["RegionName"], raw.get("regions")),
        "shippers":        ("ShipperId",             ["Carrier"], raw.get("shippers")),
        "suppliers":       ("SupplierId",            ["SupplierName"], raw.get("suppliers")),
        "shipping_methods":("ShippingMethodRequested",["ShippingMethodName"], raw.get("shipping_methods")),
    }

    for name, (keycol, wanted, lookup_df) in lookups.items():
        if lookup_df is None or lookup_df.empty:
            logger.warning(f"Skipping merge '{name}'—table missing or empty.")
            continue
        lookup_df = lookup_df.copy()
        # rename SMId ➞ ShippingMethodRequested if present
        if name=="shipping_methods" and "SMId" in lookup_df.columns:
            lookup_df = lookup_df.rename(columns={"SMId":"ShippingMethodRequested"})
        # cast key + wanted
        for c in [keycol] + [c for c in wanted if c in lookup_df.columns]:
            lookup_df[c] = lookup_df[c].astype(str)
        # only keep the columns we actually have, drop duplicates
        valid = [keycol] + [c for c in wanted if c in lookup_df.columns]
        sub = lookup_df[valid].drop_duplicates()
        if keycol not in df.columns:
            logger.warning(f"Key '{keycol}' not in main DF—skipping {name}.")
            continue
        df = df.merge(sub, on=keycol, how="left")
        logger.info(f"After merging '{name}': {len(df):,} rows")

    # 5) Packs → weight & item counts & delivery
    packs = raw.get("packs", pd.DataFrame()).copy()
    if not packs.empty and {"PickedForOrderLine","WeightLb","ItemCount","DeliveryDate"}.issubset(packs.columns):
        packs["OrderLineId"] = packs["PickedForOrderLine"].astype(str)
        psum = (
            packs
            .groupby("OrderLineId", as_index=False)
            .agg(
                WeightLb     = ("WeightLb","sum"),
                ItemCount    = ("ItemCount","sum"),
                DeliveryDate = ("DeliveryDate","max")
            )
        )
        psum["OrderLineId"] = psum["OrderLineId"].astype(str)
        df = df.merge(psum, on="OrderLineId", how="left")
        df[["WeightLb","ItemCount"]] = df[["WeightLb","ItemCount"]].fillna(0.0)
    else:
        df["WeightLb"]  = 0.0
        df["ItemCount"] = 0.0
        df["DeliveryDate"] = pd.NaT

    # 6) Numeric conversions
    for col in ["QuantityShipped","SalePrice","UnitCost","WeightLb","ItemCount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 7) *** SQL-style CASE WHEN*** for Revenue/Cost/Profit
    #    — if UnitOfBillingId == 3 use WeightLb, else use ItemCount
    df["Revenue"] = np.where(
        df.get("UnitOfBillingId","") == "3",
        df["WeightLb"] * df["SalePrice"],
        df["ItemCount"] * df["SalePrice"]
    )
    df["Cost"] = np.where(
        df.get("UnitOfBillingId","") == "3",
        df["WeightLb"] * df["UnitCost"],
        df["ItemCount"] * df["UnitCost"]
    )
    df["Profit"] = df["Revenue"] - df["Cost"]

    # 8) Exclude production items if flagged
    if "IsProduction" in df.columns:
        df["IsProduction"] = pd.to_numeric(df["IsProduction"], errors="coerce").fillna(0).astype(int)
        mask = df["IsProduction"] == 1
        df.loc[mask, ["Revenue","Cost","Profit"]] = 0.0
        logger.info(f"Excluded {mask.sum():,} production rows from margin")

    # 9) Final date fields & delivery metrics
    df["Date"]         = pd.to_datetime(df.get("CreatedAt_order"), errors="coerce").dt.normalize()
    df["ShipDate"]     = pd.to_datetime(df.get("ShipDate"),     errors="coerce")
    df["DeliveryDate"] = pd.to_datetime(df.get("DeliveryDate"), errors="coerce")
    df["DateExpected"] = pd.to_datetime(df.get("DateExpected"), errors="coerce")

    df["TransitDays"] = (df["DeliveryDate"] - df["ShipDate"]).dt.days.clip(lower=0)
    df["DeliveryStatus"] = np.where(
        df["DeliveryDate"] <= df["DateExpected"], "On Time", "Late"
    )

    logger.info(f"✅ Final data prepared: {len(df):,} rows | Total Rev=${df['Revenue'].sum():,.2f}")
    return df
