import pandas as pd
import numpy as np

def compute_holding_cost(df):
    """
    Advanced holding cost calculation for the inventory DataFrame.
    Expected columns:
      - 'Cost_pr': inventory cost (numeric, cleaned)
      - 'WeightLb': product weight
      - 'OriginDate': datetime
      - 'ItemCount': count of items for each record (if missing, default is 1)
      - 'Location': used to determine if inventory is in "Sharp Base"

    Advanced Cost Components Calculated:
      1. CapitalCost: Investment cost based on cost of capital.
      2. ServiceCost: Insurance/service cost allocated annually.
      3. StorageCost: Warehousing cost allocated by product value; escalates for long-term storage.
      4. RiskCost: Basic risk cost based on inventory value.
      5. ObsolescenceCost: Cost for potential spoilage or obsolescence (capped as a percentage).
      6. SharpBaseExtraCost: Additional cold storage/handling cost for items at "Sharp Base".
      7. TransportationCost: A proxy for transportation expenses (e.g., for moving meat products among facilities).
      8. MarketFluctuationCost: An allowance for market price fluctuations affecting holding costs.
      9. RegulatoryComplianceCost: Costs related to taxes, licenses, and other regulatory requirements.
      
    For items in "Sharp Base", the standard storage cost is set to 0 (since separate costs apply),
    and in addition, a one-time fixed charge is applied per skid.
    The fixed charge is applied when a skid enters and when it comes out. Here we assume a fixed in‑charge 
    of $52 and a fixed out‑charge of $52 per skid, i.e. $104 per skid overall.
    A skid is defined to hold 30 items by default.

    Returns:
      The updated DataFrame with new holding cost components and summary metrics.
    """
    # ---------------------------
    # 1. Time and Inventory Value Calculations
    # ---------------------------
    today = pd.to_datetime("today")
    df['DaysInStorage'] = (today - df['OriginDate']).dt.days
    df['FractionOfYear'] = np.minimum(df['DaysInStorage'] / 365.0, 1.0)
    df['InventoryValue'] = df['Cost_pr']

    total_inventory_value = df['InventoryValue'].sum()
    if total_inventory_value == 0:
        raise ValueError("Total Inventory Value is 0. Check your input data.")
    df['ValueFraction'] = df['InventoryValue'] / total_inventory_value

    # ---------------------------
    # 2. Define Advanced Cost Parameters
    # ---------------------------
    # Capital and service cost parameters:
    cost_of_capital_rate = 0.05                # 5% annual cost of capital
    inventory_service_costs_annual = 102055.0    # Annual service/insurance cost

    # Standard warehouse (storage) cost components:
    rent_total = 71466.0
    rent_warehouse_fraction = 0.40
    rent_warehouse_annual = rent_total * rent_warehouse_fraction

    wholesale_utilities = 107128.0
    retail_utilities = 48280.0
    wholesale_utilities_warehouse = wholesale_utilities * 0.70
    retail_utilities_warehouse = retail_utilities * 0.70

    wages_warehouse = 453626.0
    overheads_total = 544699.0
    overheads_warehouse_fraction = 0.50
    overheads_warehouse_annual = overheads_total * overheads_warehouse_fraction

    storage_space_cost_annual = (rent_warehouse_annual +
                                 wholesale_utilities_warehouse +
                                 retail_utilities_warehouse +
                                 wages_warehouse +
                                 overheads_warehouse_annual)

    risk_cost_rate = 0.03                      # 3% annual risk cost

    # Escalation for long-term storage (for non-Sharp Base items):
    storage_escalation_threshold_days = 365    # Beyond 1 year
    escalated_storage_multiplier = 1.05          # 5% increase in storage cost after threshold

    # Obsolescence cost parameters:
    obsolescence_threshold_days = 180          # After 180 days, obsolescence cost applies
    obsolescence_monthly_rate = 0.02           # 2% per month of inventory value beyond threshold
    max_obsolescence_percent = 0.30            # Capped at 30% of inventory value

    # Sharp Base extra cost parameters:
    default_items_per_skid = 30.0
    monthly_storage_cost_per_skid = 52.0

    # Fixed in/out charges (one-time, applied per skid):
    in_charge_per_skid = 52.0
    out_charge_per_skid = 52.0
    total_fixed_charge_per_skid = in_charge_per_skid + out_charge_per_skid  # Total = $104 per skid

    # Additional cost components:
    transportation_cost_rate = 0.02           # 2% of InventoryValue annually for transportation
    market_fluctuation_cost_rate = 0.015      # 1.5% of InventoryValue annually due to market volatility
    regulatory_compliance_rate = 0.005        # 0.5% of InventoryValue annually for taxes, licenses, and compliance

    # ---------------------------
    # 3. Calculate Standard Cost Components
    # ---------------------------
    df['CapitalCost'] = df['InventoryValue'] * cost_of_capital_rate * df['FractionOfYear']
    df['ServiceCost'] = df['ValueFraction'] * inventory_service_costs_annual * df['FractionOfYear']
    
    # Compute initial storage cost (before escalation)
    df['StorageCost'] = df['ValueFraction'] * storage_space_cost_annual * df['FractionOfYear']
    escalation_mask = df['DaysInStorage'] > storage_escalation_threshold_days
    df.loc[escalation_mask, 'StorageCost'] *= escalated_storage_multiplier

    df['RiskCost'] = df['InventoryValue'] * risk_cost_rate * df['FractionOfYear']

    # ---------------------------
    # 4. Adjust Storage Cost for Sharp Base Items
    # ---------------------------
    mask_sharp = df['Location'].astype(str).str.contains("Sharp Base", case=False, na=False)
    df.loc[mask_sharp, 'StorageCost'] = 0

    # ---------------------------
    # 5. Calculate Obsolescence Cost (Spoilage/Wastage)
    # ---------------------------
    df['ExcessMonths'] = np.maximum((df['DaysInStorage'] - obsolescence_threshold_days) / 30.0, 0)
    df['ObsolescenceCost'] = df['ExcessMonths'] * obsolescence_monthly_rate * df['InventoryValue']
    df['ObsolescenceCost'] = np.minimum(df['ObsolescenceCost'], max_obsolescence_percent * df['InventoryValue'])

    # ---------------------------
    # 6. Calculate Sharp Base Extra Cost
    # ---------------------------
    df['SharpBaseExtraCost'] = 0.0
    if mask_sharp.any():
        sharp_df = df.loc[mask_sharp].copy()
        # Ensure 'ItemCount' is numeric; default to 1 if missing.
        if 'ItemCount' in sharp_df.columns:
            sharp_df['ItemCount'] = pd.to_numeric(sharp_df['ItemCount'], errors='coerce').fillna(1)
        else:
            sharp_df['ItemCount'] = 1

        total_items_sharp = sharp_df['ItemCount'].sum()
        # Determine the number of skids required
        skid_usage = total_items_sharp / default_items_per_skid
        
        # Calculate ongoing monthly storage cost for Sharp Base items.
        monthly_cost_sharp = skid_usage * monthly_storage_cost_per_skid

        # Compute weighted average storage duration (in months)
        weighted_days = (sharp_df['ItemCount'] * sharp_df['DaysInStorage']).sum() / total_items_sharp
        weighted_months = weighted_days / 30.0
        
        # Total ongoing storage cost for Sharp Base items (capped to 12 months)
        total_sharp_cost = monthly_cost_sharp * min(weighted_months, 12)
        
        # Add the fixed in/out charge per skid (applied once per skid)
        total_sharp_cost += skid_usage * total_fixed_charge_per_skid

        total_sharp_inv_value = sharp_df['InventoryValue'].sum()
        if total_sharp_inv_value > 0:
            sharp_df['SharpBaseExtraCost'] = total_sharp_cost * (sharp_df['InventoryValue'] / total_sharp_inv_value)
        else:
            sharp_df['SharpBaseExtraCost'] = 0.0
        df.loc[mask_sharp, 'SharpBaseExtraCost'] = sharp_df['SharpBaseExtraCost']

    # ---------------------------
    # 7. Calculate Additional Costs (Transportation, Market, Regulatory)
    # ---------------------------
    df['TransportationCost'] = df['InventoryValue'] * transportation_cost_rate * df['FractionOfYear']
    df['MarketFluctuationCost'] = df['InventoryValue'] * market_fluctuation_cost_rate * df['FractionOfYear']
    df['RegulatoryComplianceCost'] = df['InventoryValue'] * regulatory_compliance_rate * df['FractionOfYear']

    # ---------------------------
    # 8. Final Aggregation of Holding Costs
    # ---------------------------
    df['TotalHoldingCost'] = (
        df['CapitalCost'] + df['ServiceCost'] + df['StorageCost'] +
        df['RiskCost'] + df['ObsolescenceCost'] + df['SharpBaseExtraCost'] +
        df['TransportationCost'] + df['MarketFluctuationCost'] + df['RegulatoryComplianceCost']
    )
    df['HoldingCostPercent'] = (df['TotalHoldingCost'] / df['InventoryValue']) * 100

    # Clean up temporary columns.
    df.drop(columns=['ExcessMonths'], inplace=True)

    return df
