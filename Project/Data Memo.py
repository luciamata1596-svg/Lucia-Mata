"ECON-5371 — Data Memo. Data documentation (summary stats + viz)
"
# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# %%
# =============================================================================
# 1. Load the data from Excel File
# =============================================================================
mexico_gdp = pd.read_excel(
    "pibt_cte_valor.xlsx",
    sheet_name="desest-valor",
    header=None
)

mexico_gdp.head()

# %%
# =============================================================================
# 2.Cleaning the data 
# =============================================================================

values = pd.to_numeric(mexico_gdp.iloc[5, 2:])

# Create quarterly dates starting in 1993 Q1
quarters = pd.period_range(
    start="1993Q1",
    periods=len(values),
    freq="Q"
)

# Organize the data
gdp_clean = gdp_clean[
    gdp_clean["Quarter"] <= pd.Period("2025Q4", freq="Q")
].copy() 

# %%
# =============================================================================
# 3. Log transformation to approximate the percentage change in real GDP
# =============================================================================
Log_GDP= np.log(gdp_clean["Real_GDP"])
gdp_clean["Log_Real_GDP"] = np.log(gdp_clean["Real_GDP"])
gdp_clean["GDP_Growth"] = (
    100 * gdp_clean["Log_Real_GDP"].diff()
)

gdp_clean
# %%
# =============================================================================
# 4. Summary of the Data
# =============================================================================
gdp_clean.describe()