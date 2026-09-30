"ECON-5371 — Data Memo. Data documentation (summary stats + viz)
"
# %%
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# %%
# =============================================================================
# 1. Load the data from Excel File
# =============================================================================
mexico_gdp = pd.read_excel(
    "Project/data/pibt_cte_valor.xlsx",
    sheet_name="desest-valor",
    header=None
)

mexico_gdp.head()
# %%
# =============================================================================
# 2.Cleaning the data 
# =============================================================================
values = pd.to_numeric(
    mexico_gdp.iloc[6, 2:],
    errors="coerce"
).dropna()

# The data begin in 1993 Q1
quarters = pd.period_range(
    start="1993Q1",
    periods=len(values),
    freq="Q"
)

gdp_clean = pd.DataFrame({
    "Quarter": quarters,
    "Real_GDP": values.to_numpy()
})

gdp_clean.head()
# %%
# =============================================================================
# 3. Log transformation to approximate the percentage change in real GDP
# =============================================================================
gdp_clean["Log_Real_GDP"] = np.log(
    gdp_clean["Real_GDP"]
)

gdp_clean["GDP_Growth"] = (
    100 * gdp_clean["Log_Real_GDP"].diff()
)

gdp_clean.head()
# %%
# =============================================================================
# 4. Summary of the Data
# =============================================================================
gdp_clean.describe()
# %%
# =============================================================================
# 4. Plot of Mexico's Seasonally Adjusted Quarterly Real GDP"
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
gdp_clean["Date"] = gdp_clean["Quarter"].dt.to_timestamp()
ax.plot(gdp_clean["Date"], gdp_clean["Real_GDP"], color="darkblue", linewidth=1.6)
ax.set_title("Mexico's Seasonally Adjusted Quarterly Real GDP")
ax.set_xlabel("Year")
ax.set_ylabel("Millions of pesos at 2018 price")
fig.tight_layout()
plt.show()