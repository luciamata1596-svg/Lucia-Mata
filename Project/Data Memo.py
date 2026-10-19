# ECON-5371 — Data Memo
# Data documentation, summary statistics, and visualization

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
# =============================================================================
# 1.Import the Excel File
# =============================================================================

mexico_gdp = pd.read_excel(
    "Project/data/pibt_cte_valor.xlsx",
    sheet_name="desest-valor",
    header=None
)

mexico_gdp.head()


# %%
# =============================================================================
# 2. Clean the data
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
# 3. Limit the research sample to 1993 Q1–2025 Q4
# =============================================================================

analysis_data = gdp_clean[
    gdp_clean["Quarter"] <= pd.Period("2025Q4", freq="Q")
].copy()
# %%
# =============================================================================
# 4. Calculate quarterly real GDP growth
# =============================================================================

analysis_data["Log_Real_GDP"] = np.log(
    analysis_data["Real_GDP"]
)

analysis_data["GDP_Growth"] = (
    100 * analysis_data["Log_Real_GDP"].diff()
)

analysis_data.head()


# %%
# =============================================================================
# 5. Summary statistics
# =============================================================================

table = analysis_data[
    ["Real_GDP", "Log_Real_GDP", "GDP_Growth"]
].describe().round(6)

print(table)

table.to_excel(
    "Project/summary_statistics.xlsx"
)

# %%
# =============================================================================
# 6. Plot Mexico's seasonally adjusted quarterly real GDP
# =============================================================================

analysis_data["Date"] = (
    analysis_data["Quarter"].dt.to_timestamp()
)

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    analysis_data["Date"],
    analysis_data["Real_GDP"],
    color="darkblue",
    linewidth=1.6
)

ax.set_title(
    "Mexico's Seasonally Adjusted Quarterly Real GDP"
)
ax.set_xlabel("Year")
ax.set_ylabel("Millions of pesos at 2018 prices")

fig.tight_layout()

fig.savefig(
    "Project/mexico_real_gdp.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# %%
# =============================================================================
# 7. Plot Mexico's quarterly real GDP growth
# =============================================================================
fig, axes = plt.subplots(4, 1, figsize=(6, 9))

ax = axes[0]
ax.plot(analysis_data["Date"],
    analysis_data["Log_Real_GDP"], color=EO_COPPER, linewidth=1.2)
ax.set_title("Log Real GDP — Level")
ax.set_ylabel("Log Millions of pesos at 2018 prices")