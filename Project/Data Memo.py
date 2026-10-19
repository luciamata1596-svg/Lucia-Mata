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
# 7. Plotting Mexico's quarterly Log Real GDP and Real GDP growth rate 
# =============================================================================
EO_COPPER = "#B87333"
EO_SAGE = "#8A9A5B"
EO_TERRACOTTA= "#E2725B"
EO_CHARCOAL = "#36454F"
fig, axes = plt.subplots(2, 1, figsize=(6, 9))

ax = axes[0]
ax.plot(analysis_data["Date"],
    analysis_data["Log_Real_GDP"], color=EO_COPPER, linewidth=1.2)
ax.set_title("Log Real GDP — Level")
ax.set_ylabel("Log Millions of pesos at 2018 prices")
ax = axes[1]
gdp_g = analysis_data["GDP_Growth"].dropna()
ax.fill_between(gdp_g.index, gdp_g, 0,
                where=gdp_g >= 0, color=EO_SAGE, alpha=0.7, label="Positive")
ax.fill_between(gdp_g.index, gdp_g, 0,
                where=gdp_g < 0, color=EO_TERRACOTTA, alpha=0.7, label="Negative")
ax.axhline(0, color=EO_CHARCOAL, linewidth=0.6, linestyle="--")
ax.set_title("Real GDP — QoQ Growth Rate (%)")
ax.set_ylabel("Percent")
ax.legend(fontsize=6)
fig.tight_layout()
plt.show()
# %%
# =============================================================================
# 8. ACF and PACF of Mexico's quarterly real GDP growth
# =============================================================================
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(6, 9))

plot_acf(analysis_data["Log_Real_GDP"].dropna(), lags=20, ax=axes[0],
         color=EO_COPPER, vlines_kwargs={"colors": EO_COPPER},
         title="ACF — Log Real GDP (Level)", zero=False, alpha=0.05)

plot_pacf(analysis_data["Log_Real_GDP"].dropna(), lags=20, ax=axes[1],
          color=EO_COPPER, vlines_kwargs={"colors": EO_COPPER},
          method="ywm", title="PACF — Log Real GDP (Level)",
          zero=False, alpha=0.05)

plot_acf(analysis_data["GDP_Growth"].dropna(), lags=20, ax=axes[2],
         color=EO_SAGE, vlines_kwargs={"colors": EO_SAGE},
         title="ACF — GDP Growth Rate", zero=False, alpha=0.05)

plot_pacf(analysis_data["GDP_Growth"].dropna(), lags=20, ax=axes[3],
          color=EO_SAGE, vlines_kwargs={"colors": EO_SAGE},
          method="ywm", title="PACF — GDP Growth Rate",
          zero=False, alpha=0.05)

for ax in axes:
    ax.set_xlabel("Lag")
    ax.set_ylabel("Correlation")
    eo_style_ax(ax)
    for line in ax.lines:
        if line.get_linestyle() == "--":
            line.set_color(EO_TERRACOTTA)
            line.set_linewidth(0.7)
fig.tight_layout()
plt.show()
# %%
# =============================================================================
# 8.ADF and KPSS tests
# =============================================================================
from statsmodels.tsa.stattools import adfuller, kpss
def _run_adf(series, regression):
    r = adfuller(series.dropna(), regression=regression, autolag="AIC")
    return {"stat": r[0], "p": r[1], "lags": r[2],
            "cv": r[4], "conc": "Stationary" if r[1] < 0.05 else "Unit root"}

def _run_kpss(series, regression):
    r = kpss(series.dropna(), regression=regression, nlags="auto")
    return {"stat": r[0], "p": r[1], "bw": r[2],
            "cv": r[3], "conc": "Unit root" if r[1] < 0.05 else "Stationary"}

def print_side_by_side(s1, n1, r1_adf, r1_kpss, s2, n2, r2_adf, r2_kpss, title):
    a1 = _run_adf(s1, r1_adf);   k1 = _run_kpss(s1, r1_kpss)
    a2 = _run_adf(s2, r2_adf);   k2 = _run_kpss(s2, r2_kpss)
    W  = 16
    print(f"\n{'\u2550'*58}")
    print(f"  {title}")
    print(f"{'\u2550'*58}")
    print(f"\n  ADF Test  (H\u2080: unit root)")
    print(f"  {'\u2500'*54}")
    print(f"  {'':22s} {n1:>{W}s} {n2:>{W}s}")
    print(f"  {'\u2500'*54}")
    print(f"  {'Spec':22s} {r1_adf:>{W}s} {r2_adf:>{W}s}")
    print(f"  {'Test statistic':22s} {a1['stat']:>{W}.4f} {a2['stat']:>{W}.4f}")
    print(f"  {'p-value':22s} {a1['p']:>{W}.4f} {a2['p']:>{W}.4f}")
    print(f"  {'Lags (AIC)':22s} {a1['lags']:>{W}d} {a2['lags']:>{W}d}")
    for lv in ['1%', '5%', '10%']:
        print(f"  {f'Critical val {lv}':22s} {a1['cv'][lv]:>{W}.4f} {a2['cv'][lv]:>{W}.4f}")
    print(f"  {'Conclusion':22s} {a1['conc']:>{W}s} {a2['conc']:>{W}s}")
    print(f"\n  KPSS Test (H\u2080: stationary)")
    print(f"  {'\u2500'*54}")
    print(f"  {'Spec':22s} {r1_kpss:>{W}s} {r2_kpss:>{W}s}")
    print(f"  {'Test statistic':22s} {k1['stat']:>{W}.4f} {k2['stat']:>{W}.4f}")
    print(f"  {'p-value':22s} {k1['p']:>{W}.4f} {k2['p']:>{W}.4f}")
    print(f"  {'Bandwidth':22s} {k1['bw']:>{W}d} {k2['bw']:>{W}d}")
    for lv in ['10%', '5%', '1%']:
        print(f"  {f'Critical val {lv}':22s} {k1['cv'][lv]:>{W}.4f} {k2['cv'][lv]:>{W}.4f}")
    print(f"  {'Conclusion':22s} {k1['conc']:>{W}s} {k2['conc']:>{W}s}")
    print(f"  {'\u2500'*54}")

print_side_by_side(
    analysis_data["Log_Real_GDP"],     "Log Real GDP",    "ct", "ct",
    analysis_data["GDP_Growth"], "GDP Growth (%)",  "c",  "c",
    title="Mexico's Real GDP")