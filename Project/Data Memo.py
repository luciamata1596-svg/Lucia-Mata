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
# 7. Plot log real GDP and quarterly real GDP growth
# =============================================================================

EO_COPPER = "#B87333"
EO_SAGE = "#8A9A5B"
EO_TERRACOTTA = "#E2725B"
EO_CHARCOAL = "#36454F"

fig, axes = plt.subplots(2, 1, figsize=(7, 9))

# Log real GDP
axes[0].plot(
    analysis_data["Date"],
    analysis_data["Log_Real_GDP"],
    color=EO_COPPER,
    linewidth=1.2
)

axes[0].set_title("Mexico’s Log Real GDP — Level")
axes[0].set_ylabel("Log millions of pesos at 2018 prices")
axes[0].set_xlabel("Year")

# Quarterly GDP growth
growth_plot = analysis_data.dropna(
    subset=["GDP_Growth"]
)

axes[1].fill_between(
    growth_plot["Date"],
    growth_plot["GDP_Growth"],
    0,
    where=growth_plot["GDP_Growth"] >= 0,
    color=EO_SAGE,
    alpha=0.7,
    label="Positive"
)

axes[1].fill_between(
    growth_plot["Date"],
    growth_plot["GDP_Growth"],
    0,
    where=growth_plot["GDP_Growth"] < 0,
    color=EO_TERRACOTTA,
    alpha=0.7,
    label="Negative"
)

axes[1].axhline(
    0,
    color=EO_CHARCOAL,
    linewidth=0.6,
    linestyle="--"
)

axes[1].set_title("Mexico’s Real GDP — Quarterly Growth Rate")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Percent")
axes[1].legend(fontsize=8)

fig.tight_layout()
plt.show()
# %%
# =============================================================================
# 8. Divide the data into estimation and evaluation samples
# =============================================================================

estimation_sample = analysis_data[
    analysis_data["Quarter"] <= pd.Period("2018Q4", freq="Q")
].copy()

evaluation_sample = analysis_data[
    analysis_data["Quarter"] >= pd.Period("2019Q1", freq="Q")
].copy()

print(
    "Estimation sample:",
    estimation_sample["Quarter"].min(),
    "to",
    estimation_sample["Quarter"].max()
)

print(
    "Evaluation sample:",
    evaluation_sample["Quarter"].min(),
    "to",
    evaluation_sample["Quarter"].max()
)
# %%
# =============================================================================
# 9. ACF and PACF
# =============================================================================

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(4, 1, figsize=(7, 10))

plot_acf(
    estimation_sample["Log_Real_GDP"].dropna(),
    lags=20,
    ax=axes[0],
    color=EO_COPPER,
    vlines_kwargs={"colors": EO_COPPER},
    title="ACF — Log Real GDP Level",
    zero=False,
    alpha=0.05
)

plot_pacf(
    estimation_sample["Log_Real_GDP"].dropna(),
    lags=20,
    ax=axes[1],
    color=EO_COPPER,
    vlines_kwargs={"colors": EO_COPPER},
    method="ywm",
    title="PACF — Log Real GDP Level",
    zero=False,
    alpha=0.05
)

plot_acf(
    estimation_sample["GDP_Growth"].dropna(),
    lags=20,
    ax=axes[2],
    color=EO_SAGE,
    vlines_kwargs={"colors": EO_SAGE},
    title="ACF — Quarterly Real GDP Growth",
    zero=False,
    alpha=0.05
)

plot_pacf(
    estimation_sample["GDP_Growth"].dropna(),
    lags=20,
    ax=axes[3],
    color=EO_SAGE,
    vlines_kwargs={"colors": EO_SAGE},
    method="ywm",
    title="PACF — Quarterly Real GDP Growth",
    zero=False,
    alpha=0.05
)

for ax in axes:
    ax.set_xlabel("Lag (quarters)")
    ax.set_ylabel("Correlation")
    ax.grid(alpha=0.20)

fig.tight_layout()
plt.show()
# %%
# =============================================================================
# 10.ADF and KPSS tests
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
    estimation_sample["Log_Real_GDP"],
    "Log Real GDP",
    "ct",
    "ct",

    estimation_sample["GDP_Growth"],
    "GDP Growth (%)",
    "c",
    "c",

    title="Mexico's Real GDP — Estimation Sample"
)

# %%
# =============================================================================
# 11. ARIMA model selection using AIC and BIC
# =============================================================================
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import warnings

growth = estimation_sample["GDP_Growth"].dropna()

model_results = []

for p in range(4):
    for q in range(4):

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = ARIMA(
                    growth,
                    order=(p, 0, q),
                    trend="c",
                    enforce_stationarity=True,
                    enforce_invertibility=True
                ).fit()

            converged = model.mle_retvals.get(
                "converged",
                True
            )

            if converged:
                model_results.append({
                    "Model": f"ARIMA({p},0,{q})",
                    "p": p,
                    "q": q,
                    "AIC": model.aic,
                    "BIC": model.bic,
                    "Converged": converged
                })

            else:
                print(
                    f"ARIMA({p},0,{q}) did not converge "
                    "and was excluded."
                )

        except Exception as error:
            print(
                f"ARIMA({p},0,{q}) could not be estimated: "
                f"{error}"
            )

ic_table = (
    pd.DataFrame(model_results)
    .sort_values("BIC")
    .reset_index(drop=True)
)

print(ic_table.round(3))
best_aic = ic_table.loc[
    ic_table["AIC"].idxmin()
]

best_bic = ic_table.loc[
    ic_table["BIC"].idxmin()
]

print("\nModel selected by AIC:")
print(best_aic)

print("\nModel selected by BIC:")
print(best_bic)
# %%
# =============================================================================
# 12. Residual diagnostics for the AIC-selected ARIMA(3,0,1) 
# and BIC-selected ARIMA(0,0,1)
# =============================================================================
#AIC selected model
model_aic = ARIMA(growth, order=(3,0,1), trend="c").fit()
#BIC selected model
model_bic = ARIMA(growth, order=(0,0,1), trend="c").fit()
print(model_aic.summary())
print(model_bic.summary())
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
resid_aic = model_aic.resid.dropna()
resid_bic = model_bic.resid.dropna()

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# AIC model residuals
axes[0, 0].plot(resid_aic, color=EO_COPPER)
axes[0, 0].axhline(0, color=EO_CHARCOAL, linestyle="--")
axes[0, 0].set_title("ARIMA(3,0,1) Residuals")
plot_acf(resid_aic, lags=20, ax=axes[0,1], zero=False, alpha=0.05, title="ACF-ARIMA(3,0,1) Residuals")
# BIC model residuals
axes[1, 0].plot(resid_bic, color=EO_SAGE)
axes[1, 0].axhline(0, color=EO_CHARCOAL, linestyle="--")
axes[1, 0].set_title("ARIMA(0,0,1) Residuals")
plot_acf(resid_bic, lags=20, ax=axes[1,1], zero=False, alpha=0.05, title="ACF-ARIMA(0,0,1) Residuals")
fig.tight_layout()
plt.show()

#Ljung–Box tests
print("ARIMA(3,0,1):")
print(
    acorr_ljungbox(
        resid_aic,
        lags=[8, 12],
        model_df=4,
        return_df=True
    )
)

print("\nARIMA(0,0,1):")
print(
    acorr_ljungbox(
        resid_bic,
        lags=[4, 8, 12],
        model_df=1,
        return_df=True
    )
)
# %%
# =============================================================================
# 13. Expanding-window out-of-sample forecast evaluation
# =============================================================================
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Complete quarterly GDP growth series
y=(analysis_data.set.index("Quarter")["GDP_Growth"].dropna())
# Last observation in the initial estimation sample
initial_end= y.index.get_loc(pd.Period("2018Q4", freq="Q"))
horizons= [1,4]
errors_arima= {1: [], 4: []}
errors_mean= {1:[], 4:[]}
forecasts_arima = {1:[], 4:[]}
forecasts_mean = {1:[], 4:[]}
forecast_dates= {1:[], 4:[]}
