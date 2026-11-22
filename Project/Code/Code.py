# ECON-5371 — Project Forecasting Mexico’s Quarterly Real GDP Growth Code
# Data documentation, summary statistics, and visualization
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_FILE = (
    PROJECT_DIR / "data" / "data" / "Raw Data" / "pibt_cte_valor.xlsx"
)
CLEAN_DATA_FILE = (
    PROJECT_DIR / "data" / "data" / "Clean Data" / "mexico_gdp_clean.csv"
)
CLEAN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
# %%
# =============================================================================
# Import Required packages
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from statsmodels.api import OLS, add_constant
from scipy import stats
import warnings
from statsmodels.tsa.arima.model import ARIMA
# %%
# =============================================================================
# 1.Import the Excel File
# =============================================================================

mexico_gdp = pd.read_excel(
    RAW_DATA_FILE,
    sheet_name="desest-valor",
    header=None
)
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
# 5. Export the final clean research dataset
# =============================================================================

clean_export = analysis_data[
    ["Quarter", "Real_GDP", "Log_Real_GDP", "GDP_Growth"]
].copy()

clean_export["Quarter"] = clean_export["Quarter"].astype(str)

clean_export.to_csv(
    CLEAN_DATA_FILE,
    index=False
)
# %%
# =============================================================================
# 6. Summary statistics
# =============================================================================
summary_statistics = (
    analysis_data[
        ["Real_GDP", "Log_Real_GDP", "GDP_Growth"]
    ]
    .describe()
    .rename_axis("Statistic")
    .reset_index()
)

print(summary_statistics.round(4))
# %%
# =============================================================================
# 7. Plot Mexico's seasonally adjusted quarterly real GDP
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
    OUTPUT_DIR / "Figure 1 Mexico’s seasonally adjusted quarterly real GDP.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
# %%
# =============================================================================
# 8. Plot log real GDP and quarterly real GDP growth
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
fig.savefig(
    OUTPUT_DIR / "Figure 2 Mexico’s log real GDP and Mexico’s quarterly real GDP growth rate.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
# %%
# =============================================================================
# 9. Divide the data into estimation and evaluation samples
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
# 10. ACF and PACF
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

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "Figure 3 ACF and PACF graphs.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
# %%
# =============================================================================
# 11. Chow tests for the 2008–2009 Global Financial Crisis
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.api import OLS

# Prepare Mexico's quarterly GDP-growth data
df = analysis_data[
    ["Quarter", "GDP_Growth"]
].copy()

# Create AR(2) lags
df["GDP_Growth_lag1"] = (
    df["GDP_Growth"].shift(1)
)

df["GDP_Growth_lag2"] = (
    df["GDP_Growth"].shift(2)
)

df = df.dropna().reset_index(drop=True)


def chow_test_ar2(data, break_period):

    # Dependent variable
    y = data["GDP_Growth"].to_numpy()

    # Constant and two growth lags
    X = np.column_stack([
        np.ones(len(data)),
        data["GDP_Growth_lag1"].to_numpy(),
        data["GDP_Growth_lag2"].to_numpy()
    ])

    # Break dummy
    D = (
        data["Quarter"] >= break_period
    ).astype(float).to_numpy()

    # Allow every coefficient to change after the break
    X_unrestricted = np.column_stack([
        X,
        D[:, None] * X
    ])

    # Estimate restricted and unrestricted models
    restricted = OLS(y, X).fit()
    unrestricted = OLS(y, X_unrestricted).fit()

    RSS_restricted = restricted.ssr
    RSS_unrestricted = unrestricted.ssr

    T = len(y)
    k = X.shape[1]  # Constant plus two lags

    # Chow F-statistic
    F_statistic = (
        ((RSS_restricted - RSS_unrestricted) / k)
        / (RSS_unrestricted / (T - 2 * k))
    )

    p_value = stats.f.sf(
        F_statistic,
        k,
        T - 2 * k
    )

    return {
        "Break": str(break_period),
        "F-statistic": F_statistic,
        "p-value": p_value,
        "Conclusion": (
            "Structural break"
            if p_value < 0.10
            else "No significant break"
        )
    }

# Test two possible crisis dates
break_dates = [
    pd.Period("2008Q4", freq="Q"),
    pd.Period("2009Q1", freq="Q")
]

chow_results = pd.DataFrame([
    chow_test_ar2(df, break_date)
    for break_date in break_dates
])

print(
    chow_results.round({
        "F-statistic": 3,
        "p-value": 4
    })
)
# %%
# =============================================================================
# 12. Structural-break test: Chow test at 2020 Q2
# =============================================================================
# Prepare Mexico's quarterly GDP-growth series
df = analysis_data[["Quarter", "GDP_Growth"]].copy()

# Create two GDP-growth lags
df["GDP_Growth_lag1"] = df["GDP_Growth"].shift(1)
df["GDP_Growth_lag2"] = df["GDP_Growth"].shift(2)

df = df.dropna().reset_index(drop=True)

# Suspected COVID-19 break
BREAK = pd.Period("2020Q2", freq="Q")

# Intercept plus two AR lags
k = 3

# Dummy equals 1 beginning in 2020 Q2
D = (
    df["Quarter"] >= BREAK
).astype(float).to_numpy()

# Restricted AR(2) model
X_full = np.column_stack([
    np.ones(len(df)),
    df["GDP_Growth_lag1"].to_numpy(),
    df["GDP_Growth_lag2"].to_numpy()
])

# Unrestricted model allows every coefficient to change after 2020 Q2
X_unrestricted = np.column_stack([
    X_full,
    D[:, None] * X_full
])

y_vec = df["GDP_Growth"].to_numpy()

restricted_model = OLS(y_vec, X_full).fit()
unrestricted_model = OLS(y_vec, X_unrestricted).fit()

RSS_restricted = restricted_model.ssr
RSS_unrestricted = unrestricted_model.ssr
T = len(y_vec)

# Chow-test statistic
F_chow = (
    ((RSS_restricted - RSS_unrestricted) / k)
    / (RSS_unrestricted / (T - 2 * k))
)
p_value = stats.f.sf(
    F_chow,
    k,
    T - 2 * k
)

print("Chow Test — AR(2) Model for Mexico's Quarterly GDP Growth")
print("Break date: 2020 Q2")
print(f"F-statistic: {F_chow:.3f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.10:
    print(
        "Reject H0 at the 10% level: "
        "evidence of a structural break."
    )
else:
    print(
        "Fail to reject H0 at the 10% level: "
        "insufficient evidence of a structural break."
    )
chow_covid_table = pd.DataFrame([{
    "Break": "2020Q2",
    "F-statistic": F_chow,
    "p-value": p_value,
    "Conclusion": (
        "Structural break"
        if p_value < 0.10
        else "No significant break"
    )
}])

print(chow_covid_table.round(4))
# %%
# =============================================================================
# 13. Structural-break robustness: 1994–95 peso-crisis dummy
# =============================================================================
# Use only the pre-2019 estimation sample
crisis_data = (
    estimation_sample
    .dropna(subset=["GDP_Growth"])
    .copy()
)

# Temporary crisis dummy:
# 1 during 1994 Q4–1995 Q4 and 0 otherwise
crisis_data["Peso_Crisis_Dummy"] = (
    (crisis_data["Quarter"] >= pd.Period("1994Q4", freq="Q"))
    &
    (crisis_data["Quarter"] <= pd.Period("1995Q4", freq="Q"))
).astype(int)

y_crisis = crisis_data["GDP_Growth"]
x_crisis = crisis_data[["Peso_Crisis_Dummy"]]

# Baseline selected model
model_without_dummy = ARIMA(
    y_crisis,
    order=(0, 0, 1),
    trend="c"
).fit()

# Same model with the crisis dummy
model_with_dummy = ARIMA(
    y_crisis,
    exog=x_crisis,
    order=(0, 0, 1),
    trend="c"
).fit()

# Compare specifications
dummy_comparison = pd.DataFrame({
    "Specification": [
        "ARIMA(0,0,1)",
        "ARIMA(0,0,1) + Peso-crisis dummy"
    ],
    "AIC": [
        model_without_dummy.aic,
        model_with_dummy.aic
    ],
    "BIC": [
        model_without_dummy.bic,
        model_with_dummy.bic
    ]
})

print(dummy_comparison.round(3))

print("\nPeso-crisis dummy coefficient:")
print(
    round(
        model_with_dummy.params["Peso_Crisis_Dummy"],
        4
    )
)

print("Peso-crisis dummy p-value:")
print(
    round(
        model_with_dummy.pvalues["Peso_Crisis_Dummy"],
        4
    )
)
# %%
# =============================================================================
# 14. ADF and KPSS tests
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
# Run the tests and store their results
adf_log = _run_adf(
    estimation_sample["Log_Real_GDP"],
    regression="ct"
)

kpss_log = _run_kpss(
    estimation_sample["Log_Real_GDP"],
    regression="ct"
)

adf_growth = _run_adf(
    estimation_sample["GDP_Growth"],
    regression="c"
)

kpss_growth = _run_kpss(
    estimation_sample["GDP_Growth"],
    regression="c"
)

stationarity_table = pd.DataFrame([
    {
        "Series": "Log Real GDP",
        "Test": "ADF",
        "Specification": "Constant and trend",
        "Statistic": adf_log["stat"],
        "p-value": adf_log["p"],
        "Lags/Bandwidth": adf_log["lags"],
        "Critical Value 1%": adf_log["cv"]["1%"],
        "Critical Value 5%": adf_log["cv"]["5%"],
        "Critical Value 10%": adf_log["cv"]["10%"],
        "Conclusion": (
            "Reject unit-root null"
            if adf_log["p"] < 0.05
            else "Fail to reject unit-root null"
        )
    },
    {
        "Series": "Log Real GDP",
        "Test": "KPSS",
        "Specification": "Constant and trend",
        "Statistic": kpss_log["stat"],
        "p-value": kpss_log["p"],
        "Lags/Bandwidth": kpss_log["bw"],
        "Critical Value 1%": kpss_log["cv"]["1%"],
        "Critical Value 5%": kpss_log["cv"]["5%"],
        "Critical Value 10%": kpss_log["cv"]["10%"],
        "Conclusion": (
            "Reject stationarity null"
            if kpss_log["p"] < 0.05
            else "Fail to reject stationarity null"
        )
    },
    {
        "Series": "GDP Growth",
        "Test": "ADF",
        "Specification": "Constant",
        "Statistic": adf_growth["stat"],
        "p-value": adf_growth["p"],
        "Lags/Bandwidth": adf_growth["lags"],
        "Critical Value 1%": adf_growth["cv"]["1%"],
        "Critical Value 5%": adf_growth["cv"]["5%"],
        "Critical Value 10%": adf_growth["cv"]["10%"],
        "Conclusion": (
            "Reject unit-root null"
            if adf_growth["p"] < 0.05
            else "Fail to reject unit-root null"
        )
    },
    {
        "Series": "GDP Growth",
        "Test": "KPSS",
        "Specification": "Constant",
        "Statistic": kpss_growth["stat"],
        "p-value": kpss_growth["p"],
        "Lags/Bandwidth": kpss_growth["bw"],
        "Critical Value 1%": kpss_growth["cv"]["1%"],
        "Critical Value 5%": kpss_growth["cv"]["5%"],
        "Critical Value 10%": kpss_growth["cv"]["10%"],
        "Conclusion": (
            "Reject stationarity null"
            if kpss_growth["p"] < 0.05
            else "Fail to reject stationarity null"
        )
    }
])

print(stationarity_table.round(4))
# %%
# =============================================================================
# 15. ARIMA model selection using AIC and BIC
# =============================================================================
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
# 16. Residual diagnostics for the AIC-selected ARIMA(3,0,1) 
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
fig.savefig(
    OUTPUT_DIR / "Figure A1 Residual diagnostics for the ARIMA(0,0,1) model.png",
    dpi=300,
    bbox_inches="tight"
)
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
lb_aic = (
    acorr_ljungbox(
        resid_aic,
        lags=[8, 12],
        model_df=4,
        return_df=True
    )
    .rename_axis("Lag")
    .reset_index()
)

lb_aic.insert(0, "Model", "ARIMA(3,0,1)")

lb_bic = (
    acorr_ljungbox(
        resid_bic,
        lags=[4, 8, 12],
        model_df=1,
        return_df=True
    )
    .rename_axis("Lag")
    .reset_index()
)

lb_bic.insert(0, "Model", "ARIMA(0,0,1)")

ljung_box_table = pd.concat(
    [lb_aic, lb_bic],
    ignore_index=True
)

ljung_box_table = ljung_box_table.rename(columns={
    "lb_stat": "Q-statistic",
    "lb_pvalue": "p-value"
})

ljung_box_table["Conclusion (10%)"] = np.where(
    ljung_box_table["p-value"] < 0.10,
    "Reject no-autocorrelation null",
    "Fail to reject no-autocorrelation null"
)

print(ljung_box_table.round(4))
# %%
# =============================================================================
# ARIMA model comparison and residual-diagnostics table
# =============================================================================

def format_pvalue(p_value):
    """Format p-values for the coefficient column."""
    if p_value < 0.001:
        return "<.001"

    return f"{p_value:.3f}".lstrip("0")


def format_coefficients(model_result, coefficient_names):
    """Combine coefficient estimates and p-values in one cell."""

    parameter_names = model_result.model.param_names

    estimates = dict(
        zip(parameter_names, model_result.params)
    )

    p_values = dict(
        zip(parameter_names, model_result.pvalues)
    )

    formatted_results = []

    for parameter, label in coefficient_names.items():
        estimate = estimates[parameter]
        p_value = p_values[parameter]

        formatted_results.append(
            f"{label}: {estimate:.3f} "
            f"({format_pvalue(p_value)})"
        )

    return "; ".join(formatted_results)


# Ljung–Box results for the two models
lb_arima_301 = acorr_ljungbox(
    resid_aic,
    lags=[8, 12],
    model_df=4,
    return_df=True
)

lb_arima_001 = acorr_ljungbox(
    resid_bic,
    lags=[8, 12],
    model_df=1,
    return_df=True
)


# Main comparison table
arima_comparison_table = pd.DataFrame([
    {
        "Model": "ARIMA(3,0,1)",
        "Dynamic coefficients (estimate; p-value)": (
            format_coefficients(
                model_aic,
                {
                    "ar.L1": "AR1",
                    "ar.L2": "AR2",
                    "ar.L3": "AR3",
                    "ma.L1": "MA1"
                }
            )
        ),
        "AIC": model_aic.aic,
        "BIC": model_aic.bic,
        "LB p-value, lag 8": (
            lb_arima_301.loc[8, "lb_pvalue"]
        ),
        "LB p-value, lag 12": (
            lb_arima_301.loc[12, "lb_pvalue"]
        )
    },
    {
        "Model": "ARIMA(0,0,1)",
        "Dynamic coefficients (estimate; p-value)": (
            format_coefficients(
                model_bic,
                {
                    "ma.L1": "MA1"
                }
            )
        ),
        "AIC": model_bic.aic,
        "BIC": model_bic.bic,
        "LB p-value, lag 8": (
            lb_arima_001.loc[8, "lb_pvalue"]
        ),
        "LB p-value, lag 12": (
            lb_arima_001.loc[12, "lb_pvalue"]
        )
    }
])

print(
    arima_comparison_table
    .round(3)
    .to_string(index=False)
)
# %%
# =============================================================================
# 17. Expanding-window out-of-sample forecasts
# =============================================================================
# Quarterly GDP growth with Quarter as the index
y = (
    analysis_data
    .set_index("Quarter")["GDP_Growth"]
    .dropna()
)

# Position of 2018 Q4 in the growth series
initial_end = y.index.get_loc(
    pd.Period("2018Q4", freq="Q")
)

horizons = [1, 4]
forecast_results = []

for h in horizons:

    for origin in range(initial_end, len(y) - h):

        # Expanding estimation window
        train = y.iloc[:origin + 1]

        forecast_origin = y.index[origin]
        target_date = y.index[origin + h]
        actual = y.iloc[origin + h]

        # ARIMA(0,0,1) forecast
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                fitted_model = ARIMA(
                    train,
                    order=(0, 0, 1),
                    trend="c",
                    enforce_stationarity=True,
                    enforce_invertibility=True
                ).fit()

            arima_forecast = (
                fitted_model
                .forecast(steps=h)
                .iloc[-1]
            )

        except Exception as error:
            print(
                f"ARIMA forecast failed at {forecast_origin}, "
                f"h={h}: {error}"
            )
            arima_forecast = np.nan

        # AR(1) benchmark
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                fitted_ar1 = ARIMA(
                    train,
                    order=(1, 0, 0),
                    trend="c",
                    enforce_stationarity=True
                ).fit()

            ar1_forecast = (
                fitted_ar1
                .forecast(steps=h)
                .iloc[-1]
            )

        except Exception as error:
            print(
                f"AR(1) forecast failed at {forecast_origin}, "
                f"h={h}: {error}"
            )
            ar1_forecast = np.nan

        # Historical-mean benchmark
        mean_forecast = train.mean()

        # Random-walk benchmark: last observed GDP growth
        rw_forecast = train.iloc[-1]

        # Store one row for every forecast
        forecast_results.append({
            "Horizon": h,
            "Origin": forecast_origin,
            "Target": target_date,
            "Actual": actual,

            "ARIMA_Forecast": arima_forecast,
            "Mean_Forecast": mean_forecast,
            "RW_Forecast": rw_forecast,
            "AR1_Forecast": ar1_forecast,

            "ARIMA_Error": actual - arima_forecast,
            "Mean_Error": actual - mean_forecast,
            "RW_Error": actual - rw_forecast,
            "AR1_Error": actual - ar1_forecast
        })

forecast_results = (
    pd.DataFrame(forecast_results)
    .dropna()
)

print(forecast_results.head())
# %%
# =============================================================================
# 18.  Check the evaluation sample
# =============================================================================
for h in horizons:

    results_h = forecast_results[
        forecast_results["Horizon"] == h
    ]

    print(f"\nHorizon h={h}")
    print("Number of forecasts:", len(results_h))
    print("First target:", results_h["Target"].min())
    print("Last target:", results_h["Target"].max())
    evaluation_check = (
    forecast_results
    .groupby("Horizon", as_index=False)
    .agg(
        Observations=("Target", "size"),
        First_Target=("Target", "min"),
        Last_Target=("Target", "max")
    )
)

evaluation_check["First_Target"] = (
    evaluation_check["First_Target"].astype(str)
)

evaluation_check["Last_Target"] = (
    evaluation_check["Last_Target"].astype(str)
)

print(evaluation_check)
    
# %%
# =============================================================================
# 19. Forecast accuracy: RMSE and MAE
# =============================================================================

def rmse(errors):
    return np.sqrt(np.mean(np.asarray(errors) ** 2))


def mae(errors):
    return np.mean(np.abs(np.asarray(errors)))


accuracy_rows = []

for h in horizons:

    results_h = forecast_results[
        forecast_results["Horizon"] == h
    ]

    accuracy_rows.append({
        "Model": "ARIMA(0,0,1)",
        "Horizon": h,
        "RMSE": rmse(results_h["ARIMA_Error"]),
        "MAE": mae(results_h["ARIMA_Error"])
    })

    accuracy_rows.append({
        "Model": "Historical Mean",
        "Horizon": h,
        "RMSE": rmse(results_h["Mean_Error"]),
        "MAE": mae(results_h["Mean_Error"])
    })

    accuracy_rows.append({
        "Model": "Random Walk",
        "Horizon": h,
        "RMSE": rmse(results_h["RW_Error"]),
        "MAE": mae(results_h["RW_Error"])
    })

    accuracy_rows.append({
        "Model": "AR(1)",
        "Horizon": h,
        "RMSE": rmse(results_h["AR1_Error"]),
        "MAE": mae(results_h["AR1_Error"])
    })


accuracy_table = pd.DataFrame(accuracy_rows)

print("Forecast accuracy:")
print(accuracy_table.round(3))
# %%
# =============================================================================
# 20. Forecast accuracy with and without the COVID-19 period
# =============================================================================

# Ensure Target is stored as a quarterly period
forecast_results["Target"] = pd.PeriodIndex(
    forecast_results["Target"].astype(str),
    freq="Q"
)

# COVID-19 evaluation quarters: 2020 Q2–2021 Q1
covid19_start = pd.Period("2020Q2", freq="Q")
covid19_end = pd.Period("2021Q1", freq="Q")

covid_mask = (
    (forecast_results["Target"] >= covid19_start)
    & (forecast_results["Target"] <= covid19_end)
)

# Two evaluation samples
full_results = forecast_results.copy()

no_covid_results = forecast_results[
    ~covid_mask
].copy()


def accuracy_by_sample(data, sample_name):

    rows = []

    for h in [1, 4]:

        results_h = data[
            data["Horizon"] == h
        ]

        rows.append({
            "Sample": sample_name,
            "Model": "ARIMA(0,0,1)",
            "Horizon": h,
            "Observations": len(results_h),
            "RMSE": rmse(results_h["ARIMA_Error"]),
            "MAE": mae(results_h["ARIMA_Error"])
        })

        rows.append({
            "Sample": sample_name,
            "Model": "Historical Mean",
            "Horizon": h,
            "Observations": len(results_h),
            "RMSE": rmse(results_h["Mean_Error"]),
            "MAE": mae(results_h["Mean_Error"])
        })

        rows.append({
            "Sample": sample_name,
            "Model": "Random Walk",
            "Horizon": h,
            "Observations": len(results_h),
            "RMSE": rmse(results_h["RW_Error"]),
            "MAE": mae(results_h["RW_Error"])
        })

        rows.append({
            "Sample": sample_name,
            "Model": "AR(1)",
            "Horizon": h,
            "Observations": len(results_h),
            "RMSE": rmse(results_h["AR1_Error"]),
            "MAE": mae(results_h["AR1_Error"])
        })

    return pd.DataFrame(rows)


# Calculate both sets of results
accuracy_full = accuracy_by_sample(
    full_results,
    "Full Sample"
)

accuracy_no_covid = accuracy_by_sample(
    no_covid_results,
    "Excluding 2020Q2–2021Q1"
)

covid_comparison = pd.concat(
    [accuracy_full, accuracy_no_covid],
    ignore_index=True
)

print(covid_comparison.round(3))
# %%
# =============================================================================
# 21. Diebold-Mariano tests with HLN small-sample correction
# =============================================================================
def dm_test_nw_hln(e1,e2, h, power=2):
# DM test using the Newey-West procedure plus the HLN small-sample correction.
    e1=np.asarray(e1)
    e2=np.asarray(e2)
#Loss differential 
    d= np.abs(e1)**power - np.abs(e2)**power
    d_c= d- d.mean()
    n= len(d)
    bw= max(h,1)
# Newey-West long-run variance
    gamma0= np.mean(d_c**2)
    lrv= gamma0
    for k in range(1, bw +1):
        weight = 1 -k/(bw + 1)
        gamma_k = np.mean(d_c[k:]* d_c[:-k])
        lrv += 2 * weight * gamma_k
# Original DM statistic
    dm_stat = d.mean()/ np.sqrt(lrv/n)
# HLN small-sample correction   
    hln_factor = np.sqrt((n + 1 - 2*h + h*(h - 1)/n) / n)

    dm_hln = dm_stat * hln_factor
    p_value = 2 * student_t.sf(
        abs(dm_hln),
        df=n - 1
    )

    return dm_hln, p_value, d

# DM-HLN tests: ARIMA versus all benchmarks
samples_dm = {
    "Full Sample": forecast_results,
    "Excluding 2020Q2–2021Q1": no_covid_results
}

benchmarks_dm = {
    "Historical Mean": "Mean_Error",
    "Random Walk": "RW_Error",
    "AR(1)": "AR1_Error"
}

dm_rows = []

for sample_name, data in samples_dm.items():

    for benchmark_name, benchmark_error in benchmarks_dm.items():

        for h in [1, 4]:

            results_h = data[
                data["Horizon"] == h
            ].dropna(
                subset=["ARIMA_Error", benchmark_error]
            )

            dm_stat, p_value, _ = dm_test_nw_hln(
                results_h["ARIMA_Error"],
                results_h[benchmark_error],
                h=h,
                power=2
            )

            dm_rows.append({
                "Sample": sample_name,
                "Comparison": f"ARIMA vs. {benchmark_name}",
                "Horizon": h,
                "Observations": len(results_h),
                "DM-HLN Statistic": dm_stat,
                "p-value": p_value,
                "Decision (10%)": (
                    "Reject H0"
                    if p_value < 0.10
                    else "Fail to reject H0"
                )
            })

dm_table = pd.DataFrame(dm_rows)

print(dm_table.round(3))
# %%
# =============================================================================
# Out-of-sample forecast comparison graph
# =============================================================================

from pathlib import Path

plot_data = forecast_results.copy()

# Convert quarterly targets into dates for the x-axis
plot_data["Target_Date"] = pd.PeriodIndex(
    plot_data["Target"].astype(str),
    freq="Q"
).to_timestamp()

fig, axes = plt.subplots(
    2, 1,
    figsize=(11, 8),
    sharex=True,
    sharey=True
)

for ax, h in zip(axes, [1, 4]):

    data_h = plot_data[
        plot_data["Horizon"] == h
    ].sort_values("Target_Date")

    # Actual GDP growth
    ax.plot(
        data_h["Target_Date"],
        data_h["Actual"],
        color="black",
        linewidth=2.2,
        label="Actual"
    )

    # Forecasts
    ax.plot(
        data_h["Target_Date"],
        data_h["ARIMA_Forecast"],
        color="#B87333",
        linewidth=1.5,
        label="ARIMA(0,0,1)"
    )

    ax.plot(
        data_h["Target_Date"],
        data_h["Mean_Forecast"],
        color="#008080",
        linestyle="--",
        linewidth=1.4,
        label="Historical Mean"
    )

    ax.plot(
        data_h["Target_Date"],
        data_h["AR1_Forecast"],
        color="royalblue",
        linestyle="-.",
        linewidth=1.4,
        label="AR(1)"
    )

    ax.plot(
        data_h["Target_Date"],
        data_h["RW_Forecast"],
        color="gray",
        linestyle=":",
        linewidth=1.4,
        label="Random Walk"
    )

    # Shade the COVID-19 target quarters
    ax.axvspan(
        pd.Period("2020Q2", freq="Q").start_time,
        pd.Period("2021Q1", freq="Q").end_time,
        color="lightgray",
        alpha=0.35,
        label="COVID-19 period"
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.6
    )

    ax.set_title(
        f"Out-of-Sample Forecasts — h = {h}"
    )

    ax.set_ylabel("GDP Growth (%)")
    ax.legend(
        fontsize=8,
        ncol=3,
        loc="lower right"
    )

axes[1].set_xlabel("Target Quarter")

fig.suptitle(
    "Actual and Forecasted Quarterly Real GDP Growth",
    fontsize=14,
    fontweight="bold"
)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "Figure 4 Actual and out-of-sample forecasts of Mexico’s quarterly real GDP growth.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
# %%
# =============================================================================
# Export all tables as separate CSV files
# =============================================================================
from pathlib import Path
# Summary statistics
summary_statistics.round(4).to_csv(
    OUTPUT_DIR / "summary_statistics.csv",
    index=False
)

# Structural-break and robustness tables
chow_results.round(4).to_csv(
    OUTPUT_DIR / "chow_gfc.csv",
    index=False
)

chow_covid_table.round(4).to_csv(
    OUTPUT_DIR / "chow_covid.csv",
    index=False
)

dummy_comparison.round(4).to_csv(
    OUTPUT_DIR / "peso_crisis_dummy.csv",
    index=False
)

# Stationarity and model-selection tables
stationarity_table.round(4).to_csv(
    OUTPUT_DIR / "stationarity_tests.csv",
    index=False
)

ic_table.round(4).to_csv(
    OUTPUT_DIR / "arima_model_selection.csv",
    index=False
)

ljung_box_table.round(4).to_csv(
    OUTPUT_DIR / "ljung_box_tests.csv",
    index=False
)
arima_comparison_table.round(3).to_csv(
    OUTPUT_DIR / "arima_model_comparison_diagnostics.csv",
    index=False
)
# Forecast evaluation tables
evaluation_check.to_csv(
    OUTPUT_DIR / "evaluation_sample.csv",
    index=False
)

accuracy_table.round(4).to_csv(
    OUTPUT_DIR / "forecast_accuracy.csv",
    index=False
)

covid_comparison.round(4).to_csv(
    OUTPUT_DIR / "forecast_accuracy_covid_comparison.csv",
    index=False
)

dm_table.round(4).to_csv(
    OUTPUT_DIR / "dm_hln_tests.csv",
    index=False
)

# Detailed forecast results
forecast_results.round(6).to_csv(
    OUTPUT_DIR / "forecast_results.csv",
    index=False
)

print("\nCSV files created:")
for file in sorted(OUTPUT_DIR.glob("*.csv")):
    print(file.name)
# %%
# =============================================================================
# Export all tables as separate CSV files
# =============================================================================


import sys
from importlib.metadata import version

packages = [
    "pandas",
    "numpy",
    "matplotlib",
    "scipy",
    "statsmodels",
    "openpyxl"
]

print("Python:", sys.version.split()[0])

for package in packages:
    print(f"{package}=={version(package)}")