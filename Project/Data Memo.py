# ECON-5371 — Data Memo
# Data documentation, summary statistics, and visualization

# %%
from numpy.lib.scimath import power
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

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

fig.tight_layout()
plt.show()

# %%
# =============================================================================
# Chow tests for the 2008–2009 Global Financial Crisis
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
# Structural-break test: Chow test at 2020 Q2
# =============================================================================

from statsmodels.api import OLS, add_constant
from scipy import stats
import pandas as pd
import numpy as np

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

p_value = 1 - stats.f.cdf(
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
    )# %%
# =============================================================================
# Chow tests for the 2008–2009 Global Financial Crisis
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
# 13. Expanding-window out-of-sample forecasts
# =============================================================================

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Quarterly GDP growth with Quarter as the index
y = (
    analysis_data
    .set_index("Quarter")["GDP_Growth"]
    .dropna()
)

# Position of 2018Q4 in the growth series
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
                f"Forecast failed at {forecast_origin}, "
                f"h={h}: {error}"
            )

            arima_forecast = np.nan

        # Historical-mean benchmark
        mean_forecast = train.mean()

        # Store one row for every forecast
        forecast_results.append({
            "Horizon": h,
            "Origin": forecast_origin,
            "Target": target_date,
            "Actual": actual,
            "ARIMA_Forecast": arima_forecast,
            "Mean_Forecast": mean_forecast,
            "ARIMA_Error": actual - arima_forecast,
            "Mean_Error": actual - mean_forecast
        })

forecast_results = pd.DataFrame(
    forecast_results
).dropna()

print(forecast_results.head())
# %%
# Check the evaluation sample
for h in horizons:

    results_h = forecast_results[
        forecast_results["Horizon"] == h
    ]

    print(f"\nHorizon h={h}")
    print("Number of forecasts:", len(results_h))
    print("First target:", results_h["Target"].min())
    print("Last target:", results_h["Target"].max())
# %%
# =============================================================================
# 14. Forecast accuracy: RMSE and MAE
# =============================================================================

def rmse(errors):
    return np.sqrt(
        np.mean(np.asarray(errors) ** 2)
    )


def mae(errors):
    return np.mean(
        np.abs(np.asarray(errors))
    )


# Accuracy table
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

accuracy_table = pd.DataFrame(accuracy_rows)

print("Forecast accuracy:")
print(accuracy_table.round(3))


# Accuracy-ratio table
ratio_rows = []

for h in horizons:

    results_h = forecast_results[
        forecast_results["Horizon"] == h
    ]

    ratio_rows.append({
        "Horizon": f"h={h}",

        "RMSE Ratio": (
            rmse(results_h["ARIMA_Error"])
            / rmse(results_h["Mean_Error"])
        ),

        "MAE Ratio": (
            mae(results_h["ARIMA_Error"])
            / mae(results_h["Mean_Error"])
        )
    })

ratio_table = pd.DataFrame(ratio_rows)

print("\nAccuracy ratios: ARIMA / Historical Mean")
print(ratio_table.round(3))
print(
    forecast_results
    .groupby("Horizon")
    .agg(
        Observations=("Target", "count"),
        First_Target=("Target", "min"),
        Last_Target=("Target", "max")
    )
)
# %%
# =============================================================================
# Forecast accuracy with and without the COVID-19 period
# =============================================================================
# Ensure Target is stared as a quaterly period 
forecast_results["Target"] = pd.PeriodIndex(
    forecast_results["Target"].astype(str),
    freq="Q"
)
# COVID-19 evaluation quarters 2020 Q2–2021 Q1
covid19_start = pd.Period("2020Q2", freq="Q")
covid19_end= pd.Period("2021Q1", freq="Q")
covid_mask= ((forecast_results["Target"]>=covid19_start) & (forecast_results["Target"]<= covid19_end))
#Two Evaluation samples
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

    return pd.DataFrame(rows)
#Calculating both sets of results
accuracy_full = accuracy_by_sample(full_results, "Full Sample")
accuracy_no_covid = accuracy_by_sample(no_covid_results, "Excluding 2020Q2–2021Q1")
covid_comparison = pd.concat([accuracy_full, accuracy_no_covid], ignore_index=True)
print(covid_comparison.round(3))
# %%
# =============================================================================
# 16. Diebold-Mariano tests with HLN small-sample correction
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

# DM-HLN test: ARIMA versus historical mean
samples_dm = {"Full sample": forecast_results, "Excluding 2020Q2–2021Q1": no_covid_results}
dm_rows = []
for sample_name, data in samples_dm.items(): 
    for h in [1,4]: 
        results_h = data[data["Horizon"]== h].dropna(subset=["ARIMA_Error", "Mean_Error"])
        dm_stat, p_value, _ = dm_test_nw_hln(results_h["ARIMA_Error"],results_h["Mean_Error"], h=h, power=2)
        dm_rows.append({
            "Sample": sample_name,
            "Horizon": h,
            "Observations": len(results_h), 
            "DM-HLN Statistic": dm_stat, 
            "p-value": p_value,
            "Decision (5%)": ("Reject H0" if p_value < 0.05 else "Fail to reject H0")})
dm_table= pd.DataFrame(dm_rows)            
print(dm_table.round(3))


