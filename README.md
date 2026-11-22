# Lucia Mata — Personal Website
Source code for my personal website, built with Quarto and
deployed via GitHub Pages.
**Live site:** https://luciamata1596-svg.github.io/Lucia-Mata/
## Built with
- [Quarto](https://quarto.org)
- GitHub Pages
## Structure
- `index.qmd` — home page (CV, research interests, skills, contact)
- `project.qmd` — research project page
- `files/` — downloadable CV
- `images/` — headshot and other images
- `docs/` — rendered site (published output, do not edit directly)
### Dashboard
Dashboard link: https://luciamata1596-svg.github.io/Lucia-Mata/dashboard/dashboard.html
## Research Project: Forecasting Mexico’s Quarterly Real GDP Growth
This project analyzes whether  ARIMA (0,0,1) produces more accurate one- and four-quarter-ahead forecasts of Mexico’s quarterly real GDP growth than historical-mean, random-walk, and AR(1) benchmarks. It also examines whether the results change after excluding the COVID-19 target quarters from 2020 Q2 through 2021 Q1. Forecast accuracy is determined by comparing RMSE and MAE, as well as DM-HLN tests, for the four forecasting methods. GDP forecasts help us understand future economic conditions, and a complicated model is valuable only if it outperforms a simple benchmark. Forecast accuracy results showed that the ARIMA (0,0,1) model did not consistently outperform the benchmarks, and the DM-HLN tests found no statistically significant difference; limitations include the short evaluation sample and structural breaks.
## Data sources

The data observed for this project comes from the Instituto Nacional de EstadÃ­stica y GeografÃ­a 
(INEGI) website. The data set series has GDP at market prices, it is seasonally adjusted, and the 
values are measured in millions of pesos at 2018 prices. The data set begins at the first quarter of 
1993 and ends at the second quarter of 2026. For the analysis of this research only the data from 
the first quarter of 1993 through the last quarter of 2025 would be observed giving a total of 132 
observations. The initial model will be estimated by using the data from the last quarter of 2018, 
while the periods from the first quarter of 2019 to the last quarter of 2025 would be used for out-
of-sample forecast evaluation.

Website page: https://www.inegi.org.mx/programas/pib/2018/#tabulados

Data set link:
https://www.inegi.org.mx/contenidos/programas/pib/2018/tabulados/des/pibt_cte_valor.xlsx

## Software Requirements

The analysis was developed and tested using Python 3.12.14.

### Required Python packages
- pandas==[3.0.6]
- numpy==[2.5.3]
- matplotlib==[3.11.1]
- scipy==[1.18.1]
- statsmodels==[0.15.0]
- openpyxl==[3.1.5]

`openpyxl` is required to import the original INEGI Excel workbook.
The analysis also uses `pathlib` and `warnings`, which are included in
the Python standard library and do not require separate installation.

The required packages can be installed from the project root with:

```bash
python -m pip install -r requirements.txt
```
## Step-by-Step Replication Instructions

All commands should be run from the main repository directory.

### Step 1: Install the Python packages

Install the required package versions using:

```bash
python -m pip install -r requirements.txt
```

This step only installs the software dependencies and does not generate
any project results.

### Step 2: Confirm the raw data location

Place the original, unmodified INEGI workbook at:

```text
Project/data/data/Raw Data/pibt_cte_valor.xlsx
```

The Excel workbook must contain the worksheet named `desest-valor`.

### Step 3: Run the Python analysis

Run the main analysis script:

```bash
python Project/Code/Code.py
```

`Code.py` performs the complete analysis in the following order:

1. Imports the original INEGI Excel workbook.
2. Cleans and restructures the quarterly real GDP data.
3. Calculates log real GDP and quarterly GDP growth.
4. Produces summary statistics and time-series figures.
5. Conducts the structural-break tests.
6. Conducts the ADF and KPSS stationarity tests.
7. Estimates and compares the candidate ARIMA models.
8. Performs the residual diagnostics and Ljung–Box tests.
9. Generates expanding-window forecasts at horizons 1 and 4.
10. Calculates RMSE and MAE for ARIMA and the three benchmarks.
11. Recalculates forecast accuracy without the COVID-19 target quarters.
12. Conducts the DM-HLN equal-predictive-accuracy tests.
13. Exports the clean data, figures, tables, and detailed forecasts.

### Step 4: Check the clean-data output

The script creates:

```text
Project/data/data/Clean Data/mexico_gdp_clean.csv
```

This file contains:

- `Quarter`
- `Real_GDP`
- `Log_Real_GDP`
- `GDP_Growth`

The expected clean dataset contains 132 quarterly real GDP observations
from 1993 Q1 through 2025 Q4. GDP growth contains 131 nonmissing
observations because calculating growth removes the first observation.

### Step 5: Check the table outputs

The following files are created in `Project/Output/`:

```text
summary_statistics.csv
chow_gfc.csv
chow_covid.csv
peso_crisis_dummy.csv
stationarity_tests.csv
arima_model_selection.csv
ljung_box_tests.csv
arima_model_comparison_diagnostics.csv
arima_301_full_results.csv
arima_001_full_results.csv
evaluation_sample.csv
forecast_accuracy.csv
forecast_accuracy_covid_comparison.csv
dm_hln_tests.csv
forecast_results.csv
```

`forecast_results.csv` contains the actual values, forecasts, and
forecast errors for each model, origin, target quarter, and horizon.

### Step 6: Check the figure outputs

The following figures are created in `Project/Output/`:

```text
figure_1_mexico_real_gdp.png
figure_2_log_gdp_and_growth.png
figure_3_acf_pacf.png
figure_A1_residual_diagnostics.png
out_of_sample_forecasts.png
```

### Step 7: Verify the forecast sample

A successful run should produce:

- 28 forecasts for horizon 1.
- 25 forecasts for horizon 4.
- 53 forecast observations in total.
- A final target quarter of 2025 Q4 for both horizons.
- No missing forecast values.

### Step 8: Render the Quarto dashboard

The Python analysis must be completed before rendering the dashboard
because the dashboard uses figures and results produced by `Code.py`.

If the dashboard is stored in the `dashboard/` folder, run:

```bash
cd dashboard
quarto render
cd ..
```

The rendered HTML files are saved in the output directory specified in
the dashboard's `_quarto.yml` file.

### Replication order

The required order is therefore:

```text
1. Install requirements.txt
2. Place the raw Excel file in Project/data/data/Raw Data/
3. Run Project/Code/Code.py
4. Verify the clean data, CSV tables, figures, and forecast counts
5. Render the Quarto dashboard
```

`Project.ipynb` is not required to reproduce the analysis. The official
replication file is `Project/Code/Code.py`.


### Folder structure:
- `Project/data/data/Raw Data/` — original INEGI workbook.
- `Project/data/data/Clean Data/` — cleaned quarterly GDP data.
- Code/ — all Python scripts used for this project.  
- Output/ — all figures and tables generated by the Python code.
- paper/ — Final PDF file of the research project. 
- dashboard/ — Quarto dashboard files.