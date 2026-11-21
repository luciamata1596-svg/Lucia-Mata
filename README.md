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
## Research Project: Forecasting Mexico’s Quarterly Real GDP Growth
This project analyzes whether  ARIMA (0,0,1) produces more accurate one- and four-quarter-ahead forecasts of Mexico’s quarterly real GDP growth than historical-mean, random-walk, and AR(1) benchmarks. It also examines whether the results change after excluding the COVID-19 target quarters from 2020 Q2 through 2021 Q1. Forecast accuracy is determined by comparing RMSE and MAE, as well as DM-HLN tests, for the four forecasting methods. GDP forecasts help us understand future economic conditions, and a complicated model is valuable only if it outperforms a simple benchmark. Forecast accuracy results showed that the ARIMA (0,0,1) model did not consistently outperform the benchmarks, and the DM-HLN tests found no statistically significant difference; limitations include the short evaluation sample and structural breaks.
## Data sources=

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

The analysis was developed and tested using Python 3.12.15.

### Required Python packages

- pandas==[insert version]
- numpy==[insert version]
- matplotlib==[insert version]
- scipy==[insert version]
- statsmodels==[insert version]
- openpyxl==[insert version]

`openpyxl` is required for reading the original Excel data file. The
project also uses `pathlib` and `warnings`, which are included in the
Python standard library and do not require separate installation.

The required packages can be installed from the project root with:

```bash
python -m pip install -r requirements.txt

