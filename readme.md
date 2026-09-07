# Extension of Henriques & Sadorsky (2008)

## Oil Prices and the Stock Prices of Alternative Energy Companies

Replication and data update of:

> Henriques, I., & Sadorsky, P. (2008). *Oil prices and the stock prices of alternative energy companies*. **Energy Economics, 30(3), 998–1010.**

This project replicates the paper using an extended dataset through 2026. The analysis examines the relationships among clean-energy stocks, technology stocks, crude oil prices, and short-term interest rates using a VAR framework, Toda–Yamamoto modified Wald tests, and Pesaran–Shin generalized impulse response functions.

The project also implements the paper's robustness tests by comparing the technology-stock variable with an orthogonalized technology-stock measure that removes the component explained by the broad stock market.

---

## 1. Research Question

The original paper investigates the relationship between alternative-energy stock prices and several economic and financial variables, particularly:

- alternative-energy stock prices,
- technology stock prices,
- crude oil prices, and
- interest rates.

The central question motivating this replication is:

> **Do oil prices and technology-stock prices contain predictive information about clean-energy stock prices, and how do shocks to these variables affect clean-energy equities over time?**

The updated replication asks whether the relationships identified in the original early-2000s sample remain present in a longer period.

---

# 2. Original Paper

Henriques and Sadorsky (2008) examine weekly data using a four-variable VAR framework.

The original variables are:

| Variable | Description |
|---|---|
| ECO | WilderHill Clean Energy Index |
| PSE | NYSE Arca Tech 100 Index |
| OIL | WTI crude oil price |
| RATE | 3-month U.S. Treasury bill rate |

The original analysis:

1. tests the variables for unit roots;
2. determines the VAR lag length;
3. estimates an augmented VAR following Toda and Yamamoto (1995);
4. conducts modified Wald tests for Granger causality;
5. calculates generalized impulse response functions;
6. performs a robustness check using an orthogonalized technology-stock variable.

The original paper selects:

- **VAR lag length:** $k = 8$
- **Maximum order of integration:** $d_{\max} = 2$
- **Augmented VAR:** $VAR(k+d_{\max}) = VAR(10)$
  
This project follows the same overall econometric structure but determines these quantities using the updated data.

---

# 3. Data

The updated dataset contains weekly observations from **August 2004 through 2026**, using Wednesday observations.

The four variables used in the main model are:

| Variable | Series | Source |
|---|---|---|
| ECO | WilderHill Clean Energy Index | WilderHill |
| PSE | NYSE Arca Tech 100 | Yahoo Finance |
| OIL | WTI Crude Oil Futures (`CL=F`) | Yahoo Finance |
| RATE | 3-month U.S. Treasury Bill Secondary Market Rate (`DTB3`) | FRED |

The S&P 500 (`^GSPC`) is additionally collected for the robustness analysis.

### Why Wednesday observations?

Following Henriques and Sadorsky (2008), weekly observations are constructed using Wednesdays.

Wednesday observations were retained because they reduce the influence of varying numbers of trading days across weeks and are less affected by holidays than some other weekdays. When a Wednesday observation was unavailable, the previous available closing value was carried forward before selecting the Wednesday observations.

---

# 4. Data Preparation

The raw data are combined and cleaned in `src/data.py`.

The main data-processing steps are:

1. import the ECO historical data;
2. download PSE, WTI oil, and S&P 500 data;
3. retrieve the 3-month Treasury bill rate from FRED;
4. align the series by date;
5. carry forward the most recent available observation where necessary;
6. retain Wednesday observations;
7. save the cleaned weekly dataset.

The resulting dataset is used for all subsequent analysis.

---

# 5. Log Transformation

Following Henriques and Sadorsky (2008), natural logarithms are applied to the ECO, PSE, oil-price, and Treasury-bill-rate series.

The main transformed variables are therefore:

$$
LECO_t = \ln(ECO_t)
$$

$$
LPSE_t = \ln(PSE_t)
$$

$$
LOIL_t = \ln(OIL_t)
$$

$$
LRATE_t = \ln(RATE_t)
$$

### Treasury bill rate adjustment

The extended sample contains seven weekly observations for which the Treasury bill rate is zero or slightly negative. Since the natural logarithm is undefined for non-positive values, these observations cannot be transformed directly.

For the replication, the RATE series was therefore floored at:

$$
0.0001
$$

before applying the natural logarithm:

```python
LRATE = np.log(RATE.clip(lower=0.0001))
```

The original RATE observations are **not changed in the raw dataset**; the floor is applied only during the logarithmic transformation.

This is a methodological deviation from a literal log transformation and is disclosed here for transparency.

---

# 6. Unit Root Tests

The integration properties of the variables are examined using three complementary tests:

- Augmented Dickey–Fuller (ADF)
- Phillips–Perron (PP)
- Kwiatkowski–Phillips–Schmidt–Shin (KPSS)

### Hypotheses

#### ADF

$$
\begin{aligned}
H_0 &: \text{series contains a unit root} \\
H_1 &: \text{series is stationary}
\end{aligned}
$$

#### Phillips–Perron

$$
\begin{aligned}
H_0 &: \text{series contains a unit root} \\
H_1 &: \text{series is stationary}
\end{aligned}
$$

#### KPSS

$$
\begin{aligned}
H_0 &: \text{series is stationary} \\
H_1 &: \text{series is non-stationary}
\end{aligned}
$$

ADF and KPSS tests are implemented with an intercept specification. ADF lag selection uses the Schwarz/Bayesian Information Criterion (BIC).

Each variable is tested both:

- in levels;
- in first differences.

---

## 6.1 Unit Root Results

The main results are:

| Variable | Level | First Difference |
|---|---|---|
| LECO | I(1) | Stationary |
| LPSE | I(1) | Stationary |
| LOIL | I(0) | Stationary |
| LRATE | Mixed evidence at level | Stationary |

More specifically:

- **LECO:** ADF and PP fail to reject a unit root; KPSS rejects stationarity. First difference is stationary.
- **LPSE:** ADF and PP fail to reject a unit root; KPSS rejects stationarity. First difference is stationary.
- **LOIL:** ADF and PP reject a unit root and KPSS does not reject stationarity. Oil is treated as I(0).
- **LRATE:** ADF and PP reject a unit root, while KPSS rejects stationarity at conventional levels. The first difference is stationary.

There is **no evidence of an I(2) variable**.

Therefore:

$$
\boxed{d_{\max} = 1}
$$

is used in the updated analysis.

This differs from the original paper, which uses $$d_{\max}=2$$.

---

# 7. VAR Lag Selection

A VAR allows each variable to depend on:

1. its own past values; and
2. the past values of the other variables.

The lag length determines how many previous observations enter the system.

The lag length is selected using a sequential **likelihood-ratio (LR) procedure**.

For the updated main model, the procedure evaluates candidate VARs from lag 1 through lag 15.

To ensure that the models being compared use the same effective sample, the estimation uses a common sample based on the maximum lag.

For each candidate lag:

1. estimate the VAR;
2. obtain the maximum-likelihood residual covariance matrix;
3. calculate its determinant;
4. compare it with the covariance determinant from the previous lag;
5. calculate the LR statistic.

The LR statistic is calculated as:

$$LR = T
\left[
\ln|\hat{\Sigma}_{k-1}|

\ln|\hat{\Sigma}_{k}|
\right]
$$

where:

- $T$ is the common effective sample size;
- $\hat{\Sigma}_{k-1}$ is the residual covariance matrix from the previous lag;
- $\hat{\Sigma}_{k}$ is the residual covariance matrix from the current lag.

The LR statistic is evaluated against a chi-square distribution with:

$$
K^2=4^2=16
$$

degrees of freedom.

The significance level is:

$$
\alpha=0.05
$$

---

## 7.1 Main Model Lag Selection

The sequential LR procedure gives:

$$
\boxed{k=9}
$$

Therefore, the main model uses nine unrestricted lags.

---

# 8. Toda–Yamamoto Augmented VAR

Because the variables are not all stationary in levels, standard VAR-based Granger causality tests can have non-standard distributions.

The Toda–Yamamoto approach avoids requiring pre-testing and differencing of the variables for the causality test.

The procedure estimates an augmented VAR containing:

$$
k+d_{\max}
$$

lags.

For the updated main model:

$$
k=9
$$

and:

$$
d_{\max}=1
$$

Therefore:

$$
9+1=10
$$

and the estimated model is:

$$
\boxed{VAR(10)}
$$

The first nine lags are used in the modified Wald tests, while the additional lag is included to account for the maximum integration order.

---

# 9. Model Diagnostics

Before conducting causality analysis, the estimated VAR is examined for residual serial correlation.

Two diagnostic measures are used.

## 9.1 Adjusted R-squared

Equation-level adjusted $R^2$ values are reported to describe how much variation in each dependent variable is explained by the estimated lag structure, accounting for the number of parameters.

Main-model values:

| Equation | Adjusted $R^2$ |
|---|---:|
| LECO | 0.991 |
| LPSE | 0.999 |
| LOIL | 0.969 |
| LRATE | 0.932 |

These values indicate substantial in-sample explanatory power, although high $R^2$ values alone do not establish causal relationships or model adequacy.

## 9.2 Multivariate LM Test

A multivariate Lagrange Multiplier test is used to examine residual serial correlation.

The null hypothesis is:

$$
H_0:\text{no residual serial correlation at lag }h
$$

The alternative hypothesis is:

$$
H_1:\text{residual serial correlation at lag }h
$$

The test is evaluated at the 5% significance level.

Following the original paper, the following lags are examined:

- lag 1;
- lag 4;
- lag 12.

The main model produces:

| Lag | LM statistic | p-value |
|---|---:|---:|
| 1 | 12.39 | 0.7165 |
| 4 | 12.22 | 0.7289 |
| 12 | 15.53 | 0.4863 |

All p-values exceed 0.05.

Therefore, the null hypothesis of no residual serial correlation is not rejected at the tested lags.

---

# 10. Toda–Yamamoto Modified Wald Tests

The main objective of the causality analysis is to determine whether the lagged values of one variable contain predictive information about another variable.

For example, to test whether oil prices Granger-cause clean-energy stock prices, the following restrictions are imposed in the LECO equation:

$$ H_0:\quad \beta_{LOIL,1} = \beta_{LOIL,2} = \cdots = \beta_{LOIL,9} = 0 $$

The tenth lag is not included in the restrictions because it is the additional $d_{\max}$ lag.

The modified Wald statistic follows an asymptotic chi-square distribution:

$$
\chi^2(9)
$$

A rejection of $H_0$ indicates that the lagged values of the proposed causal variable contain statistically significant predictive information for the dependent variable.

The procedure is repeated for all 12 possible directional relationships among the four variables.

---

## 10.1 Main Model Results

The main-model Wald statistics are:

| Dependent | LECO | LPSE | LOIL | LRATE |
|---|---:|---:|---:|---:|
| **LECO** | — | 7.284 | 16.416* | 9.652 |
| **LPSE** | 15.112* | — | 18.583** | 12.808 |
| **LOIL** | 23.739*** | 28.002*** | — | 37.909*** |
| **LRATE** | 22.532*** | 16.838* | 33.760*** | — |

Significance levels:

- `***` $p<0.01$
- `**` $p<0.05$
- `*` $p<0.10$

At the 5% level, the updated sample provides evidence of predictive relationships including:

- LECO → LOIL
- LPSE → LOIL
- LRATE → LOIL
- LECO → LRATE
- LOIL → LPSE

The relationship LOIL → LECO is significant at the 10% level but not at 5%.

Importantly, these tests represent **Granger/predictive causality**, not structural or economic causality.

---

# 11. Generalized Impulse Response Functions

Granger-causality tests identify predictive relationships, but they do not show the dynamic response of one variable to an innovation in another variable.

To examine dynamic responses, the project calculates **Pesaran–Shin generalized impulse response functions (GIRFs)**.

The generalized impulse response to a one-standard-deviation innovation in variable $j$ is calculated as:

$$GIRF_j(h)=\frac{\Phi_h\Sigma e_j}{\sqrt{\sigma_{jj}}}$$

where:

- $\Phi_h$ is the moving-average coefficient matrix at horizon $h$;
- $\Sigma$ is the VAR residual covariance matrix;
- $e_j$ selects the shocked variable;
- $\sigma_{jj}$ is the variance of the innovation.

Unlike orthogonalized impulse responses, generalized impulse responses do not require a particular ordering of the variables.

The responses are calculated over a:

$$
\boxed{10\text{-week horizon}}
$$

---

## 11.1 Confidence Intervals

Henriques and Sadorsky (2008) construct confidence intervals for the generalized impulse responses using **analytically calculated standard errors**.

This replication uses an alternative procedure.

The point estimates of the Pesaran–Shin generalized impulse responses are calculated directly from the estimated VAR. Uncertainty bands are then obtained using a **recursive residual bootstrap**.

The bootstrap procedure:

1. resamples VAR residuals;
2. recursively generates simulated datasets using the estimated VAR;
3. re-estimates the VAR for each simulated dataset;
4. recalculates the generalized impulse responses;
5. obtains the empirical standard deviation of the simulated GIRFs;
6. constructs approximately ±2 standard-error bands.

This alternative was adopted instead of implementing the analytical gradient-based variance calculation used in the original paper. It provides a computationally tractable method for quantifying sampling uncertainty around the generalized impulse responses.

This is an explicit methodological deviation from the original paper.

---

# 12. Robustness Check

The original paper conducts a robustness exercise to distinguish technology-specific movements from movements associated with the broader stock market.

The updated replication follows the same conceptual approach.

The S&P 500 is introduced as the broad-market control variable.

First, the following regression is estimated:

$$
LPSE_t
\alpha
+
\beta LSP500_t
+
\epsilon_t
$$

The residual:

$$
LOTECH_t=\hat{\epsilon}_t
$$

is used as the orthogonalized technology-stock variable.

LOTECH therefore represents the component of technology-stock prices that is not explained by the S&P 500 within this regression.

The robustness VAR contains:

- LECO
- LOTECH
- LOIL
- LRATE

---

## 12.1 Robustness Unit Roots

The same ADF, PP, and KPSS procedure is applied.

LOTECH is found to be non-stationary in levels but stationary after first differencing.

Therefore:

$$
d_{\max}=1
$$

remains appropriate for the robustness specification.

---

## 12.2 Robustness Lag Selection

For comparability and to avoid an excessively parameterized robustness VAR, the LR lag search is restricted to a maximum of 12 lags.

The sequential LR procedure selects:

$$
\boxed{k=12}
$$

The robustness model therefore contains:

$$
12+1=13
$$

lags:

$$
\boxed{VAR(13)}
$$

The 12-lag maximum is a modelling choice for the robustness specification and is disclosed rather than presented as an unconstrained global optimum.

---

## 12.3 Robustness Diagnostics

The robustness model produces:

| Equation | Adjusted $R^2$ |
|---|---:|
| LECO | 0.991 |
| LOTECH | 0.990 |
| LOIL | 0.969 |
| LRATE | 0.934 |

The multivariate LM tests give:

| Lag | LM statistic | p-value |
|---|---:|---:|
| 1 | 24.89 | 0.0717 |
| 4 | 24.91 | 0.0714 |
| 12 | 5.96 | 0.9885 |

The null hypothesis of no serial correlation is not rejected at any of the tested lags.

---

## 12.4 Robustness Causality Results

The robustness Wald statistics are:

| Dependent | LECO | LOIL | LOTECH | LRATE |
|---|---:|---:|---:|---:|
| **LECO** | — | 24.456** | 21.060** | 12.392 |
| **LOIL** | 26.222*** | — | 11.872 | 38.546*** |
| **LOTECH** | 33.350*** | 25.898** | — | 35.032*** |
| **LRATE** | 31.279*** | 44.201*** | 29.658*** | — |

A key result is:

$$
\boxed{LOTECH \rightarrow LECO,\quad p=0.0495}
$$

The orthogonalized technology-stock variable significantly predicts clean-energy stock prices at the 5% level.

By comparison, in the main model:

$$
LPSE \rightarrow LECO,\quad p=0.6076
$$

This suggests that the technology-specific component of technology-stock prices has predictive content for clean-energy equities even after removing the component associated with broad market movements.

Again, this should be interpreted as **predictive/Granger causality rather than structural causality**.

---

# 13. Main Results and Interpretation

The updated sample produces several differences from the original paper.

The main findings are:

### Technology and clean-energy stocks

The generalized impulse responses indicate a strong dynamic relationship between technology and clean-energy equities.

The robustness specification strengthens this result: the orthogonalized technology-stock component significantly predicts LECO.

### Oil and clean-energy stocks

Oil prices appear more relevant to clean-energy equities in the updated sample than in the original sample.

The main TY test finds:

$$
LOIL \rightarrow LECO
$$

at the 10% level, while the robustness specification finds the relationship significant at the 5% level.

The updated GIRFs also show a positive response of LECO to an oil-price innovation.

### Oil and technology stocks

The updated sample shows stronger interaction between oil and technology-stock variables than the original early-2000s sample.

### Interest rates

Interest rates are strongly interconnected with the other variables in the updated system, particularly in the oil and technology equations.

---

# 14. Replication vs. Original Paper

This project should be understood as a **data-updated replication**, rather than an exact reproduction of the original estimates.

| Component | Original Paper | This Project |
|---|---|---|
| Sample | Early 2000s | 2004–2026 |
| Frequency | Weekly | Weekly |
| Observation day | Wednesday | Wednesday |
| ECO | WilderHill Clean Energy | WilderHill Clean Energy |
| PSE | Technology-stock index | Technology-stock index |
| Oil | WTI | WTI |
| Interest rate | 3-month T-bill | 3-month T-bill |
| Unit-root tests | ADF, PP, KPSS | ADF, PP, KPSS |
| Lag selection | LR procedure | LR procedure |
| $k$ | 8 | 9 |
| $d_{\max}$ | 2 | 1 |
| Augmented VAR | VAR(10) | VAR(10) |
| GIRF | Pesaran–Shin | Pesaran–Shin |
| GIRF SEs | Analytical | Recursive residual bootstrap |
| Robustness | Orthogonalized technology variable | Orthogonalized technology variable |

The differences in results should therefore not be interpreted simply as coding discrepancies. They may reflect the substantially different economic environment and longer sample period.

---

# 15. Project Structure

```text
Henriques-and-Sadorsky-replication/
│
├── data/
│   ├── raw/
│   │   └── ECO_2004-2026.csv
│   │
│   └── cleaned/
│       ├── wed_data.csv
│       ├── log_data.csv
│       └── robust_log_data.csv
│
├── results/
│   ├── unit_root_tests.csv
│   ├── lag_selection.csv
│   ├── stat_table.csv
│   ├── pval_table.csv
│   ├── rob_unit_root_tests.csv
│   ├── rob_lag.csv
│   ├── rob_stat_table.csv
│   ├── rob_pval_table.csv
│   │
│   ├── figure.png
│   └── rob_figure.png
│
├── src/
│   ├── data.py
│   ├── transform.py
│   ├── tests.py
│   ├── var_model.py
│   ├── granger.py
│   ├── plots.py
│   └── robustness.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 16. Code Modules

### `src/data.py`

Responsible for:

- collecting external market data;
- cleaning and aligning the series;
- handling missing observations;
- constructing Wednesday observations;
- producing the cleaned dataset.

### `src/transform.py`

Responsible for:

- logarithmic transformations;
- construction of the transformed dataset;
- preparation of the S&P 500 variable used in the robustness analysis.

### `src/tests.py`

Implements:

- ADF;
- Phillips–Perron;
- KPSS;
- level and first-difference stationarity tests.

### `src/var_model.py`

Implements:

- likelihood-ratio lag selection;
- augmented VAR estimation;
- model diagnostics;
- multivariate LM tests.

### `src/granger.py`

Implements:

- Toda–Yamamoto augmented VAR Wald tests;
- all directional causality tests;
- p-value and significance-star tables.

### `src/plots.py`

Implements:

- Pesaran–Shin generalized impulse responses;
- recursive bootstrap simulations;
- ±2 standard-error uncertainty bands;
- impulse-response figures.

### `src/robustness.py`

Implements the orthogonalized technology-stock variable:

$$
LOTECH = LPSE-\widehat{LPSE}
$$

where predicted technology-stock prices are obtained from the S&P 500 regression.

---

# 17. Reproducibility

To reproduce the analysis:

### 1. Clone the repository

```bash
git clone <repository-url>
cd Henriques-and-Sadorsky-replication
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python main.py
```

The script sequentially:

1. cleans the data;
2. performs logarithmic transformations;
3. conducts unit-root tests;
4. selects the VAR lag length;
5. estimates the augmented VAR;
6. performs model diagnostics;
7. conducts Toda–Yamamoto causality tests;
8. calculates generalized impulse responses;
9. performs the robustness analysis;
10. saves the resulting tables and figures.

---

# 18. Limitations and Methodological Notes

Several limitations and deviations from the original paper should be kept in mind.

### 1. Updated ECO data

The WilderHill Clean Energy Index historical data available for the extended sample required consolidation of historical observations and subsequent reporting data.

### 2. Treasury bill rate transformation

Seven zero or slightly negative RATE observations required a positive floor before logarithmic transformation.

### 3. Different integration order

The updated unit-root tests provide no evidence of I(2) variables, resulting in:

$$
d_{\max}=1
$$

rather than the $d_{\max}=2$ used by the original paper.

### 4. GIRF uncertainty bands

The original paper uses analytical standard errors. This project uses recursive residual bootstrap standard errors.

### 5. Robustness lag ceiling

The robustness lag search is restricted to 12 lags to avoid an excessively parameterized robustness VAR and maintain a manageable comparison with the main specification.

### 6. Granger causality is not structural causality

The Toda–Yamamoto tests identify predictive relationships in the time-series system. They should not be interpreted as evidence of structural causal mechanisms.

---

# 19. Why This Replication?

The original paper studies an important question at the intersection of:

- energy economics;
- financial markets;
- technology;
- alternative energy;
- macroeconomic variables.

Updating the analysis to 2026 provides an opportunity to examine whether relationships identified in the early development of the clean-energy sector continue to hold after substantial changes in:

- renewable-energy investment;
- energy markets;
- technology markets;
- monetary policy;
- financial markets;
- the global energy transition.

The purpose of this project is not to force the updated data to reproduce the original results.

Instead, the objective is to **reproduce the original methodology as closely as possible, identify where the updated data produce different results, and understand those differences.**

---

# 20. References

Henriques, I., & Sadorsky, P. (2008). Oil prices and the stock prices of alternative energy companies. *Energy Economics, 30*(3), 998–1010.

Pesaran, H. H., & Shin, Y. (1998). Generalized impulse response analysis in linear multivariate models. *Economics Letters, 58*(1), 17–29.

Toda, H. Y., & Yamamoto, T. (1995). Statistical inference in vector autoregressions with possibly integrated processes. *Journal of Econometrics, 66*(1–2), 225–250.
