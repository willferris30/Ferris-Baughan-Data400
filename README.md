# Predicting Equity Returns: Integrating Financial Ratios and Sentiment
### Will Ferris & Charlie Baughan

## Overview

This project builds a machine learning-based stock selection model using S&P 500 data from 2011–2025. The project combines financial ratios, market-based indicators, volatility measures, and sentiment data to predict next-year equity returns and construct a systematic portfolio strategy.

The central goal is to evaluate whether a data-driven model can identify stocks with stronger future returns than the broader market.

## Project Objective

- Collect and clean historical S&P 500 financial data
- Build a stock-year panel dataset from 2011–2025
- Engineer next-year return targets
- Incorporate sentiment data from 2023–2025
- Use machine learning to identify important predictors of future returns
- Rank stocks into quintiles based on model scores
- Compare model performance against market averages and baseline strategies
- Construct a diversified 2026 portfolio using 2025 data

## Data

The analysis uses Bloomberg financial and market data across S&P 500 companies from 2011–2025.

### Main Data Sources

- Historical S&P 500 financial variables
- Bloomberg financial ratios
- Daily news sentiment data
- GICS sector classifications

### Final Dataset

After cleaning and filtering, the final modeling dataset includes:

- 7,118 stock-year observations
- 504 unique tickers
- Data from 2011 through 2025
- 502 stocks available in the 2025 portfolio input
- Sector coverage across 11 GICS sectors

## Key Variables

The model uses financial, market, volatility, valuation, and sentiment features. The most important predictors from the Random Forest model were:

| Feature | Importance |
|---|---:|
| CUR_MKT_CAP | 0.172286 |
| PX_LAST | 0.091918 |
| CURRENT_EV_TO_T12M_EBITDA | 0.086401 |
| VOLATILITY_360D | 0.081129 |
| VOLATILITY_90D | 0.077854 |
| sentiment_mean | 0.055593 |
| VOLUME_AVG_30D | 0.043226 |
| BS_TOT_ASSET | 0.038477 |
| VOLATILITY_30D | 0.038207 |
| RETURN_ON_ASSET | 0.037775 |
| EBITDA_MARGIN | 0.036586 |
| CURR_ENTP_VAL | 0.029527 |
| GROSS_MARGIN | 0.027948 |
| RETURN_COM_EQY | 0.026606 |
| sentiment_obs | 0.025963 |

## Methodology

### 1. Data Cleaning

The raw Bloomberg files were reshaped from a wide format into a stock-year panel. Tickers were standardized, benchmark rows were removed, missingness was evaluated, and variables with excessive missing data were dropped.

### 2. Target Variable

The model predicts next-year returns using:

```python
return_next_year = px_next_year / PX_LAST - 1
```

### 3. Sentiment Engineering

Daily sentiment data was aggregated into yearly stock-level features, including:

- sentiment_mean  
- sentiment_std  
- sentiment_median  
- sentiment_min  
- sentiment_max  
- sentiment_obs (number of observations)

These variables capture both the average tone and dispersion of sentiment for each stock.

Sentiment data coverage is used in the later years of the sample (2023–2025). These features were merged into the broader 2011–2025 dataset, allowing the model to evaluate the incremental contribution of sentiment alongside traditional financial variables.

While sentiment is not the dominant predictor, it provides additional explanatory power, particularly when combined with valuation, volatility, and size-related factors.

### 4. Machine Learning Model

A Random Forest Regressor was used to identify nonlinear relationships between company characteristics and future returns.

Model settings:

RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

The model was trained on historical stock-year observations with available next-year returns.

### 5. Scoring System

The top model features were standardized by year and averaged into a composite score. Stocks were then ranked into quintiles from lowest score to highest score.

Top selected features:

- CUR_MKT_CAP
- PX_LAST
- CURRENT_EV_TO_T12M_EBITDA
- VOLATILITY_360D
- VOLATILITY_90D
- sentiment_mean

### Results

Average Return by Quintile

Quintile: Average Next-Year Return
- Q1: 11.77%
- Q2: 14.40%
- Q3: 14.45%
- Q4: 18.36%
- Q5: 25.68%

The top quintile produced the highest average next-year return, suggesting that the model successfully ranked stocks by expected performance.

### Statistical Validation: Regression Analysis

To validate that the model’s performance is not driven by randomness, we conducted a series of cross-sectional regression analyses using over 6,000 stock-year observations from 2011–2024.

The goal of this analysis is to test whether the model’s ranking signal is statistically associated with future returns, while controlling for time and sector effects.

Model Specification

We estimate regressions of the form:

Future returns as a function of:
Model score (standardized)
Top quintile indicator
Sentiment
Year fixed effects
Sector fixed effects

Key Results

| Variable        | Coefficient | Interpretation                                  |
| --------------- | ----------- | ----------------------------------------------- |
| Model Score (z) | 0.029***    | Higher-ranked stocks earn higher future returns |
| Top Quintile    | 0.052***    | Top 20% stocks outperform by ~5.2% annually     |
| Sentiment (z)   | 0.035***    | Positive sentiment predicts higher returns      |
| R²              | ~0.21       | Model explains meaningful variation in returns  |

This analysis demonstrates that the model is not simply overfitting historical data. Instead, it captures persistent relationships between firm characteristics, sentiment, and future stock performance.

### Strategy Comparison

| Strategy                | Top Quintile Return | Q5–Q1 Spread | Top Hit Rate | Top Quintile Sharpe |
| ----------------------- | ------------------: | -----------: | -----------: | ------------------: |
| Model WITH Sentiment    |              25.71% |       13.88% |       69.69% |               0.409 |
| Model WITHOUT Sentiment |              24.43% |       12.21% |       70.17% |               0.395 |
| Random                  |              17.25% |        0.37% |       70.01% |               0.411 |
| Earnings Yield          |              15.59% |       -6.24% |       68.16% |               0.473 |
| Low Volatility          |              11.36% |      -15.94% |       70.38% |               0.508 |

### Key Findings

The model’s top quintile returned approximately 25.7%, compared to a market average of approximately 16.9%.
The Q5–Q1 spread was positive, showing that the model created meaningful separation between high- and low-ranked stocks.
Sentiment improved the model’s spread and top-quintile return.
Market capitalization, price, valuation, volatility, and sentiment were among the strongest predictors.
Simple standalone strategies, such as earnings yield or low volatility, did not outperform the combined model.
These results are further supported by regression analysis, which confirms that the model’s ranking signal is statistically significant and not driven by randomness.

### Portfolio Simulation

A yearly rebalanced simulation was conducted using the top-quintile model portfolio.

Starting with $1,000 at the beginning of 2012:

| Strategy | Ending Value | 
| --------- | -------- |
Model Top Quintile | $20,579.96
Market Average | $8,321.74

The model portfolio outperformed the market average by approximately $12,258 over the simulation period.

### 2026 Portfolio Construction

Using 2025 data, the model ranks stocks for a 2026 portfolio. The final portfolio is diversified by sector, with a maximum of five stocks selected from each sector.

The resulting portfolio contains 50 stocks across sectors including:

Financials
Consumer Discretionary
Communication Services
Information Technology
Materials
Industrials
Health Care
Energy
Utilities
Real Estate
Consumer Staples


### Limitations

Sentiment data is only available for the later years of the sample.
The analysis does not include transaction costs, taxes, or portfolio turnover constraints.
Historical relationships may not persist in future market environments.
Some results may be influenced by market regime effects.
The model is intended for academic analysis, not as investment advice.

### Ethical and Practical Considerations

This project uses structured financial data and sentiment signals to evaluate equity selection. While the model shows strong historical performance, real-world investment decisions require careful consideration of risk, liquidity, market conditions, and regulatory constraints.

### Conclusion

This project demonstrates that combining financial ratios, volatility measures, valuation indicators, and sentiment data can improve systematic stock selection. The Random Forest model identifies meaningful predictors of next-year returns, ranks stocks into performance-based quintiles, and produces a top-quintile strategy that outperforms the market average over the historical testing period.
