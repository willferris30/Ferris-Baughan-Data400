# Predicting Equity Returns with Machine Learning and Sentiment  
## Will Ferris and Charlie Baughan

## Overview  
Welcome — this project explores whether data-driven methods can systematically outperform the stock market.  

Beating the market consistently is a long-standing challenge, and traditional strategies (such as value or low volatility) often produce inconsistent results. At the same time, new data sources and machine learning techniques offer an opportunity to uncover patterns that may not be visible through conventional analysis.  

This project builds a machine learning model to predict next-year stock returns for S&P 500 companies using financial metrics and sentiment data. The goal is to determine whether a structured, data-driven approach can identify high-performing stocks and generate consistent outperformance.

---

## Data Sources  
- S&P 500 financial and market data (2011–2025)  
- Bloomberg-style financial ratios (valuation, profitability, volatility)  
- News-based sentiment data (aggregated at the stock-year level)  
- GICS sector classifications  

**Final dataset:**
- ~7,000 stock-year observations  
- ~500 companies per year  

---

## Key Result 1: Model Identifies High-Return Stocks  
Stocks were ranked into quintiles based on a machine learning score.

- Top Quintile (Q5): **25.7% avg return**  
- Bottom Quintile (Q1): **~11–12% avg return**  
- Spread (Q5 – Q1): **~13.9%**

The model successfully separates high- and low-performing stocks.

---

## Key Result 2: Statistical Validation
Regression analysis confirms the model captures real signals:

- Model score: **+2.9% return per standard deviation**  
- Top quintile: **+5.2% excess return**  
- Sentiment: **positive and statistically significant**  
- R² ≈ **0.21**

Results remain significant after controlling for year and sector effects.

---

## Takeaway  
Combining financial data with sentiment improves stock selection.  

The model consistently identifies higher-return stocks and generates meaningful outperformance relative to the market, supported by both backtesting and statistical validation.
