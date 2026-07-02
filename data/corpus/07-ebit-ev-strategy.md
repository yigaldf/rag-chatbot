---
title: "EBIT to Enterprise Value Investment Strategy — Back Test"
source_url: https://www.quant-investing.com/blog/ebit-to-enterprise-value-investment-strategy-back-test
topic: systematic-value-investing
---

# EBIT to Enterprise Value Investment Strategy Back Test

The EBIT to Enterprise Value ratio (EBIT/EV), also known as **Earnings Yield**, is described as "the 80/20 valuation ratio when it comes to investing." The strategy identifies undervalued companies by comparing earnings before interest and taxes (EBIT) against enterprise value (EV). In the Quant Investing screener the terms "EBIT/EV" and "Earnings Yield" are interchangeable.

## The strategy

Rank the investable universe by Earnings Yield (EBIT/EV) and buy the cheapest companies — those with the highest earnings yield relative to their enterprise value. This is the same earnings-yield component that sits at the heart of Joel Greenblatt's Magic Formula and Tobias Carlisle's Acquirer's Multiple, used here as a standalone systematic value strategy.

## Back test results (1999–2011)

Over the 12-year period from 13 June 1999 to 13 June 2011 in European markets the strategy showed substantial outperformance:

- The cheapest 20% of companies (Q1) significantly outperformed the most expensive 20% (Q5).
- The approach worked best for medium and large companies.
- The broader European market returned only 30.54% in total (about 2.25% per year with dividends included) over the same period.

The back-test universe was roughly 1,500 companies across 17 Eurozone countries, excluding banks, insurance companies, investment funds, and REITs.

## Enhancing the strategy with momentum

Combining EBIT/EV with a momentum indicator substantially improved returns. Pairing the value rank with "Price Index 6 months" or "Price Index 12 months" (six- or twelve-month price momentum) produced the best results.

## How to implement it (screener steps)

1. Select the top 20% of companies with the highest Earnings Yield (EBIT/EV).
2. Add a second filter for the top 20% with the highest 6-month price momentum.
3. Choose your target countries.
4. Set a minimum daily trading volume (≈ $125,000 recommended).
5. Set a minimum market capitalization (≈ $65 million recommended).
6. Sort results by Earnings Yield from highest to lowest.

## Key takeaway

Rely on historical accounting data rather than analyst estimates: "forecasted numbers are just about useless." Analyst forecasts historically showed average errors of about 40%, so the strategy uses only reported figures.
