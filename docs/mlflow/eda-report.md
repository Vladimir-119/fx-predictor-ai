# ECB USD/EUR: data review

Observations: 7109. Dates: 1999-01-04 — 2026-10-07. Missing values: 0. Duplicate dates: 0.

The source quotes EUR/USD. We take its reciprocal: EUR per one USD. Only reference rates are used. Weekends and holidays are not filled with copies.

Minimum: 0.625391; maximum: 1.211827. Standard deviation of one-step changes: 0.005101.

Largest absolute changes (retained; a large move is not automatically an error):

```text
      date     rate    change
2000-09-22 1.124859 -0.048299
2022-11-11 0.970120 -0.034501
2008-12-19 0.717360  0.033178
2015-01-23 0.893017  0.032283
2009-03-19 0.731475 -0.030139
```

The level and volatility change over time, so we use chronological validation. The target is the next publication, not a weekend market quote. Training predicts a change; the API adds it to the last known rate. The last-value baseline is essential. News and causal effects are not studied here. This EDA checks data quality; model selection uses training/validation only.
