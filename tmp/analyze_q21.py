from __future__ import annotations

import pandas as pd


telemetry = pd.read_csv(
    r"data\processed\aacc7c1ab3d9\telemetry.csv.gz",
    usecols=["timestamp", "ht:Q20", "ht:Q21"],
    parse_dates=["timestamp"],
)
quality = pd.read_csv(
    r"data\processed\aacc7c1ab3d9\quality.csv.gz",
    usecols=["signal_id", "source", "measured_at", "value"],
    parse_dates=["measured_at"],
)
pak = quality.loc[
    quality["signal_id"].eq("ht:2:Mg.Sulfur") & quality["source"].eq("pak"),
    ["measured_at", "value"],
].rename(columns={"measured_at": "timestamp", "value": "pak_sulfur"})
merged = telemetry.merge(pak, on="timestamp", how="inner")

for column in ("ht:Q20", "ht:Q21"):
    values = telemetry[column].dropna()
    print("\n", column)
    print("count", len(values), "missing", telemetry[column].isna().sum())
    print("quantiles", values.quantile([0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1]).to_dict())
    print(
        "gt_10",
        int((values > 10).sum()),
        "gt_50",
        int((values > 50).sum()),
        "gt_100",
        int((values > 100).sum()),
    )
    near_307 = telemetry.loc[values.index[(values - 307).abs() < 0.5], ["timestamp", column]]
    print("near_307_count", len(near_307))
    print(near_307.head(20).to_string(index=False))

valid = merged[["ht:Q21", "pak_sulfur"]].dropna()
print("\nexact timestamp overlap", len(valid))
print("pearson", valid.corr().iloc[0, 1])
print("spearman", valid.corr(method="spearman").iloc[0, 1])
diff = valid["ht:Q21"] - valid["pak_sulfur"]
print(
    "median q21-pak", diff.median(), "mae", diff.abs().mean(), "p95_abs", diff.abs().quantile(0.95)
)
clean = valid.loc[valid["ht:Q21"].between(0, 50)]
print(
    "clean_overlap",
    len(clean),
    "pearson",
    clean.corr().iloc[0, 1],
    "mae",
    (clean["ht:Q21"] - clean["pak_sulfur"]).abs().mean(),
)

joined = telemetry.set_index("timestamp").join(pak.set_index("timestamp"), how="inner")
for lag in range(-12, 13):
    corr = joined["ht:Q21"].shift(lag).corr(joined["pak_sulfur"])
    print("lag_steps", lag, "corr", corr)
