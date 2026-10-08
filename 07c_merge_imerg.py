"""
PAPER 3 - SCRIPT 07c: merge IMERG trigger features back onto the samples
===========================================================================
Takes the per-cell-month CSVs exported from Earth Engine by 07a, joins them onto
every sample via (cell, year, month), and writes samples_V2_imerg.npz -- the same
arrays as samples_V2.npz plus a new `trigger` array (n_samples x n_trigger_feats)
and `trigger_names`.

USAGE
  # after downloading imerg_features_V2_20*.csv from Drive into this folder:
  python 07c_merge_imerg.py

WHAT IT CHECKS (rather than assuming the export went fine)
  - every sample gets a trigger row; reports how many did not and why
  - flags features that are constant or all-zero (a sign the GEE reducer returned
    nulls, which would silently look like "no rain" instead of "no data")
  - reports how many DISTINCT trigger vectors exist relative to independence
    clusters -- since IMERG is 0.1 deg, these features are constant within a
    group, and this quantifies exactly how much independent signal they add
  - compares the IMERG monthly total against the existing monthly rainfall channel
    already in seq[:, :, 4], as a cross-check that the two agree in magnitude
"""
import glob
import numpy as np
import pandas as pd

VARIANT = "V2"
TRIGGER_COLS = ["total_month", "max_30min", "max_1h", "max_3h", "max_6h",
                "max_12h", "max_24h", "wet_hours", "n_storms", "max_storm_total",
                "max_storm_dur_h", "max_storm_int", "ante_7d", "ante_15d",
                "ante_30d", "ante_45d", "api_k09"]


def main(variant=VARIANT):
    files = sorted(glob.glob(f"imerg_features_{variant}_*.csv"))
    if not files:
        raise SystemExit(
            f"No imerg_features_{variant}_*.csv found. Download the GEE exports first.")
    print(f"reading {len(files)} export files: {[f.split('/')[-1] for f in files]}")
    feats = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"IMERG rows: {len(feats)}")

    # the GEE 'point' column is "cell|year|month"; split it back out
    parts = feats["point"].astype(str).str.split("|", expand=True)
    feats["cell"] = parts[0]
    feats["year"] = parts[1].astype(int)
    feats["month_num"] = parts[2].astype(int)
    feats = feats.drop_duplicates(["cell", "year", "month_num"])
    print(f"unique (cell, year, month) in export: {len(feats)}")

    smap = pd.read_csv(f"sample_cell_map_{variant}.csv")
    print(f"sample rows to fill: {len(smap)}")

    merged = smap.merge(feats[["cell", "year", "month_num"] + TRIGGER_COLS],
                        on=["cell", "year", "month_num"], how="left")
    missing = merged[TRIGGER_COLS[0]].isna().sum()
    print(f"\nsamples with NO matching IMERG row: {missing} / {len(merged)}")
    if missing:
        miss = merged[merged[TRIGGER_COLS[0]].isna()]
        print("  missing (cell, year, month) combos:")
        print(miss[["cell", "year", "month_num"]].drop_duplicates().head(20).to_string(index=False))

    # ---- quality checks on the extracted features themselves ----
    print("\nfeature sanity:")
    for c in TRIGGER_COLS:
        col = merged[c]
        n_nan = col.isna().sum()
        n_zero = (col == 0).sum()
        flag = ""
        if col.nunique(dropna=True) <= 1:
            flag = "  <-- CONSTANT, suspicious"
        elif n_zero > 0.9 * len(col):
            flag = "  <-- >90% zero, check the GEE reducer returned real values"
        print(f"  {c:18s} nan={n_nan:5d}  zero={n_zero:5d}  "
              f"min={col.min():9.3f}  med={col.median():9.3f}  max={col.max():10.3f}{flag}")

    # ---- how much independent signal do these actually add? ----
    d = dict(np.load(f"samples_{variant}.npz", allow_pickle=True))
    n_groups = len(set(d["group"]))
    trig = merged[TRIGGER_COLS].round(4)
    n_distinct = len(trig.drop_duplicates())
    print(f"\ndistinct trigger vectors: {n_distinct} across {len(merged)} samples")
    print(f"distinct 0.1-deg groups in the data: {n_groups}")
    print("  (these features are constant within a group-month by construction --"
          " state this in the paper)")

    # ---- cross-check against the monthly rainfall channel already in seq ----
    seq_rain_month_mean = d["seq"][:, :, 4].mean(axis=1)
    both = pd.DataFrame({"imerg_total": merged["total_month"].values,
                         "seq_rain_mean": seq_rain_month_mean})
    ok = both.dropna()
    if len(ok) > 10:
        r = np.corrcoef(ok["imerg_total"], ok["seq_rain_mean"])[0, 1]
        print(f"\ncorrelation(IMERG monthly total, existing seq rainfall channel mean) = {r:.3f}")
        print("  a moderate-to-high positive correlation is expected and reassuring;")
        print("  near zero would mean one of the two rainfall sources is wrong.")

    # ---- write the augmented sample file ----
    trigger = merged[TRIGGER_COLS].to_numpy(dtype=np.float32)
    out = {k: v for k, v in d.items()}
    out["trigger"] = trigger
    out["trigger_names"] = np.array(TRIGGER_COLS)
    np.savez_compressed(f"samples_{variant}_imerg.npz", **out)
    print(f"\nwrote samples_{variant}_imerg.npz  "
          f"(trigger shape {trigger.shape})")
    print("Feature sets for the next modelling run:")
    print("  E = terrain + trigger            (13 + %d)" % len(TRIGGER_COLS))
    print("  F = terrain + x90 + trigger      (13 + 90 + %d)" % len(TRIGGER_COLS))


if __name__ == "__main__":
    main()
