"""
PAPER 3 - SCRIPT 07b: build the GEE input table for IMERG extraction
=======================================================================
Reads samples_V2.npz and writes imerg_input_V2.csv -- the table to upload to
Earth Engine as an asset and point 07a_imerg_extract.js at.

KEY EFFICIENCY POINT
  IMERG is 0.1 degrees (~11 km). Every sample point inside the same 0.1-deg cell
  in the same month gets the SAME rainfall value -- extracting per point would do
  identical work twice. Verified on V2: 1,918 unique (point, year, month) combos
  collapse to 952 unique (cell, year, month) combos, so extracting per cell-month
  halves the Earth Engine job and loses nothing. The cell centroid is used as the
  extraction geometry; 07c_merge_imerg.py joins the result back onto every sample
  in that cell.

  The 0.1-deg IMERG grid is aligned to the same 0.1-deg convention already used by
  the `group` field in this project, so a cell here corresponds to a `group`. This
  is worth stating plainly in the paper: the rainfall trigger features are constant
  within a group, i.e. within an independence cluster. They cannot discriminate
  between neighbouring points -- their discriminative power is across TIME at a
  location, which is exactly what the month-and-year-matched V2 negative sampling
  design isolates.

USAGE
  python 07b_make_imerg_input.py          # writes imerg_input_V2.csv
Then:
  1. In Earth Engine: Assets > NEW > CSV upload, pick imerg_input_V2.csv
     (x = lon, y = lat), name it e.g. imerg_input_V2.
  2. Put that asset id into POINTS_ASSET at the top of 07a_imerg_extract.js.
  3. Run the script, then RUN each export task in the Tasks tab.
"""
import numpy as np
import pandas as pd

VARIANT = "V2"
CELL_DEG = 0.1


def main(variant=VARIANT):
    d = dict(np.load(f"samples_{variant}.npz", allow_pickle=True))
    df = pd.DataFrame({
        "point": d["point"],
        "lat": d["lat"],
        "lon": d["lon"],
        "year": d["year"].astype(int),
        "month_str": d["month"],
        "group": d["group"],
    })

    # `month` is stored as a 'YYYY_MM' string; GEE needs a numeric month
    df["month_num"] = df["month_str"].astype(str).str.split("_").str[1].astype(int)

    # sanity: the numeric month should agree with cal_month if present
    if "cal_month" in d:
        cal = pd.Series(d["cal_month"]).astype(int)
        mismatch = (cal.values != df["month_num"].values).sum()
        print(f"month_num vs cal_month mismatches: {mismatch} (expect 0)")

    # 0.1-degree cell index -- the unit IMERG actually resolves
    df["cell_i"] = np.floor(df["lat"] / CELL_DEG).astype(int)
    df["cell_j"] = np.floor(df["lon"] / CELL_DEG).astype(int)
    df["cell"] = df["cell_i"].astype(str) + "_" + df["cell_j"].astype(str)

    # one extraction per (cell, year, month); use the cell centroid
    cells = (df.drop_duplicates(["cell", "year", "month_num"])
               .loc[:, ["cell", "cell_i", "cell_j", "year", "month_num"]]
               .copy())
    cells["lat"] = (cells["cell_i"] + 0.5) * CELL_DEG
    cells["lon"] = (cells["cell_j"] + 0.5) * CELL_DEG
    # GEE script expects a 'point' column as the row id
    cells["point"] = cells["cell"] + "|" + cells["year"].astype(str) + "|" + \
                     cells["month_num"].astype(str)

    out = cells[["point", "lat", "lon", "year", "month_num", "cell"]]
    out.to_csv(f"imerg_input_{variant}.csv", index=False)

    print(f"\nsample rows                     : {len(df)}")
    print(f"unique (point, year, month)     : {len(df.drop_duplicates(['point','year','month_num']))}")
    print(f"unique (cell, year, month)      : {len(out)}   <- rows to extract in GEE")
    print(f"\nper-year export sizes:")
    print(out.groupby("year").size().to_string())
    print(f"\nwrote imerg_input_{variant}.csv")

    # also save the sample->cell mapping so the merge step is unambiguous
    df[["point", "year", "month_num", "cell", "group"]].to_csv(
        f"sample_cell_map_{variant}.csv", index=False)
    print(f"wrote sample_cell_map_{variant}.csv (join key for 07c)")


if __name__ == "__main__":
    main()
