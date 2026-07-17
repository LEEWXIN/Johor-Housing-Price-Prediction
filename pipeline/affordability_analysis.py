"""
affordability_analysis.py
================
BDS23124 - Johor Property DSS
First-pass affordability analysis, added in response to feedback that the project
should explore supply-and-demand / buyer-affordability context beyond the listing
price data itself.

Data:
  - johor_final_clean.csv   (this project's cleaned listing data, 2,511 rows,
                              18 property-level areas)
  - johor_income.csv        (DOSM-style district household income, 2019/2022/2024)
  - johor_poverty.csv       (DOSM-style district poverty rate, 2019/2022/2024)

Note on granularity: the property dataset's 18 "areas" are township-level
(e.g. Skudai, Iskandar Puteri, Pasir Gudang), while income/poverty are published
at the 10 official Johor districts (daerah) level. MAP below assigns each
township-level area to its parent official district. Rows whose area is "Other"
(637 of 2,511, ~25%) cannot be mapped and are excluded from this analysis -
this is disclosed as a coverage limitation in the report.

Metric: price-to-income ratio = district median house price / district median
annual household income (2024). This is a standard housing-affordability
measure (higher = less affordable).

Run:
    python affordability_analysis.py
"""
import pandas as pd

MAP = {
    "Johor Bahru": "Johor Bahru", "Skudai": "Johor Bahru", "Iskandar Puteri": "Johor Bahru",
    "Johor Jaya": "Johor Bahru", "Mount Austin": "Johor Bahru", "Bukit Indah": "Johor Bahru",
    "Taman Molek": "Johor Bahru", "Gelang Patah": "Johor Bahru", "Permas Jaya": "Johor Bahru",
    "Pasir Gudang": "Johor Bahru", "Ulu Tiram": "Johor Bahru",
    "Kulai": "Kulai", "Kluang": "Kluang", "Batu Pahat": "Batu Pahat", "Pontian": "Pontian",
    "Segamat": "Segamat", "Muar": "Muar", "Other": None,
}


def main():
    df = pd.read_csv("johor_final_clean.csv")
    inc = pd.read_csv("johor_income.csv")
    pov = pd.read_csv("johor_poverty.csv")

    df["daerah"] = df["Area"].map(MAP)
    n_total = len(df)
    n_unmapped = df["daerah"].isna().sum()
    mapped = df.dropna(subset=["daerah"])
    print(f"[MAP] {n_total:,} listings -> {n_total - n_unmapped:,} mapped to an official "
          f"district ({(n_total - n_unmapped) / n_total:.0%}); {n_unmapped:,} ('Other') excluded")

    med_price = mapped.groupby("daerah")["Price_RM"].median().rename("median_price_rm")
    n_by = mapped.groupby("daerah").size().rename("n_listings")

    inc24 = inc[inc["date"] == "2024-01-01"].set_index("district")["income_median"]
    pov24 = pov[pov["date"] == "2024-01-01"].set_index("district")[["poverty_absolute", "poverty_relative"]]

    tab = pd.concat([med_price, n_by, inc24.rename("income_median_monthly"), pov24], axis=1)
    tab = tab.dropna(subset=["median_price_rm"])
    tab["annual_income"] = tab["income_median_monthly"] * 12
    tab["price_to_income"] = (tab["median_price_rm"] / tab["annual_income"]).round(1)
    tab = tab.sort_values("price_to_income", ascending=False)

    print("\n" + "=" * 78)
    print(tab[["median_price_rm", "n_listings", "annual_income", "price_to_income",
                "poverty_relative"]].to_string())
    print("=" * 78)

    tab.to_csv("affordability_by_district.csv")
    print("\n[SAVE] affordability_by_district.csv written")


if __name__ == "__main__":
    main()
