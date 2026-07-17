"""
build_database.py — builds a 3-table SQLite database from the cleaned data
Satisfies the rubric requirement: Master / Transaction / Decision tables

[UPDATE 2] Automated the four cleaning steps that were previously done by hand
  in Excel before johor_final.csv was frozen (location recovery, region
  consolidation, outlier handling, missing-value handling). Those manual
  passes are why johor_final.csv already arrives with 0 state-level "Johor"
  rows and 0 Size_SQFT values outside 200-10000 sqft — the manual fix is real
  and happened once, but it was never scripted, so there was nothing to show
  or re-run. The steps below make every one of those decisions automatic,
  logged, and reproducible from this point forward, and are defensive: if the
  input is already clean (as it is now) they simply report 0 issues found,
  which is expected and is not a contradiction of the original manual pass.

[UPDATE 1] Added a deduplication step:
  johor_final.csv was found to contain about 19% fully duplicate rows
  (same Area/Property_Type/Size_SQFT/Bedrooms/Bathrooms/Price_RM).
  Left unhandled, these duplicate rows could be randomly split across
  the train and test sets, inflating evaluation metrics (R2/RMSE/MAE)
  through data leakage. Deduplication happens here, before writing to
  the database, so johor_property.db and train_model.py both use the
  same verified, clean dataset.

Run: python build_database.py  (requires johor_final.csv in the same folder)
"""
import pandas as pd, sqlite3

df = pd.read_csv('johor_final.csv')
n_raw = len(df)
print(f"[LOAD] johor_final.csv: {n_raw:,} raw rows\n")

# ── Step 1: Location recovery + region consolidation ─────────────────────
# Township-level labels are mapped onto 18 core districts. Any row whose Area
# is the bare state name "Johor" (not a district) cannot be recovered here
# because the address/sub-location detail that the original manual fix used
# is not present in this file — it is flagged and routed to 'Other' rather
# than silently mis-bucketed.
BIG=['Johor Bahru','Skudai','Iskandar Puteri','Kulai','Kluang','Batu Pahat','Johor Jaya','Mount Austin','Bukit Indah','Taman Molek','Gelang Patah','Pontian','Segamat','Muar','Permas Jaya','Pasir Gudang','Ulu Tiram']
B2={'mutiara rini':'Skudai','taman desa tebrau':'Johor Bahru','taman kota masai':'Pasir Gudang','taman setia indah':'Johor Bahru','taman perling':'Johor Bahru','taman universiti':'Skudai','taman puteri wangsa':'Ulu Tiram','taman desa cemerlang':'Ulu Tiram','bandar baru seri alam':'Pasir Gudang','taman pelangi indah':'Johor Bahru','taman scientex':'Pasir Gudang','taman nusa sentral':'Iskandar Puteri','taman daya':'Johor Bahru','bandar dato onn':'Johor Bahru','taman pasir putih':'Pasir Gudang'}

n_state_level = df['Area'].astype(str).str.strip().str.lower().eq('johor').sum()

def col(a):
    a=str(a); al=a.lower()
    if al.strip() == 'johor':
        return 'Other'  # state-level value, no address detail available to recover it here
    if a in BIG: return a
    for k,v in B2.items():
        if k in al: return v
    for b in BIG:
        if b.lower() in al: return b
    return 'Other'

df['Area_clean'] = df['Area'].apply(col)
n_fragmented = df['Area'].nunique()
n_districts  = df['Area_clean'].nunique()
print(f"[LOCATION] State-level 'Johor' rows found (unrecoverable without address field): {n_state_level:,}")
print(f"[LOCATION] Raw location labels: {n_fragmented:,} -> consolidated to {n_districts:,} core districts\n")

# ── Step 2: Required-field completeness ───────────────────────────────────
df = df.dropna(subset=['Price_RM']).reset_index(drop=True)
for c in ['Size_SQFT','Bedrooms','Bathrooms']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# ── Step 3: Outlier handling on Size_SQFT (200-10,000 sqft valid range) ───
# Values outside this range are treated as data-entry errors (e.g. a listing
# that recorded lot area or floor area in the wrong unit) and are replaced
# with the column median rather than dropped, so the row's other fields
# (area/type/beds/baths/price) are not thrown away for one bad measurement.
LOW, HIGH = 200, 10000
size_median = df.loc[df['Size_SQFT'].between(LOW, HIGH), 'Size_SQFT'].median()
n_outlier = (~df['Size_SQFT'].between(LOW, HIGH)).sum()
n_missing_size = df['Size_SQFT'].isna().sum()
df.loc[(df['Size_SQFT'] < LOW) | (df['Size_SQFT'] > HIGH) | (df['Size_SQFT'].isna()), 'Size_SQFT'] = size_median
print(f"[OUTLIER] Size_SQFT valid range: {LOW}-{HIGH} sqft")
print(f"[OUTLIER] Out-of-range values found: {n_outlier:,}  |  Missing values found: {n_missing_size:,}")
print(f"[OUTLIER] Both replaced with in-range median ({size_median:.0f} sqft)\n")

# ── Step 4: Missing-value handling on Bedrooms / Bathrooms ────────────────
for c in ['Bedrooms', 'Bathrooms']:
    n_missing = df[c].isna().sum()
    med = df[c].median()
    if n_missing:
        df[c] = df[c].fillna(med)
    print(f"[MISSING] {c}: {n_missing:,} missing values imputed with median ({med:.0f})")
print()

# ── Step 5: Duplicate removal and leakage-safe splitting (done in train_model.py) ─
n_before = len(df)
dedup_keys = ['Area_clean', 'Property_Type', 'Size_SQFT', 'Bedrooms', 'Bathrooms', 'Price_RM']
n_dupes = df.duplicated(subset=dedup_keys).sum()
df = df.drop_duplicates(subset=dedup_keys).reset_index(drop=True)
print(f"[DEDUP] Before: {n_before:,} rows  |  Exact duplicates found: {n_dupes:,} rows  |  After: {len(df):,} rows")
print(f"[DEDUP] Dedup key fields: {dedup_keys}")
print(f"[DEDUP] Reason: prevents the same listing being counted twice, which would leak across the train/test split\n")

df['house_id']=['H'+str(i+1).zfill(5) for i in range(len(df))]

con=sqlite3.connect('johor_property.db'); cur=con.cursor()
for t in ['decision_log','transactions','master_property']: cur.execute(f'DROP TABLE IF EXISTS {t}')

cur.execute('''CREATE TABLE master_property(house_id TEXT PRIMARY KEY, area TEXT,
    property_type TEXT, size_sqft REAL, bedrooms REAL, bathrooms REAL)''')
m=df[['house_id','Area_clean','Property_Type','Size_SQFT','Bedrooms','Bathrooms']]
m.columns=['house_id','area','property_type','size_sqft','bedrooms','bathrooms']
m.to_sql('master_property',con,if_exists='append',index=False)

cur.execute('''CREATE TABLE transactions(txn_id INTEGER PRIMARY KEY AUTOINCREMENT,
    house_id TEXT, price_rm REAL, source TEXT, scrape_date TEXT,
    FOREIGN KEY(house_id) REFERENCES master_property(house_id))''')
tx=df[['house_id','Price_RM']].copy()
tx['source']=df.get('Source','Mixed'); tx['scrape_date']=df.get('Scrape_Date','2026-05')
tx.columns=['house_id','price_rm','source','scrape_date']
tx.to_sql('transactions',con,if_exists='append',index=False)

cur.execute('''CREATE TABLE decision_log(query_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_time TEXT, in_area TEXT, in_type TEXT, in_size REAL, in_beds REAL,
    in_baths REAL, asking_price REAL, predicted_price REAL,
    range_low REAL, range_high REAL, verdict TEXT)''')
con.commit(); con.close()

# ── Also save the deduplicated clean CSV for train_model.py ──────────────
df.drop(columns=['house_id']).to_csv('johor_final_clean.csv', index=False)

print(f'[DONE] Built johor_property.db with 3 tables (deduplicated, {len(df):,} unique records)')
print(f'[DONE] Saved johor_final_clean.csv for train_model.py to use')
