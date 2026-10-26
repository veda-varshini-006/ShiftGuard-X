#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB135-R — NB118-Matched Decimal Consistency Audit

**Objective:** Test decimal-accounting consistency on the **same source-row window** used in NB118: first 125,000 source rows for Yellow Jan, first 125,000 for Yellow Feb, and all 53,577 Green Feb source rows. This is **NOT an all-row audit of the yellow Parquet files**. Filtering differences are explicitly logged.

**Crucial limitation:** `Decimal(str(float(x)))` recovers the shortest round-trip decimal printing of a stored float, *not* the upstream billing amount. Matching decimal sums do not identify a vendor's arithmetic procedure.

**Replication control:** The five exploratory addition orders in NB135 are not demonstrated equivalent to NB118's named implementations (`left_to_right`, `reverse`, `balanced`, `tip_last`, `taxes_fees_first`). Therefore `extra_exact_path` is labeled **exploratory**, never an NB118 replication statistic. Compare it with published NB118 counts only as a diagnostic; disagreement is not evidence of a changed source.

**Schema control:** NYC TLC uses `Airport_fee` in the Yellow schemas; earlier NB135 checked only lowercase `airport_fee`, accidentally setting the fee to zero. This revision resolves charge fields case-insensitively and records the mapping.

This notebook is a falsification-oriented audit, not an invention or evidence of compression novelty.
"""

# %pip -q install numpy pandas pyarrow


import io,os,json,zipfile,hashlib,urllib.request,platform,math
from pathlib import Path
from decimal import Decimal, ROUND_HALF_EVEN
import numpy as np, pandas as pd, pyarrow.parquet as pq
OUT=Path('/content/SGX_NB135_results');OUT.mkdir(exist_ok=True)
URLS={'yellow_2024_01':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet',
      'yellow_2024_02':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet',
      'green_2024_02':'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-02.parquet'}
CHARGES=['fare_amount','extra','mta_tax','tip_amount','tolls_amount','improvement_surcharge','congestion_surcharge','airport_fee']
LIMIT=125000
print('Environment:',platform.python_version(),np.__version__,pd.__version__)


def read_data(label,url):
 path=Path('/content')/(label+'.parquet')
 if not path.exists():
  print('Downloading',label,flush=True);urllib.request.urlretrieve(url,path)
 pf=pq.ParquetFile(path); present=pf.schema_arrow.names
 fieldmap={x:next((actual for actual in present if actual.casefold()==x.casefold()),None) for x in CHARGES}
 columns=['total_amount']+[fieldmap[x] for x in CHARGES if fieldmap[x] is not None]+(['VendorID'] if 'VendorID' in present else [])
 frames=[];count=0
 for batch in pf.iter_batches(columns=columns,batch_size=32768):
  frames.append(batch.to_pandas());count+=len(frames[-1])
  if count>=LIMIT:break
 df=pd.concat(frames,ignore_index=True).iloc[:LIMIT].copy()
 for canonical,actual in fieldmap.items():
  if actual is not None and actual != canonical:df.rename(columns={actual:canonical},inplace=True)
 for x in CHARGES:
  if x not in df:df[x]=0.0   # absent field, not row-level missing
 for x in CHARGES+['total_amount']:
  df[x]=pd.to_numeric(df[x],errors='coerce')
 df['VendorID']=pd.to_numeric(df['VendorID'],errors='coerce').fillna(-1).astype('int64') if 'VendorID' in df else -1
 valid=np.isfinite(df[CHARGES+['total_amount']].to_numpy(dtype=np.float64)).all(axis=1)
 return df.loc[valid].reset_index(drop=True),dict(source=url,original_sampled_rows=len(df),eligible_rows=int(valid.sum()),excluded_rows=int((~valid).sum()),present_columns=columns,absent_optional_columns=[x for x in CHARGES if fieldmap[x] is None],resolved_column_mapping=fieldmap,parquet_source_rows=pf.metadata.num_rows)
DATA={};META={}
for key,url in URLS.items():DATA[key],META[key]=read_data(key,url)
print(json.dumps(META,indent=2))


"""
## Decimal-accounting consistency comparison

The exact sum is formed from decimal strings of the eight charge floats. `decimal_match` means equality to the recorded total *as printed decimal*, not IEEE-754 float bit equality. `cent_match` means both totals quantize to equal cents. A negative difference or out-of-range amount is kept if finite; exclusions are reported. Vendor comparison is **descriptive**, not a causal inference.
"""

def dec(x):return Decimal(str(float(x)))
def cents(x):return x.quantize(Decimal('0.01'),rounding=ROUND_HALF_EVEN)
def addition_paths(row):
 a=[float(v) for v in row]
 orders=[list(range(8)),list(reversed(range(8))),[0,2,4,6,1,3,5,7],[7,5,3,1,6,4,2,0],[0,1,3,2,4,5,7,6]]
 p=[]
 for order in orders:
  t=np.float64(0)
  for j in order:t=np.add(t,np.float64(a[j]),dtype=np.float64)
  p.append(np.asarray(t,dtype='<f8').view('<u8').item())
 return p
ROWS=[]
for name,df in DATA.items():
 for idx,(a,total,vendor) in enumerate(zip(df[CHARGES].to_numpy(dtype=np.float64),df.total_amount.to_numpy(dtype=np.float64),df.VendorID.to_numpy())):
  ds=sum((dec(x) for x in a),Decimal(0)); dt=dec(total)
  paths=addition_paths(a); target=np.asarray(total,dtype='<f8').view('<u8').item()
  alt=(paths[0]!=target and any(x==target for x in paths[1:]))
  ROWS.append(dict(dataset=name,index=idx,vendor=int(vendor),decimal_match=ds==dt,cent_match=cents(ds)==cents(dt),exploratory_extra_exact_path=alt,
                   nonzero_charge_count=int(np.count_nonzero(a)),negative_charge_count=int(np.count_nonzero(a<0)),total_abs_bin=int(min(20,abs(total)//20)),
                   decimal_difference=str(dt-ds) if idx<25 else ''))
 print('Finished',name,len(df),flush=True)
r=pd.DataFrame(ROWS)
r.to_csv(OUT/'row_flags.csv',index=False)
print('Rows processed',len(r))


"""
## Exploratory vendor comparison with charge-composition strata

Compare vendors within `(dataset, nonzero_charge_count, total_abs_bin)` strata, retaining only strata containing at least 30 records from **each** vendor. This is a descriptive stratified analysis, not causal identification. If large vendor effects persist, further transaction-level controls and provenance checks remain necessary.
"""

agg=r.groupby(['dataset','vendor']).agg(rows=('index','size'),decimal_match_rate=('decimal_match','mean'),cent_match_rate=('cent_match','mean'),extra_path_rate=('exploratory_extra_exact_path','mean')).reset_index()
display(agg)
agg.to_csv(OUT/'vendor_summary.csv',index=False)
strata=r[r.vendor.isin([1,2])].groupby(['dataset','nonzero_charge_count','total_abs_bin','vendor']).agg(n=('index','size'),rate=('exploratory_extra_exact_path','mean')).reset_index()
piv=strata.pivot(index=['dataset','nonzero_charge_count','total_abs_bin'],columns='vendor',values=['n','rate']).dropna()
valid=piv[(piv[('n',1)]>=30)&(piv[('n',2)]>=30)]
out=[]
for (dataset,charges,bin_),row in valid.iterrows():
 out.append(dict(dataset=dataset,nonzero_charge_count=charges,total_abs_bin=bin_,vendor1_n=int(row[('n',1)]),vendor2_n=int(row[('n',2)]),vendor1_rate=float(row[('rate',1)]),vendor2_rate=float(row[('rate',2)]),difference=float(row[('rate',2)]-row[('rate',1)])))
strat=pd.DataFrame(out,columns=['dataset','nonzero_charge_count','total_abs_bin','vendor1_n','vendor2_n','vendor1_rate','vendor2_rate','difference'])
strat.to_csv(OUT/'within_strata.csv',index=False)
display(strat.head(25))


results={}
for k,g in r.groupby('dataset'):
 results[k]=dict(rows=len(g),decimal_exact_share=float(g.decimal_match.mean()),cents_exact_share=float(g.cent_match.mean()),additional_path_share=float(g.exploratory_extra_exact_path.mean()),additional_path_and_decimal_match=int((g.exploratory_extra_exact_path & g.decimal_match).sum()),additional_path_total=int(g.exploratory_extra_exact_path.sum()))
report=dict(notebook='NB135-R',scientific_question='decimal consistency and vendor confounding of arithmetic path matches',sources=META,results=results,matched_strata=len(strat),
 novelty_established=False,claimed_vendor_arithmetic_fingerprint=False,
 limitations=['Source decimal representations reconstructed via Decimal(str(float)), not original billing strings','NB118-matched first 125000 rows for each Yellow and 53577 Green rows; not all source rows','Only truly absent charge fields zero-filled; columns resolved case-insensitively including Airport_fee','Within-stratum comparisons descriptive only','Exploratory arithmetic paths are NOT verified replicas of the five NB118 path implementations'],
 decision='Interpret decimal accounting and vendor association cautiously; do not revive evaluation-path compressor without substantial originality and compression evidence')
(OUT/'decision.json').write_text(json.dumps(report,indent=2))
(OUT/'README.md').write_text('NB135-R NB118-matched decimal consistency audit over NYC TLC source-row windows. Does not establish novel compression or vendor calculation order.\n')
archive='/content/SGX_NB135_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.iterdir():z.write(p,p.name)
print(json.dumps(results,indent=2))
from google.colab import files
files.download(archive)

