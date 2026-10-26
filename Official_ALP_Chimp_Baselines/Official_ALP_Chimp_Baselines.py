#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB124 — Official ALP / Chimp Baseline Audit

**Purpose:** Test actual DuckDB implementations of the published **ALP** and **Chimp** floating-point codecs, not a homemade proxy. Identify whether NB123's apparent representation gaps survive specialized compression.

**Not a new codec / no novelty claim.** DuckDB uses persistent storage framing, so **whole DuckDB database sizes are not comparable** to standalone zstd streams. Report them separately; use `pragma_storage_info` to identify actual compression used. If a forced codec isn't supported, record that fact rather than substituting another codec. This notebook tests the exact IEEE-754 bits for finite non-null inputs.

**Data:** Same publicly available TLC monetary column (Yellow Jan, Green Feb), and UCI red wine quality numerical attribute (`alcohol`). Raw double words including sign of zero are verified after round trip. Input rows are fixed and dataset hashes are exported.

**Sources:** https://github.com/cwida/ALP and https://duckdb.org/docs/stable/internals/storage; https://github.com/Spatio-Temporal-Lab/elf . The latter is not benchmarked here—this is an explicit limitation.
"""

# %pip -q install duckdb pyarrow numpy pandas zstandard


import duckdb, numpy as np, pandas as pd, pyarrow.parquet as pq
import pathlib, urllib.request, hashlib, json, os, time, zipfile, zstandard as zstd
from IPython.display import display
ROOT=pathlib.Path('/content'); OUT=ROOT/'SGX_NB124_results';OUT.mkdir(exist_ok=True)
print('DuckDB version',duckdb.__version__)
SOURCES={'yellow_jan_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet','green_feb_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-02.parquet'}
N=125000

def taxi(name,url):
    p=ROOT/(name+'.parquet')
    if not p.exists():urllib.request.urlretrieve(url,p)
    data=[];got=0
    for b in pq.ParquetFile(p).iter_batches(batch_size=32768,columns=['total_amount']):
        a=b.column(0).to_numpy(zero_copy_only=False)
        data.append(np.asarray(a,dtype='<f8'));got+=len(a)
        if got>=N:break
    arr=np.concatenate(data)[:N];arr=arr[np.isfinite(arr)]
    return np.ascontiguousarray(arr,dtype='<f8')

def wine():
    p=ROOT/'winequality-red.csv'
    url='https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    if not p.exists():urllib.request.urlretrieve(url,p)
    a=pd.read_csv(p,sep=';')['alcohol'].to_numpy(dtype='<f8')
    return np.ascontiguousarray(a)
D={k:taxi(k,u) for k,u in SOURCES.items()};D['uci_wine_red_alcohol']=wine()
print({k:len(a) for k,a in D.items()})


"""
## Independent exact-float baseline
Raw IEEE-754 binary64 values are compressed using official `zstandard`. The DuckDB storage comparisons below are **within-engine comparisons**; we do not present database file bytes as pure codec payload bytes.
"""

rows=[];manifest=[]
for name,values in D.items():
    inp=values.tobytes();manifest.append(dict(dataset=name,rows=len(values),sha256=hashlib.sha256(inp).hexdigest(),raw_bytes=len(inp),source=SOURCES.get(name,'UCI wine quality red')))
    t=time.perf_counter(); enc=zstd.ZstdCompressor(level=3).compress(inp); secs=time.perf_counter()-t
    assert zstd.ZstdDecompressor().decompress(enc)==inp
    rows.append(dict(dataset=name,method='raw_zstd3',status='ok',file_bytes=len(enc),storage_codec='zstd',seconds=secs,exact=True,scope='standalone_stream'))
display(pd.DataFrame(rows))


"""
## Official DuckDB ALP and Chimp

For each dataset/codec a new persistent database is built and checkpointed. Read `pragma_storage_info` to confirm the actual storage encoding. The raw word stream is reconstructed from stored `DOUBLE` values with deterministic row order. DuckDB may change codec availability; an unsupported codec is reported as `unavailable`. Database files include common engine/page overhead and do not represent raw ALP payload size.
"""

for name,values in D.items():
    for method in ('ALP','CHIMP','UNCOMPRESSED'):
        db=ROOT/f'NB124_{name}_{method}.duckdb'
        for suffix in ('','-wal'):
            p=pathlib.Path(str(db)+suffix)
            if p.exists():p.unlink()
        con=None
        try:
            con=duckdb.connect(str(db))
            con.execute('CREATE TABLE data (i INTEGER, x DOUBLE)')
            con.execute('PRAGMA force_compression='+repr(method))
            tmp=pd.DataFrame({'i':np.arange(len(values),dtype=np.int32),'x':values})
            con.register('input_df',tmp)
            t=time.perf_counter();con.execute('INSERT INTO data SELECT i, x FROM input_df ORDER BY i');con.execute('CHECKPOINT');sec=time.perf_counter()-t
            seg=con.execute("SELECT DISTINCT compression FROM pragma_storage_info('data') WHERE column_name='x' AND segment_type='DOUBLE'").fetchall()
            used=','.join(sorted(set(str(s[0]) for s in seg))) or 'unknown'
            got=con.execute('SELECT x FROM data ORDER BY i').fetchnumpy()['x'].astype('<f8',copy=False)
            assert np.array_equal(got.view('<u8'),values.view('<u8')),f'bitwise mismatch {name} {method}'
            con.close();con=None
            size=db.stat().st_size
            rows.append(dict(dataset=name,method='duckdb_'+method.lower(),status='ok',file_bytes=size,storage_codec=used,seconds=sec,exact=True,scope='whole_duckdb_database'))
        except Exception as e:
            if con is not None:
                try:con.close()
                except Exception:pass
            rows.append(dict(dataset=name,method='duckdb_'+method.lower(),status='unavailable_or_failed: '+repr(e)[:200],file_bytes=None,storage_codec='',seconds=None,exact=False,scope='whole_duckdb_database'))
            print('Method unavailable or failed:',name,method,repr(e)[:240])
res=pd.DataFrame(rows);display(res)
res.to_csv(OUT/'baseline_results.csv',index=False)
pd.DataFrame(manifest).to_csv(OUT/'dataset_manifest.csv',index=False)


"""
## Decision & scientific caveats

No discovery can be claimed from this notebook alone: ALP and Chimp have different storage framing; standalone zstd is not directly size-comparable to database files. This is a **baseline availability and correctness audit**. Meaningful claims of algorithmic advantage require official encoder payload-bit counts or well-controlled common-format comparisons, additional domains, and prior-art differentiation.
"""

successful=res[res.status=='ok'];ducks=successful[successful.method.str.startswith('duckdb_')]
verified_alp=bool(((ducks.method=='duckdb_alp') & ducks.storage_codec.str.contains('ALP',case=False,na=False)).any())
verified_chimp=bool(((ducks.method=='duckdb_chimp') & ducks.storage_codec.str.contains('CHIMP',case=False,na=False)).any())
decision={'notebook':'NB124','purpose':'published_baseline_availability_and_exactness','verified_alp_storage':verified_alp,'verified_chimp_storage':verified_chimp,'all_successful_roundtrips_exact':bool(successful.exact.all()),'novel_algorithm_established':False,'standalone_vs_db_comparable':False,'next':'Obtain comparable official ALP / Elf bitstream outputs and profile residual gaps; no new mechanism justified yet.'}
(OUT/'decision.json').write_text(json.dumps(decision,indent=2))
(OUT/'environment.json').write_text(json.dumps({'duckdb':duckdb.__version__,'numpy':np.__version__,'rows_cap':N},indent=2))
(OUT/'README.md').write_text('Official DuckDB forced ALP/Chimp availability and exactness; database bytes NOT directly comparable to standalone zstd. No new algorithm established.\n')
print(json.dumps(decision,indent=2))
archive=ROOT/'SGX_NB124_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir():z.write(p,p.name)
from google.colab import files
files.download(str(archive))

