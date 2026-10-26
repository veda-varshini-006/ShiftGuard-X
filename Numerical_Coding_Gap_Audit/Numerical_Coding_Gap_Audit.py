#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB123 — Numerical Residual Coding-Gap Audit

**Purpose:** Find measurable, repeatable shortcomings in *simple* bit-exact numerical representations before proposing an original mechanism. This is a **diagnostic**, not a novel codec, and is not a benchmark against ALP/Elf*/Gorilla. None of the entropy figures is an achievable upper bound on compression gains.

**Decision:** Proceed to a new algorithm specification only if a coding gap survives three independent datasets, multiple representations, fair total-byte comparisons, and subsequent comparison to official specialized codecs. Otherwise reject this hypothesis.

**Prior art:** ALP (SIGMOD 2024) compresses decimal-origin and other floats adaptively; Elf (PVLDB) and Elf* (IEEE IoT Journal 2025) exploit bit-level residual structure. A context model or XOR transform alone cannot establish novelty. See the references at the end.
"""

# %pip -q install pyarrow numpy pandas zstandard


import os, json, time, hashlib, zipfile, urllib.request, math, platform
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, zlib, lzma, zstandard as zstd
OUT=Path('/content/SGX_NB123_results'); OUT.mkdir(exist_ok=True)
URLS={'yellow_2024_01':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet',
      'green_2024_02':'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-02.parquet'}
N=100000
print('Versions',platform.python_version(),np.__version__,pd.__version__)


"""
## Data

Two NYC taxi sources, plus a distinct UCI Wine Quality dataset (red + white) from UCI archives. For Parquet, ignore nulls, and preserve the original bit patterns of all *finite* `total_amount` doubles. For Wine Quality, parse decimal source values into float64; the evaluated source is the resulting binary64 stream, **not the exact original decimal text**. No synthetic data is counted as real evidence.
"""

def taxi(name,url):
    path=Path('/content')/(name+'.parquet')
    if not path.exists():
        print('Downloading',name,flush=True);urllib.request.urlretrieve(url,path)
    arrays=[]; count=0
    for bat in pq.ParquetFile(path).iter_batches(batch_size=32768,columns=['total_amount']):
        series=bat.to_pandas()
        arr=series.to_numpy(dtype='<f8')
        arr=arr[np.isfinite(arr)]
        arrays.append(arr);count+=len(arr)
        if count>=N:break
    return np.ascontiguousarray(np.concatenate(arrays)[:N],dtype='<f8')
def wines():
    base='https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/'
    arrays=[]
    for fn in ['winequality-red.csv','winequality-white.csv']:
        path=Path('/content')/fn
        if not path.exists():urllib.request.urlretrieve(base+fn,path)
        df=pd.read_csv(path,sep=';');arrays.append(df['alcohol'].to_numpy(dtype='<f8'))
    return np.ascontiguousarray(np.concatenate(arrays),dtype='<f8')
DATA={name:taxi(name,url) for name,url in URLS.items()};DATA['uci_wine_alcohol']=wines()
assert len(DATA)==3 and all(len(v)>100 for v in DATA.values())
MANIFEST={k:dict(rows=len(v),sha256_float64=hashlib.sha256(v.tobytes()).hexdigest(),dtype='<f8',source=URLS.get(k,'UCI wine-quality red and white alcohol column')) for k,v in DATA.items()}
(OUT/'dataset_manifest.json').write_text(json.dumps(MANIFEST,indent=2))
print(MANIFEST)


"""
## Three reversible representations

- Raw IEEE-754 64-bit words.
- Previous-word XOR (first residual = original first word).
- Previous-word integer difference modulo 2^64 (first residual = first word).

Every representation has a deterministic inverse. The container header is the same size for each format and each compression backend. This avoids oracle predictors and side-channel leakage.
"""

MASK=(1<<64)-1

def transform(bits,method):
    b=np.asarray(bits,dtype='<u8');q=np.empty_like(b)
    if method=='raw':q[:]=b
    elif method=='xor':
        q[0]=b[0];q[1:]=np.bitwise_xor(b[1:],b[:-1])
    elif method=='delta':
        q[0]=b[0];q[1:]=np.subtract(b[1:],b[:-1],dtype=np.uint64)
    else:raise ValueError(method)
    return q

def inverse(residual,method):
    r=np.asarray(residual,dtype='<u8')
    if method=='raw':return r.copy()
    if method=='xor':
        # Pure integer-domain inverse; no float operation.
        return np.bitwise_xor.accumulate(r)
    if method=='delta':return np.cumsum(r,dtype=np.uint64)
    raise ValueError(method)

ADVERSARIAL=np.array([0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,1,0x7fefffffffffffff],dtype='<u8')
for method in ('raw','xor','delta'):
    for orig in [ADVERSARIAL,*[v.view('<u8') for v in DATA.values()]]:
        assert np.array_equal(inverse(transform(orig,method),method),orig)
print('Bit-for-bit transformation inverses PASS on adversarial words and all data')


"""
## Actual bytes vs descriptive entropy

Bytewise empirical Shannon entropy of each transformed stream is a *diagnostic statistic only*: it ignores long-range structure and cannot establish an exploitable coding gap. Actual compressed bytes are measured with zlib(6), zstd(3), and lzma(3). The fixed 16-byte experimental header is charged to every compressed output. This compares simple representations under the same backend; **not** specialized published codecs.
"""

def hbyte(payload):
    a=np.frombuffer(payload,dtype=np.uint8);counts=np.bincount(a,minlength=256)
    probs=counts[counts>0]/len(a)
    return float(-np.sum(probs*np.log2(probs)))
def pack(payload,backend):
    if backend=='zlib':return zlib.compress(payload,6)
    if backend=='zstd':return zstd.ZstdCompressor(level=3).compress(payload)
    if backend=='lzma':return lzma.compress(payload,preset=3)
    raise ValueError(backend)
def unpack(payload,backend):
    if backend=='zlib':return zlib.decompress(payload)
    if backend=='zstd':return zstd.ZstdDecompressor().decompress(payload)
    if backend=='lzma':return lzma.decompress(payload)
    raise ValueError(backend)
ROWS=[]
for name,v in DATA.items():
    words=v.view('<u8');raw_bytes=words.nbytes
    for method in ('raw','xor','delta'):
        residual=transform(words,method);stream=residual.tobytes()
        for backend in ('zlib','zstd','lzma'):
            t=time.perf_counter();enc=pack(stream,backend);encode_s=time.perf_counter()-t
            t=time.perf_counter();dec=unpack(enc,backend);decode_s=time.perf_counter()-t
            assert dec==stream
            restored=inverse(np.frombuffer(dec,dtype='<u8'),method)
            assert np.array_equal(words,restored)
            ROWS.append(dict(dataset=name,rows=len(v),method=method,backend=backend,raw_input_bytes=raw_bytes,
                             payload_bytes=len(enc),framing_bytes=16,total_compressed_bytes=16+len(enc),
                             byte_entropy_bits_per_byte=hbyte(stream),encode_seconds=encode_s,decode_seconds=decode_s,
                             roundtrip=True))
RESULTS=pd.DataFrame(ROWS);RESULTS.to_csv(OUT/'coding_gap_measurements.csv',index=False)
display(RESULTS.pivot_table(index=['dataset','backend'],columns='method',values='total_compressed_bytes'))


# Holdout-domain comparison: select best (method, backend) on Yellow Jan only,
# then report the same frozen pair on Green February and UCI Wine, alongside the
# unattainable hindsight oracle clearly labeled as such.
train=RESULTS.query("dataset=='yellow_2024_01'").sort_values('total_compressed_bytes').iloc[0]
chosen=(train['method'],train['backend'])
rows=[]
for name in DATA:
    x=RESULTS[RESULTS.dataset==name]
    fixed=x[(x.method==chosen[0])&(x.backend==chosen[1])].iloc[0]
    raw_zstd=x[(x.method=='raw')&(x.backend=='zstd')].iloc[0]
    oracle=x.sort_values('total_compressed_bytes').iloc[0]
    rows.append(dict(dataset=name,frozen_selection=f'{chosen[0]}+{chosen[1]}',frozen_bytes=int(fixed.total_compressed_bytes),
                     raw_zstd_bytes=int(raw_zstd.total_compressed_bytes),hindsight_best_bytes=int(oracle.total_compressed_bytes),
                     hindsight_best_method=f'{oracle.method}+{oracle.backend}',frozen_vs_raw_zstd_bytes=int(fixed.total_compressed_bytes-raw_zstd.total_compressed_bytes)))
SUMMARY=pd.DataFrame(rows);SUMMARY.to_csv(OUT/'frozen_vs_hindsight.csv',index=False)
display(SUMMARY)
# Objective stopping standard: no claim of novelty based only on these measurements.
DECISION={'notebook':'NB123','stage':'representation-gap diagnostic','bit_exact_all':bool(RESULTS.roundtrip.all()),
  'novelty_established':False,'specialized_codecs_benchmarked':False,
  'selected_on':'yellow_2024_01','next_gate':'Official ALP, Elf*, and additional independent datasets plus mechanism-level prior-art comparison are REQUIRED before algorithm design',
  'recommendation':'Do not claim a new algorithm from simple transforms or descriptive entropy. Use these measurements to prioritize a controlled official-codec comparison.'}
(OUT/'decision.json').write_text(json.dumps(DECISION,indent=2))
(OUT/'README.md').write_text('NB123 is a bit-exact reversible representation diagnostic, NOT an original codec or a benchmark against ALP/Elf*. Values are original float64 for TLC and float64 parsed from UCI decimal CSV for Wine.\n')
print(json.dumps(DECISION,indent=2))
zip_path='/content/SGX_NB123_results.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for file in OUT.iterdir():z.write(file,arcname=file.name)
from google.colab import files
files.download(zip_path)


"""
## Prior art and paper-readiness boundary

- ALP, Afroozeh et al., SIGMOD 2024: https://duckdb.org/library/alp/ ; original code: https://github.com/cwida/ALP
- Elf, Li et al., PVLDB 2023: https://www.vldb.org/pvldb/vol16/p1763-li.pdf
- Elf*, IEEE IoT Journal, DOI 10.1109/JIOT.2025.3554999

**Interpretation:** An observed advantage against a generic backend is not evidence of novelty. An unexpected repeated residual structure must be paired with a new mathematical encoding rule and compared against prior work before any claim. Negative results remain part of the research record.
"""
