#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB134 — Bounded Random-Access Compression Benchmark

**Question:** How much storage, decoding work, and temporary memory are required for *bit-exact single-value access* to independently compressed floating-point pages?

**Methods:** Pcodec standalone, Zstd-3, LZMA-6; page sizes 128, 512, 2048, 8192 values. All page offsets and lengths count toward storage; every request decodes its enclosing page (no hidden cache). **This does not benchmark native indexed modes of every compressor, nor establish algorithmic novelty.**

**Sources:** NYC TLC Yellow Jan 2024 and Green Feb 2024, plus UCI wine-quality red/white alcohol columns. No synthetic input counted as a real-data success. Pure float64 column streams; not complete Parquet files. The goal is a *diagnostic Pareto frontier* prior to any new mathematical construction.

**Gate:** seek a material reproducible compression-size / random-access-work tradeoff on two independent domains. A tradeoff alone is not proof of novelty; prior art includes page-indexed codecs, ALP, Elias–Fano, ZFP, and locally decodable codes.
"""

# %pip -q install numpy pandas pyarrow zstandard pcodec


import os, json, time, lzma, zlib, zipfile, hashlib, urllib.request, platform, importlib.metadata
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, zstandard as zstd
from pcodec import standalone, ChunkConfig
OUT=Path('/content/SGX_NB134_results');OUT.mkdir(exist_ok=True)
URLS={'yellow_jan_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet','green_feb_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-02.parquet'}
MAX_ROWS=100000; PAGE_SIZES=[128,512,2048,8192]; REQUESTS=120; SEED=134

def load_taxi(name,url):
 p=Path('/content')/(name+'.parquet')
 if not p.exists():urllib.request.urlretrieve(url,p)
 chunks=[];read=0
 for batch in pq.ParquetFile(p).iter_batches(columns=['total_amount'],batch_size=32768):
  col=batch.column(0); valid=~col.is_null().to_numpy(zero_copy_only=False)
  v=np.asarray(col.to_numpy(zero_copy_only=False),dtype='<f8')[valid]
  chunks.append(v);read+=len(col)
  if read>=MAX_ROWS:break
 return np.ascontiguousarray(np.concatenate(chunks)[:MAX_ROWS],dtype='<f8')

def load_wine():
 arrays=[]
 for fn in ('winequality-red.csv','winequality-white.csv'):
  p=Path('/content')/fn
  if not p.exists():urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/'+fn,p)
  arrays.append(pd.read_csv(p,sep=';')['alcohol'].to_numpy(dtype='<f8'))
 return np.ascontiguousarray(np.concatenate(arrays),dtype='<f8')
D={k:load_taxi(k,u) for k,u in URLS.items()};D['uci_wine_alcohol']=load_wine()
print({k:len(v) for k,v in D.items()})


"""
## 1. Exact page codec and index

Each compressed page has a fixed 16-byte index entry: an 8-byte offset, 4-byte compressed length, and 4-byte element count. The stream additionally carries a fixed 32-byte header. Each codec's own framing is retained. Page boundaries are explicit, and indexes are charged to total size. Decoder reconstructs a full page to answer a single query.
"""

def encode_page(x,method):
 if method=='pcodec':return standalone.simple_compress(x,ChunkConfig())
 if method=='zstd':return zstd.ZstdCompressor(level=3).compress(x.tobytes())
 if method=='lzma':return lzma.compress(x.tobytes(),preset=6)
 raise ValueError(method)
def decode_page(blob,method,n):
 if method=='pcodec':out=np.ascontiguousarray(standalone.simple_decompress(blob),dtype='<f8')
 elif method=='zstd':out=np.frombuffer(zstd.ZstdDecompressor().decompress(blob),dtype='<f8').copy()
 elif method=='lzma':out=np.frombuffer(lzma.decompress(blob),dtype='<f8').copy()
 else:raise ValueError(method)
 if len(out)!=n:raise ValueError('wrong row count')
 return out

def pack_pages(values,method,page_size):
 payloads=[];lengths=[];counts=[]
 for start in range(0,len(values),page_size):
  page=np.ascontiguousarray(values[start:start+page_size],dtype='<f8')
  p=encode_page(page,method);payloads.append(p);lengths.append(len(p));counts.append(len(page))
 total=32+16*len(payloads)+sum(lengths)
 return payloads,counts,total

def exact(a,b):return a.shape==b.shape and np.array_equal(a.view('<u8'),b.view('<u8'))
CONTROL=np.array([0,0x8000000000000000,1,0x7ff0000000000000,0x7ff8000000000001,0x7fefffffffffffff],dtype='<u8').view('<f8')
for method in ('pcodec','zstd','lzma'):
 try:
  assert exact(CONTROL,decode_page(encode_page(CONTROL,method),method,len(CONTROL)))
  print(method,'edge case bit exact PASS')
 except Exception as e: print(method,'edge case NOT VERIFIED:',repr(e))


"""
## 2. Actual random-access timing and work

Each request fetches a uniformly sampled index, decodes the relevant compressed page and returns the original 64-bit word. Values are checked by original bytes; every request is uncached. Timing reflects the full Python decoder path on the current Colab instance and is not a hardware-independent bound. Report *page values processed* as a machine-independent work proxy and measured decode milliseconds separately.
"""

rows=[];failures=[];manifest={}
for name,values in D.items():
 manifest[name]={'rows':len(values),'input_sha256':hashlib.sha256(values.tobytes()).hexdigest(),'source':URLS.get(name,'UCI Wine Quality red+white CSV alcohol')}
 rng=np.random.default_rng(SEED);indices=rng.integers(0,len(values),size=REQUESTS)
 for page_size in PAGE_SIZES:
  for method in ('pcodec','zstd','lzma'):
   try:
    t=time.perf_counter();blobs,counts,total=pack_pages(values,method,page_size);encode_s=time.perf_counter()-t
    # Independent full reconstruction on each page.
    for k,blob in enumerate(blobs):
     got=decode_page(blob,method,counts[k]);expected=values[k*page_size:k*page_size+counts[k]]
     if not exact(got,expected):raise AssertionError('page roundtrip mismatch')
    observations=[];work=[]
    for pos in indices:
     page_id=int(pos)//page_size; local=int(pos)%page_size
     t=time.perf_counter_ns();page=decode_page(blobs[page_id],method,counts[page_id]);dt=(time.perf_counter_ns()-t)/1e6
     if page[local:local+1].view('<u8')[0]!=values[int(pos):int(pos)+1].view('<u8')[0]:raise AssertionError('query mismatch')
     observations.append(dt);work.append(counts[page_id])
    rows.append(dict(dataset=name,method=method,page_size=page_size,values=len(values),total_bytes=total,compressed_payload_bytes=sum(map(len,blobs)),index_header_bytes=total-sum(map(len,blobs)),
            avg_page_values_decoded=float(np.mean(work)),max_page_values_decoded=int(max(work)),request_ms_median=float(np.median(observations)),request_ms_p95=float(np.percentile(observations,95)),encode_seconds=encode_s,bitwise_verified=True))
   except Exception as e:
    failures.append(dict(dataset=name,method=method,page_size=page_size,error=repr(e)))
res=pd.DataFrame(rows);res.to_csv(OUT/'random_access_results.csv',index=False)
pd.DataFrame(failures,columns=['dataset','method','page_size','error']).to_csv(OUT/'failures.csv',index=False)
(OUT/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
display(res[['dataset','method','page_size','total_bytes','avg_page_values_decoded','request_ms_p95']])
print('Errors:',failures[:6])


"""
## 3. Pareto decision and export

The computed frontier uses storage bytes and average values decoded, not noisy timings. **A frontier is expected even in ordinary page-indexed compression; it does not prove novelty or a gap.** The next research step must first identify a new mathematical trade-off not already covered by published indexed, partitioned, and locally decodable compression.
"""

front=[]
for name,g in res.groupby('dataset'):
 for row in g.itertuples():
  dominant=((g.total_bytes<=row.total_bytes)&(g.avg_page_values_decoded<=row.avg_page_values_decoded)&((g.total_bytes<row.total_bytes)|(g.avg_page_values_decoded<row.avg_page_values_decoded))).any()
  if not dominant:front.append(dict(dataset=name,method=row.method,page_size=row.page_size,total_bytes=row.total_bytes,avg_page_values_decoded=row.avg_page_values_decoded))
frontier=pd.DataFrame(front);frontier.to_csv(OUT/'pareto_frontier.csv',index=False);display(frontier)
decision={'notebook':'NB134','purpose':'size_vs_local_decode_benchmark','all_measured_roundtrips_bitwise':bool(len(res)>0 and res.bitwise_verified.all()),'failures_count':len(failures),'new_algorithm_discovered':False,'novelty_established':False,'official_native_random_access_algorithms_compared':False,
 'decision':'Diagnostic baseline only; do NOT claim indexed block coding is novel or propose NB135 without a distinct mathematical contribution.'}
(OUT/'decision.json').write_text(json.dumps(decision,indent=2))
(OUT/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'pcodec':importlib.metadata.version('pcodec'),'zstandard':importlib.metadata.version('zstandard'),'page_sizes':PAGE_SIZES,'seed':SEED,'requests_per_data':REQUESTS},indent=2))
(OUT/'README.md').write_text('NB134: same-input standalone page-indexed baseline with explicit 16-byte per-page index entry and 32-byte global header. Measures total bytes, random-access time, and page decode work. It is not a novel algorithm or complete specialist codec comparison.\n')
archive='/content/SGX_NB134_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for f in OUT.iterdir():z.write(f,f.name)
print(json.dumps(decision,indent=2))
from google.colab import files
files.download(archive)

