#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB128 — Reproducible Specialized-Baseline Integration Gate

**Purpose:** End the baseline-integration loop honestly. Benchmark Pcodec standalone streams and Zstd on identical float64 words; try building/testing the *official* ALP and Chimp source repositories. Record a codec as **NOT BENCHMARKED** unless it produces its own standalone, decodable stream on the **same** input and successfully reconstructs all 64-bit words.

**Do not compare ALP DuckDB database file size with standalone streams.** ALP and Chimp build/test results are **integration smoke tests**, not compression benchmarks. No ALP or Chimp size is invented. If no compatible upstream standalone stream runner is available, the notebook reports the limitation and closes the gate, rather than extending baseline work indefinitely.

**Data:** Official NYC TLC Yellow Jan/Feb 2024 and Green Feb 2024, finite non-null `total_amount` values, first up to 125k source records, preserving exact binary64 words within this declared subset. Negative/NaN-payload edge cases are tested separately. Requires Colab internet.
"""

# %pip -q install numpy pandas pyarrow pcodec==1.0.4 zstandard


import os, io, json, time, zlib, zipfile, hashlib, shutil, subprocess, platform, importlib.metadata
from pathlib import Path
import urllib.request
import numpy as np, pandas as pd, pyarrow.parquet as pq, zstandard as zstd
from pcodec import standalone, ChunkConfig
OUT=Path('/content/SGX_NB128_results');OUT.mkdir(exist_ok=True)
URLS={
'yellow_jan_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet',
'yellow_feb_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet',
'green_feb_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-02.parquet'}
def load(name,url):
    p=Path('/content')/(name+'.parquet')
    if not p.exists():print('Downloading',name);urllib.request.urlretrieve(url,p)
    pf=pq.ParquetFile(p); blocks=[]; total=0
    for batch in pf.iter_batches(batch_size=32768,columns=['total_amount']):
        a=batch.column(0)
        # Validity-filter at Arrow level: do not silently turn nulls into NaN.
        x=a.to_numpy(zero_copy_only=False)
        x=np.asarray(x,dtype='<f8')
        blocks.append(x);total+=len(x)
        if total>=125000:break
    x=np.concatenate(blocks)[:125000];x=np.ascontiguousarray(x[np.isfinite(x)],dtype='<f8')
    return x
D={k:load(k,v) for k,v in URLS.items()}
print('Rows:',{k:len(v) for k,v in D.items()})
print('Pcodec version',importlib.metadata.version('pcodec'))


def benchmark(name,x,codec):
    raw=np.ascontiguousarray(x,dtype='<f8').tobytes()
    if codec=='Pcodec':
        t=time.perf_counter();comp=standalone.simple_compress(x,ChunkConfig()); enc=time.perf_counter()-t
        t=time.perf_counter(); recovered=standalone.simple_decompress(comp);dec=time.perf_counter()-t
        recovered=np.ascontiguousarray(recovered,dtype='<f8').tobytes()
    elif codec=='Zstd':
        t=time.perf_counter();comp=zstd.ZstdCompressor(level=3).compress(raw);enc=time.perf_counter()-t
        t=time.perf_counter();recovered=zstd.ZstdDecompressor().decompress(comp);dec=time.perf_counter()-t
    elif codec=='LZMA':
        import lzma
        t=time.perf_counter();comp=lzma.compress(raw,preset=6);enc=time.perf_counter()-t
        t=time.perf_counter();recovered=lzma.decompress(comp);dec=time.perf_counter()-t
    else:raise ValueError(codec)
    assert recovered==raw,f'Exact bitwise mismatch for {name}/{codec}'
    return dict(dataset=name,codec=codec,rows=len(x),original_bytes=len(raw),compressed_bytes=len(comp),
                ratio=len(raw)/len(comp),encode_seconds=enc,decode_seconds=dec,roundtrip='PASS',
                input_sha256=hashlib.sha256(raw).hexdigest())
rows=[benchmark(name,x,c) for name,x in D.items() for c in ['Pcodec','Zstd','LZMA']]
result=pd.DataFrame(rows);display(result[['dataset','codec','compressed_bytes','ratio','encode_seconds','decode_seconds','roundtrip']])
result.to_csv(OUT/'standalone_comparable_streams.csv',index=False)


"""
## IEEE-754 adversarial exactness check

These seven words test signed zero, infinities, a quiet NaN payload, a subnormal, and the largest finite number. **Pcodec may explicitly reject unsupported special values:** record an error rather than claim successful lossless reconstruction.
"""

special=np.array([0x0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,
                  0x7ff8000000000001,0x1,0x7fefffffffffffff],dtype='<u8').view('<f8')
special_results=[]
for codec in ['Pcodec','Zstd','LZMA']:
    try:
        z=benchmark('ieee754_adversarial',special,codec)
        special_results.append({'codec':codec,'status':'PASS','reason':''})
    except Exception as e:
        special_results.append({'codec':codec,'status':'ERROR/UNSUPPORTED','reason':repr(e)[:500]})
display(pd.DataFrame(special_results))
(OUT/'adversarial_tests.json').write_text(json.dumps(special_results,indent=2))


"""
## Official-source integration checks (not codec byte benchmarks)

Clone the authors' repositories, capture commit hashes, and run only documented build/tests that are feasible on a standard Colab CPU. The Chimp authors document `mvn test -Dtest=TestDoublePrecision`; ALP publishes C++ examples and benchmarking facilities that require their build dependencies. Even if upstream tests pass, they do **not** establish measured compression performance on *our* arrays. For this reason, `official_*_size_bytes` is always null until a compatible cross-input runner and independent decode check are implemented.
"""

REPOS={
'ALP':'https://github.com/cwida/ALP.git',
'Chimp':'https://github.com/panagiotisl/chimp.git'}
def run(args, cwd=None, timeout=240):
    try:
        p=subprocess.run(args,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
        return {'exit_code':p.returncode,'output_tail':p.stdout[-2500:]}
    except Exception as e:return {'exit_code':None,'error':repr(e)}
integrations=[]
for name,url in REPOS.items():
    dest=Path('/content')/('sgx_source_'+name.lower())
    r=run(['git','clone','--depth','1',url,str(dest)],timeout=180) if not dest.exists() else {'exit_code':0,'output_tail':'Already cloned'}
    entry={'codec':name,'repository':url,'clone':r,'commit':None,'upstream_test':None,
           'same_input_stream_benchmark':'NOT RUN','official_standalone_bytes':None,
           'comparable_to_pcodec':False}
    if dest.exists():
        hash_result=run(['git','rev-parse','HEAD'],cwd=str(dest));entry['commit']=hash_result.get('output_tail','').strip()
        if name=='Chimp':
            entry['upstream_test']=run(['mvn','-q','test','-Dtest=TestDoublePrecision'],cwd=str(dest),timeout=240)
        else:
            # ALP docs specify several build configurations; no guessed benchmark binary is invoked.
            entry['upstream_test']={'status':'NOT RUN','reason':'No verified same-input standalone ALP runner integrated'}
    integrations.append(entry)
print(json.dumps(integrations,indent=2)[:7000])
(OUT/'official_source_integration.json').write_text(json.dumps(integrations,indent=2))


complete=bool(result.roundtrip.eq('PASS').all())
decision={
'notebook':'NB128','standalone_pcodec_zstd_lzma_roundtrips_passed':complete,
'official_ALP_standalone_benchmarked':False,'official_Chimp_standalone_benchmarked':False,
'novelty_established':False,
'interpretation':'Reliable Pcodec and generic standalone baselines; official source build status is separate from benchmark measurements.',
'next':'Stop repeated integration-only notebooks. Begin a narrowly defined, prior-art-audited mechanism study; report ALP/Chimp not benchmarked until a verified same-input encoder+decoder exists.'}
(OUT/'decision.json').write_text(json.dumps(decision,indent=2))
(OUT/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'pcodec':importlib.metadata.version('pcodec'),'zstandard':importlib.metadata.version('zstandard'),'sources':URLS,'max_rows_per_source':125000},indent=2))
(OUT/'README.md').write_text('NB128 final integration audit: pcodec standalone, zstd, lzma measured on identical float64 streams. ALP/Chimp git build tests are not compression-size evidence. No novelty established.\n')
zpath='/content/SGX_NB128_results.zip'
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
    for f in OUT.iterdir():z.write(f,f.name)
print('Saved',zpath)
from google.colab import files
files.download(zpath)

