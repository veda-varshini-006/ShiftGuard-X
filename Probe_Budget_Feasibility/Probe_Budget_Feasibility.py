#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB122 — Paid-Probing Feasibility Gate

**Objective:** Identify whether codec-probing decisions on heterogeneous, drifting *lossless byte streams* have enough economic significance for a new algorithm. **This notebook does not claim novelty or implement a new published-quality algorithm.**

**Prior-art risk:** GRASP (2026) already samples for adaptive compressor selection; Waved Zero Streaming (2026) uses nonstationary adaptive selection; Reyzin, Saha & Wu (2026) study costly expert observations. The next innovation must exceed these mechanisms.

**Protocol:** Test on public NYC TLC data plus explicitly labelled synthetic-drift controls. Use a frozen warmup, actual per-block compressed bytes, measured encode/probe time, counts for headers, and a *strict online replay* (the policy only sees an alternative codec's performance when it pays to probe it). Compare fixed, periodic probes, drift-triggered probes, and omniscient oracle (upper bound only).

**Decision rule:** Proceed to algorithm derivation only if (i) codec winners change in real data, (ii) the oracle-fixed gap is material (predeclared ≥2% of fixed bytes), and (iii) selective probing improves total runtime-adjusted cost against a matched-budget periodic policy on real data. Otherwise pivot, not another notebook of tuning.
"""

# %pip -q install pandas numpy pyarrow zstandard


import io, os, json, math, time, zlib, bz2, lzma, hashlib, zipfile, platform, urllib.request
from pathlib import Path
import numpy as np, pandas as pd, zstandard as zstd, pyarrow.parquet as pq
OUT=Path('/content/SGX_NB122_results');OUT.mkdir(exist_ok=True)
URLS={'yellow_jan_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet',
      'yellow_feb_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet',
      'green_feb_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-02.parquet'}
MAX_ROWS=100000; BLOCK_BYTES=32768; MAX_BLOCKS=100
print('Python',platform.python_version(),'NumPy',np.__version__)


# Encode *exact* target bit patterns in source row order; this is not whole Parquet compression.
def fetch(name,url):
    p=Path('/content')/(name+'.parquet')
    if not p.exists():
        print('Downloading',name,flush=True);urllib.request.urlretrieve(url,p)
    pf=pq.ParquetFile(p); frames=[]; got=0
    for b in pf.iter_batches(batch_size=32768,columns=['total_amount']):
        arr=b.to_pandas().to_numpy(dtype='<f8');frames.append(arr);got+=len(arr)
        if got>=MAX_ROWS:break
    arr=np.concatenate(frames)[:MAX_ROWS]
    # Preserve NaN bit patterns where available; reject nulls (which have no original IEEE-754 word).
    # Pandas null-to-NaN conversion could lose payload identity; only finite values are used in this diagnostic.
    arr=arr[np.isfinite(arr)]
    data=np.ascontiguousarray(arr,dtype='<f8').tobytes()
    return data,hashlib.sha256(data).hexdigest(),len(arr)
REAL={name:fetch(name,url) for name,url in URLS.items()}
print({k:(len(v[0]),v[2]) for k,v in REAL.items()})


# Real dataset controls plus one *labeled synthetic* regime-shift stress test.
rng=np.random.default_rng(122)
synthetic=b''.join([(b'ABCD'*8192) if j%3==0 else (b'\x00'*32768) if j%3==1 else rng.bytes(32768) for j in range(45)])
STREAMS={k:v[0] for k,v in REAL.items()}
STREAMS['synthetic_drift_CONTROL']=synthetic
CODECS=('zlib','zstd','bz2','lzma')
def encode(x,mode):
    if mode=='zlib':return zlib.compress(x,6)
    if mode=='zstd':return zstd.ZstdCompressor(level=3).compress(x)
    if mode=='bz2':return bz2.compress(x,compresslevel=5)
    if mode=='lzma':return lzma.compress(x,preset=3)
    raise ValueError(mode)
def decode(y,mode):
    if mode=='zlib':return zlib.decompress(y)
    if mode=='zstd':return zstd.ZstdDecompressor().decompress(y)
    if mode=='bz2':return bz2.decompress(y)
    if mode=='lzma':return lzma.decompress(y)
HEADER_BYTES=9  # codec ID (1), original length (4), compressed length (4), per independently decoded block
blocks={}
for name,data in STREAMS.items():
    blocks[name]=[data[i:i+BLOCK_BYTES] for i in range(0,len(data),BLOCK_BYTES)][:MAX_BLOCKS]
print({k:len(v) for k,v in blocks.items()})


"""
## Actual codec costs and bit-exact checks

Forensic table exhaustively evaluates each compressor once to construct the oracle and simulate online choices. The **online replay must not read hidden costs until a probe is purchased**. Its reported extra probing CPU is accumulated separately; already-compressed candidate payload can be reused for the chosen codec. Header cost is added to every deployed block.
"""

records=[]
for dataset,arr in blocks.items():
    for bi,block in enumerate(arr):
        for codec in CODECS:
            t=time.perf_counter_ns();payload=encode(block,codec);sec=(time.perf_counter_ns()-t)/1e9
            assert decode(payload,codec)==block
            records.append(dict(dataset=dataset,block=bi,codec=codec,bytes=len(payload)+HEADER_BYTES,seconds=max(sec,1e-9),
                                sha256=hashlib.sha256(block).hexdigest()))
cost=pd.DataFrame(records)
cost.to_csv(OUT/'full_cost_trace.csv',index=False)
display(cost.groupby(['dataset','codec'])[['bytes','seconds']].sum().round(5))


"""
## Causal online policy replay

Both policies start with one codec learned from the first 10 blocks of **January yellow only**, using compressed size plus a runtime penalty. Test on the two February datasets (held out). Periodic policy buys extra probes at preassigned positions. Drift policy spends the *same* number of extra probes on blocks with the largest changes in cheap byte-histogram features (calculated from current input bytes without running another compressor). Scheduling thresholds are selected on January alone. This is a **diagnostic comparison**, not a statistically validated learned algorithm.
"""

LAMBDA=250000.0 # bytes-equivalent per second; fixed before evaluating February
warm=cost[cost.dataset=='yellow_jan_2024'].query('block < 10')
means=warm.groupby('codec').apply(lambda z:(z['bytes']+LAMBDA*z['seconds']).mean(),include_groups=False)
START=means.idxmin()
print('Frozen start codec:',START,'objective λ bytes/sec:',LAMBDA)
def features(arr):
    prev=None; scores=[]
    for b in arr:
        hist=np.bincount(np.frombuffer(b,dtype=np.uint8),minlength=256).astype(float)
        hist/=max(1,len(b));score=0 if prev is None else np.abs(hist-prev).sum()
        scores.append(score);prev=hist
    return np.asarray(scores)
def replay(name,policy,budget_frac):
    arr=blocks[name]; n=len(arr)
    probe_count=int(np.floor(n*budget_frac))
    candidates=[i for i in range(1,n)]
    if policy=='periodic':
        positions=set(np.linspace(1,n-1,probe_count,dtype=int).tolist()) if probe_count else set()
    elif policy=='drift':
        s=features(arr);positions=set(sorted(candidates,key=lambda i:(-s[i],i))[:probe_count])
    else:positions=set()
    lookup={(int(r.block),r.codec):(float(r.bytes),float(r.seconds)) for r in cost[cost.dataset==name].itertuples()}
    current=START; total_bytes=0; chosen_cpu=0.; extra_probe_cpu=0.; switches=0
    for i in range(n):
        observed={current:lookup[i,current]}
        if i in positions:
            # Pay to try the *other* codecs; their observations become available only now.
            for c in CODECS:
                if c!=current:
                    observed[c]=lookup[i,c]
                    extra_probe_cpu+=lookup[i,c][1]
            winner=min(observed,key=lambda c: observed[c][0]+LAMBDA*observed[c][1])
            switches+=winner!=current;current=winner
        b,t=lookup[i,current];total_bytes+=b;chosen_cpu+=t
    return dict(dataset=name,policy=policy,budget_fraction=budget_frac,blocks=n,bytes=total_bytes,
                encode_seconds=chosen_cpu,extra_probe_seconds=extra_probe_cpu,total_seconds=chosen_cpu+extra_probe_cpu,
                total_cost=total_bytes+LAMBDA*(chosen_cpu+extra_probe_cpu),probe_events=len(positions),switches=switches)
out=[]
for name in blocks:
    for f in (0,0.05,0.1,0.25):
        for pol in ('periodic','drift'):out.append(replay(name,pol,f))
policies=pd.DataFrame(out);policies.to_csv(OUT/'policy_replay.csv',index=False)
display(policies[policies.dataset=='yellow_feb_2024'].round(4))


summary=[]
for name in blocks:
    sub=cost[cost.dataset==name]
    pivot=sub.pivot(index='block',columns='codec',values='bytes')
    fixed=min(pivot[c].sum() for c in CODECS)
    oracle=pivot.min(axis=1).sum()
    winners=pivot.idxmin(axis=1)
    summary.append(dict(dataset=name,blocks=len(pivot),best_fixed_bytes=int(fixed),
                        full_information_oracle_bytes=int(oracle),oracle_gap_fraction=float((fixed-oracle)/fixed),
                        winner_changes=int((winners!=winners.shift()).sum()-1),
                        is_real=not name.startswith('synthetic')))
gates=pd.DataFrame(summary);gates.to_csv(OUT/'oracle_gap.csv',index=False)
display(gates)
held=policies[(policies.dataset.isin(['yellow_feb_2024','green_feb_2024']))&(policies.budget_fraction>0)]
pv=held.pivot(index=['dataset','budget_fraction'],columns='policy',values='total_cost')
genuine=bool((gates.loc[gates.dataset.isin(['yellow_feb_2024','green_feb_2024']),'oracle_gap_fraction']>=.02).all())
gain=bool((pv['drift']<pv['periodic']).any())
decision={'notebook':'NB122','kind':'feasibility_diagnostic_not_new_algorithm',
          'real_holdout_oracle_gap_at_least_2pct_both':genuine,
          'drift_replay_outperforms_periodic_at_some_budget':gain,
          'gate_pass':genuine and gain,'novelty_established':False,
          'limitations':['Fixed lambda is an illustrative predeclared cost tradeoff','Only NYC taxi constitutes real data here',
                         'Oracle is unattainable and ignores probing overhead','Timing can vary across Colab hardware',
                         'GRASP and costly-observation learning constitute close prior art'],
          'recommendation':'If pass: derive and prior-art-audit a causal compression-specific probing rule; otherwise reject this hypothesis without further tuning.'}
(OUT/'decision.json').write_text(json.dumps(decision,indent=2))
(OUT/'dataset_manifest.json').write_text(json.dumps({k:{'source_url':URLS.get(k),'sha256_stream':REAL[k][1] if k in REAL else hashlib.sha256(synthetic).hexdigest(),'blocks_tested':len(blocks[k]),'synthetic':k.startswith('synthetic')} for k in blocks},indent=2))
(OUT/'README.md').write_text('NB122 paid-probing diagnostic. These costs derive from actual compressed block bytes; online policies use a precomputed forensic trace, never future outcomes to choose actions. Not a novel algorithm.\n')
print(json.dumps(decision,indent=2))
archive='/content/SGX_NB122_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir():z.write(p,p.name)
from google.colab import files
files.download(archive)

