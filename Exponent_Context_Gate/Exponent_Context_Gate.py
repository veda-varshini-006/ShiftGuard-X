#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB129 — Reversible exponent-context partitioning gate

**Scientific status:** A clearly labeled *candidate feasibility experiment*, not proof of novelty. Byte-shuffling, exponent grouping and bit splitting have prior art (ALP, ZipNN, SLEEK, etc.). We test whether a deterministic exponent-conditioned reordering can create a material coding gap relative to strong existing compressors, **with all permutation/metadata costs charged**. If not, reject this mechanism. A positive result calls for a new mathematical contribution and literature audit; it does not establish novelty.

**Data:** Yellow Jan/Feb 2024 and Green Feb 2024 official NYC TLC `total_amount` finite binary64 data, first 125,000 source rows. Exact bit-pattern reconstruction. At present no official ALP/Chimp same-input standalone measurements are available: no superiority claim about them.

**Hypothesis:** On some nonstationary numerical streams, blockwise exponent-conditioned *stable partitioning* can improve backend coding enough to pay for the information required to reconstruct original order. The baseline includes raw, conventional byte shuffle, and specialist Pcodec. If we fail to beat **both best plain general codec and Pcodec** by at least 3% on both held-out February datasets, reject.
"""

# %pip -q install numpy pandas pyarrow zstandard pcodec


import io,os,json,hashlib,zipfile,urllib.request,time,lzma,platform
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq,zstandard as zstd
from pcodec import standalone,ChunkConfig
OUT=Path('/content/SGX_NB129_results');OUT.mkdir(exist_ok=True)
URLS={'yellow_jan_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet','yellow_feb_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet','green_feb_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-02.parquet'}
N=125000; BLOCK=8192

def load(k,u):
 p=Path('/content')/(k+'.parquet')
 if not p.exists():print('Download:',k,flush=True);urllib.request.urlretrieve(u,p)
 pf=pq.ParquetFile(p); parts=[]; seen=0
 for batch in pf.iter_batches(batch_size=32768,columns=['total_amount']):
  s=batch.column(0).to_pandas().to_numpy(dtype='<f8');parts.append(s);seen+=len(s)
  if seen>=N:break
 x=np.concatenate(parts)[:N];x=np.ascontiguousarray(x[np.isfinite(x)],dtype='<f8')
 return x
D={k:load(k,u) for k,u in URLS.items()}
print({k:len(v) for k,v in D.items()})


"""
## Reversible encoding controls

`shuffle` is byte-transposition within a block. `exponent_partition` splits records into 8 coarse exponent classes, writes a **compressed per-row class stream** and stores 8 subgroup byte-shuffled payloads; stable grouping is invertible using the class stream. The class stream is **mandatory** (not free side information). The codec header includes version, per-block counts, and lengths. No value is rounded. Each representation is independently decoded and compared using exact original bytes.
"""

def raw_bytes(x):return np.ascontiguousarray(x,dtype='<f8').tobytes()
def enc_backend(payload,kind):
 if kind=='zstd':return zstd.ZstdCompressor(level=3).compress(payload)
 if kind=='lzma':return lzma.compress(payload,preset=6)
 raise ValueError(kind)
def dec_backend(payload,kind):
 if kind=='zstd':return zstd.ZstdDecompressor().decompress(payload)
 if kind=='lzma':return lzma.decompress(payload)
 raise ValueError(kind)
def shuffle(x):
 a=np.frombuffer(x,dtype=np.uint8).reshape(-1,8)
 return a.T.copy().tobytes()
def unshuffle(x):
 a=np.frombuffer(x,dtype=np.uint8).reshape(8,-1)
 return a.T.copy().tobytes()
def classes(x):
 bits=np.frombuffer(x,dtype='<u8')
 exp=((bits>>52)&2047).astype(np.int32)
 return (exp//256).astype(np.uint8)
def pack_parts(parts):
 # Big-endian 4-byte lengths are included in the payload for each part.
 return b''.join(len(p).to_bytes(4,'big')+p for p in parts)
def unpack_parts(data,k):
 pos=0;out=[]
 for _ in range(k):
  if pos+4>len(data):raise ValueError('short header')
  sz=int.from_bytes(data[pos:pos+4],'big');pos+=4
  if pos+sz>len(data):raise ValueError('short part')
  out.append(data[pos:pos+sz]);pos+=sz
 if pos!=len(data):raise ValueError('extra bytes')
 return out

def encode_rep(x,mode,backend):
 b=raw_bytes(x); n=len(x)
 if mode=='raw':parts=[enc_backend(b,backend)]
 elif mode=='shuffle':parts=[enc_backend(shuffle(b),backend)]
 elif mode=='exponent_partition':
  c=classes(b);groups=[]
  for k in range(8):groups.append(enc_backend(shuffle(np.ascontiguousarray(x[c==k]).tobytes()),backend))
  parts=[enc_backend(c.tobytes(),backend)]+groups
 else:raise ValueError(mode)
 out=b'SGX129'+bytes([{'raw':0,'shuffle':1,'exponent_partition':2}[mode],0 if backend=='zstd' else 1])+n.to_bytes(4,'big')+pack_parts(parts)
 return out

def decode_rep(blob):
 assert blob[:6]==b'SGX129'
 mode=blob[6];backend='zstd' if blob[7]==0 else 'lzma';n=int.from_bytes(blob[8:12],'big');parts=unpack_parts(blob[12:],9 if mode==2 else 1)
 if mode==0:ans=dec_backend(parts[0],backend)
 elif mode==1:ans=unshuffle(dec_backend(parts[0],backend))
 else:
  c=np.frombuffer(dec_backend(parts[0],backend),dtype=np.uint8)
  assert len(c)==n and np.all(c<8)
  groups=[np.frombuffer(unshuffle(dec_backend(p,backend)),dtype='<f8').copy() for p in parts[1:]]
  assert all(len(groups[k])==int((c==k).sum()) for k in range(8))
  # Reconstruction of order uses only explicit class stream.
  ansarr=np.empty(n,dtype='<f8')
  for k in range(8):ansarr[c==k]=groups[k]
  ans=ansarr.tobytes()
 if len(ans)!=n*8:raise ValueError('length mismatch')
 return ans

# Signed zero, infinities, noncanonical NaN payload, subnormal; no assumption about numeric equality.
special=np.array([0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,1,0x7fefffffffffffff],dtype='<u8').view('<f8')
for m in ('raw','shuffle','exponent_partition'):
 for b in ('zstd','lzma'):
  packed=encode_rep(special,m,b);assert decode_rep(packed)==special.tobytes(),(m,b)
print('All IEEE bit-pattern controls PASS')


records=[]
for dataset,x in D.items():
 for bi,start in enumerate(range(0,len(x),BLOCK)):
  block=np.ascontiguousarray(x[start:start+BLOCK],dtype='<f8');orig=raw_bytes(block)
  for mode in ('raw','shuffle','exponent_partition'):
   for backend in ('zstd','lzma'):
    t=time.perf_counter();frame=encode_rep(block,mode,backend);sec=time.perf_counter()-t
    t=time.perf_counter();rest=decode_rep(frame);dec=time.perf_counter()-t
    assert rest==orig,(dataset,bi,mode,backend)
    records.append(dict(dataset=dataset,block=bi,mode=mode,backend=backend,compressed_bytes=len(frame),input_bytes=len(orig),enc_seconds=sec,dec_seconds=dec,bitexact=True))
  # Official Pcodec standalone compresses the same float64 block; its framing already counts.
  t=time.perf_counter();payload=standalone.simple_compress(block,ChunkConfig());sec=time.perf_counter()-t
  t=time.perf_counter();recovered=standalone.simple_decompress(payload);dec=time.perf_counter()-t
  assert np.ascontiguousarray(recovered,dtype='<f8').tobytes()==orig
  records.append(dict(dataset=dataset,block=bi,mode='pcodec',backend='official_standalone',compressed_bytes=len(payload),input_bytes=len(orig),enc_seconds=sec,dec_seconds=dec,bitexact=True))
results=pd.DataFrame(records);results.to_csv(OUT/'results_per_block.csv',index=False)
summary=results.groupby(['dataset','mode','backend'],as_index=False)[['compressed_bytes','input_bytes','enc_seconds','dec_seconds']].sum()
summary.to_csv(OUT/'summary.csv',index=False);display(summary)


# Freeze choice of simple backend/mode using January before looking at held-out February.
training=summary[(summary.dataset=='yellow_jan_2024') & (summary['mode'].isin(['raw','shuffle']))]
choice=training.loc[training.compressed_bytes.idxmin(),['mode','backend']].to_dict()
held=['yellow_feb_2024','green_feb_2024'];scores=[]
for k in held:
 a=summary[summary.dataset==k];p=int(a.loc[a['mode']=='pcodec','compressed_bytes'].iloc[0])
 raw=int(a.loc[(a['mode']==choice['mode'])&(a['backend']==choice['backend']),'compressed_bytes'].iloc[0])
 best=min(p,raw)
 proposed=int(a.loc[(a['mode']=='exponent_partition')&(a['backend']==choice['backend']),'compressed_bytes'].iloc[0])
 scores.append(dict(dataset=k,frozen_baseline_choice=choice,baseline_bytes=best,partition_bytes=proposed,relative_gain=(best-proposed)/best))
gate=pd.DataFrame(scores);display(gate)
pass_gate=bool((gate.relative_gain>=0.03).all())
report={'notebook':'NB129','all_roundtrips_pass':bool(results.bitexact.all()),'gate_requires_three_percent_both_heldout':pass_gate,'novelty_established':False,'near_prior_art':['ALP 2024','ZipNN byte grouping','SLEEK 2026','Elf* 2025','Falcon 2025'],'recommendation':'Investigate distinct mathematics/prior-art only if gate passes; otherwise reject exponent partition as a core invention.'}
(OUT/'decision.json').write_text(json.dumps(report,indent=2));(OUT/'dataset_sources.json').write_text(json.dumps(URLS,indent=2))
(OUT/'README.md').write_text('NB129 diagnostic: exponent-partition candidate compared with raw and byte-shuffle zstd/lzma and official Pcodec, at independent block granularity. Exact stream bytes and decoding verified. Not novelty proof.\n')
archive='/content/SGX_NB129_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.iterdir():z.write(p,p.name)
print(report,'\nArchive',archive)
from google.colab import files
files.download(archive)

