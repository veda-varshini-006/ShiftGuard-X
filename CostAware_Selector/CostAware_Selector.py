#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# NB04B — Matched-Budget Residual and Codec Evaluation

**Purpose:** Repair the mismatch revealed by NB00–NB04. All methods use the same byte-exact encoding container, block partition, codecs and transforms. The offline oracle is evaluation-only. The synthetic datasets are smoke tests, not evidence of broad generalization. To evaluate real data, upload files to `sgx_public_data/` before running.
"""

# Optional production codecs; safe to rerun.
# %pip -q install zstandard lz4 brotli



import os,sys,time,random,hashlib,json,csv,struct,platform,statistics,math,zipfile
from pathlib import Path
from collections import Counter
import zlib,bz2,lzma
try:
 import zstandard as zstd
except ImportError: zstd=None
try:
 import lz4.frame as lz4f
except ImportError: lz4f=None
try:
 import brotli
except ImportError: brotli=None
OUT=Path('sgx_next_outputs'); OUT.mkdir(exist_ok=True)
BLOCK=16384
CODECS=['raw','zlib6','bz2','lzma1','zstd1','zstd3','lz4','brotli3']
AVAILABLE=[c for c in CODECS if (not c.startswith('zstd') or zstd is not None) and (c!='lz4' or lz4f is not None) and (c!='brotli3' or brotli is not None)]
# Fixed, byte-exact forward/inverse transformations. Transform states reset at every block.
TRANSFORMS=['identity','delta1','xor1']
def transform(b,mode):
 if mode=='identity':return b
 if not b:return b
 if mode=='delta1':return bytes([b[0]])+bytes((b[i]-b[i-1])&255 for i in range(1,len(b)))
 if mode=='xor1':return bytes([b[0]])+bytes(b[i]^b[i-1] for i in range(1,len(b)))
 raise ValueError(mode)
def inverse(b,mode):
 if mode=='identity':return b
 if not b:return b
 result=bytearray([b[0]])
 if mode=='delta1':
  for c in b[1:]:result.append((result[-1]+c)&255)
 elif mode=='xor1':
  for c in b[1:]:result.append(result[-1]^c)
 else:raise ValueError(mode)
 return bytes(result)
def compress(b,codec):
 if codec=='raw':return b
 if codec=='zlib6':return zlib.compress(b,6)
 if codec=='bz2':return bz2.compress(b,compresslevel=5)
 if codec=='lzma1':return lzma.compress(b,preset=1)
 if codec=='zstd1':return zstd.ZstdCompressor(level=1).compress(b)
 if codec=='zstd3':return zstd.ZstdCompressor(level=3).compress(b)
 if codec=='lz4':return lz4f.compress(b,compression_level=0)
 if codec=='brotli3':return brotli.compress(b,quality=3)
 raise ValueError(codec)
def decompress(b,codec):
 if codec=='raw':return b
 if codec=='zlib6':return zlib.decompress(b)
 if codec=='bz2':return bz2.decompress(b)
 if codec=='lzma1':return lzma.decompress(b)
 if codec in ('zstd1','zstd3'):return zstd.ZstdDecompressor().decompress(b)
 if codec=='lz4':return lz4f.decompress(b)
 if codec=='brotli3':return brotli.decompress(b)
 raise ValueError(codec)
# Header: magic(4), version(1), transform id(1), codec id(1), original length(4), payload len(4), CRC32(4).
HEADER=struct.Struct('<4sBBBIII')
def encode(b,action):
 tr,co=action
 payload=compress(transform(b,tr),co)
 hdr=HEADER.pack(b'SGX2',1,TRANSFORMS.index(tr),CODECS.index(co),len(b),len(payload),zlib.crc32(b))
 return hdr+payload
def decode(blob):
 if len(blob)<HEADER.size:raise ValueError('Truncated header')
 magic,ver,tid,cid,n,k,crc=HEADER.unpack_from(blob)
 if magic!=b'SGX2' or ver!=1 or tid>=len(TRANSFORMS) or cid>=len(CODECS) or len(blob)!=HEADER.size+k:raise ValueError('Invalid container')
 result=inverse(decompress(blob[HEADER.size:],CODECS[cid]),TRANSFORMS[tid])
 if len(result)!=n or zlib.crc32(result)!=crc:raise ValueError('CRC/length mismatch')
 return result
ACTIONS=[(t,c) for t in TRANSFORMS for c in AVAILABLE]
FAST_ACTIONS=[('identity',c) for c in AVAILABLE if c in ('raw','zlib6','zstd1','lz4')]
def stats(b):
 if not b:return [0.,0.,0.]
 sample=b[:4096];counts=Counter(sample);n=len(sample)
 entropy=-sum((v/n)*math.log2(v/n) for v in counts.values())
 changes=sum(sample[i]!=sample[i-1] for i in range(1,n))/max(1,n-1)
 return [entropy,len(counts)/256.,changes]
def change(a,b):
 return math.sqrt(sum((x-y)**2 for x,y in zip(a,b))) if a and b else 9.0
def synthetic(seed=2026):
 rng=random.Random(seed)
 make=lambda n:bytes(rng.randrange(256) for _ in range(n))
 walk=bytearray();x=127
 for i in range(262144):
  x=(x+rng.choice([-1,0,0,1]))&255;walk.append(x)
 log=b''.join((f'INFO request={i%500:04d} GET /items user={i%97:03d} status=ok\n').encode() for i in range(5500))
 jsonl=b''.join((f'{{"id":{i},"region":"APAC","status":"ok","v":{i%17}}}\n').encode() for i in range(5500))
 random_bytes=make(262144)
 regimes=bytes(walk[:65536])+log[:65536]+random_bytes[:65536]+jsonl[:65536]
 recurring=log[:65536]+random_bytes[:65536]+log[65536:131072]+bytes(walk[:65536])
 return {'sensor_walk':bytes(walk),'logs':log,'jsonl':jsonl,'random':random_bytes,'regimes':regimes,'recurring':recurring}
def data_sources():
 d=synthetic()
 data_dir=Path('sgx_public_data')
 if data_dir.exists():
  for p in sorted(data_dir.iterdir()):
   if p.is_file() and p.stat().st_size>10000 and p.stat().st_size<=25000000:
    d['external_'+p.name]=p.read_bytes()
 return d
def chunks(b,n=BLOCK):return [b[i:i+n] for i in range(0,len(b),n)]
def run_policy(blocks,policy,budget=2,seed=123,threshold=.7,include_oracle=False):
 rng=random.Random(seed);prev=None;prev_feature=None;records=[];all_blobs=[]
 incumbent=('identity','zstd1' if 'zstd1' in AVAILABLE else 'zlib6')
 start=time.perf_counter();probes=0
 for i,b in enumerate(blocks):
  t0=time.perf_counter();features=stats(b);shift=change(prev_feature,features)
  # Caches hold only tried actions for current block. No peeking at unchosen actions.
  cache={};probe_cost=0.0
  def eval_action(a):
   nonlocal probes,probe_cost
   if a not in cache:
    st=time.perf_counter();cache[a]=encode(b,a);probe_cost+=time.perf_counter()-st;probes+=1
   return cache[a]
  if policy=='fixed': chosen=incumbent
  elif policy=='periodic':
   pool= rng.sample(ACTIONS,min(budget,len(ACTIONS))) if i%4==0 else []
   for a in pool:eval_action(a)
   chosen=min(cache,key=lambda a:len(cache[a])) if cache else incumbent
  elif policy=='random':
   for a in rng.sample(ACTIONS,min(budget,len(ACTIONS))):eval_action(a)
   chosen=min(cache,key=lambda a:len(cache[a])) if cache else incumbent
  elif policy=='drift':
   # Exact matched maximum candidate-probe budget; threshold alone does not mandate probing.
   if prev_feature is None or shift>threshold:
    pool=[('identity',c) for c in AVAILABLE if c in ('zstd1','zlib6','raw','lz4')]
    pool += [('delta1',c) for c in AVAILABLE if c in ('zstd1','zlib6')]
    pool += [('xor1',c) for c in AVAILABLE if c in ('zstd1','zlib6')]
    ordered=[a for a in pool if a!=incumbent]
    for a in ordered[:budget]:eval_action(a)
    # Compare byte savings only; spent probe time is still included in total runtime.
    chosen=min(cache,key=lambda a:len(cache[a])) if cache else incumbent
   else:chosen=incumbent
  elif policy=='oracle':
   for a in ACTIONS:eval_action(a)
   chosen=min(cache,key=lambda a:len(cache[a]))
  else:raise ValueError(policy)
  # Already encoded candidates are reusable, otherwise do a single actual encode.
  blob=eval_action(chosen) if chosen not in cache else cache[chosen]
  if decode(blob)!=b:raise AssertionError('Lossless mismatch')
  incumbent=chosen;prev_feature=features
  record={'block':i,'policy':policy,'codec':chosen[1],'transform':chosen[0],'shift':shift,'encoded_bytes':len(blob),'original_bytes':len(b),'block_wall_s':time.perf_counter()-t0,'probe_compute_s':probe_cost,'probes':len(cache)-int(chosen not in cache)}
  # Number of encodes is len(cache), including chosen; every encode counted in wall time.
  record['encodes']=len(cache);records.append(record);all_blobs.append(blob)
 total=time.perf_counter()-start
 return {'bytes':sum(map(len,all_blobs)),'seconds':total,'encodes':sum(r['encodes'] for r in records),'probe_s':sum(r['probe_compute_s'] for r in records),'records':records,'sha256':hashlib.sha256(b''.join(blocks)).hexdigest()}
def write_csv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def save_json(p,obj):Path(p).write_text(json.dumps(obj,indent=2,default=str))
def export(tag):
 bundle=Path(f'{tag}_ShiftGuardX_results.zip')
 files=[p for p in OUT.rglob('*') if p.is_file()]
 manifest={'notebook':tag,'utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'python':platform.python_version(),'platform':platform.platform(),'codecs':AVAILABLE,'codec_ids':CODECS,'files':[{'file':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
 save_json(OUT/'manifest.json',manifest)
 with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for p in OUT.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(OUT))
 print('CREATED',bundle.resolve(),bundle.stat().st_size,'bytes')
 try:
  from google.colab import files as colab_files
  colab_files.download(str(bundle))
 except ImportError:pass
 return bundle
print('Available codecs:', AVAILABLE)


"""
## Run fair matched-container comparisons
"""


import gc
sources=data_sources()
rows=[]; details=[]
for name,raw in sources.items():
 blocks=chunks(raw)
 # Fixed action experiments: exact same container, blocks, transform coding and measurement path.
 for action in ACTIONS:
  t=time.perf_counter(); blobs=[encode(b,action) for b in blocks];seconds=time.perf_counter()-t
  assert b''.join(decode(z) for z in blobs)==raw
  size=sum(map(len,blobs))
  rows.append({'dataset':name,'method':'fixed','action':'/'.join(action),'raw_bytes':len(raw),'encoded_bytes':size,'bits_per_byte':8*size/len(raw),'seconds':seconds,'throughput_mib_s':len(raw)/1048576/max(seconds,1e-12),'encodes':len(blocks),'budget':0,'verified':True})
 # Online methods use identical blocks and a max of 2 alternative evaluations per block.
 for policy in ['random','periodic','drift','oracle']:
  for seed in ([11,22,33] if policy!='oracle' else [11]):
   r=run_policy(blocks,policy,budget=2,seed=seed,threshold=.7)
   rows.append({'dataset':name,'method':policy,'action':f'seed={seed}','raw_bytes':len(raw),'encoded_bytes':r['bytes'],'bits_per_byte':8*r['bytes']/len(raw),'seconds':r['seconds'],'throughput_mib_s':len(raw)/1048576/max(r['seconds'],1e-12),'encodes':r['encodes'],'budget':2,'verified':True})
   for rec in r['records']:details.append({'dataset':name,'seed':seed,**rec})
 gc.collect()
write_csv(OUT/'NB04B_matched_benchmarks.csv',rows)
write_csv(OUT/'NB04B_block_decisions.csv',details)
save_json(OUT/'NB04B_protocol.json',{'block_size':BLOCK,'dataset_names':list(sources),'seed_list':[11,22,33],'budget_alternatives_per_block':2,'oracle_is_not_deployable':True,'note':'Synthetic data only unless external_ datasets supplied. Timing single-shot; rerun on stable hardware for paper claims.'})
for name in sources:
 x=[z for z in rows if z['dataset']==name]
 best=min([z for z in x if z['method']=='fixed'],key=lambda z:z['encoded_bytes'])
 drift=[z for z in x if z['method']=='drift'][0]
 print(name,'best fixed',best['action'],best['encoded_bytes'],'drift',drift['encoded_bytes'],'ratio-to-best-fixed',round(drift['encoded_bytes']/best['encoded_bytes'],3))


"""
## Automatic results download
Upload the ZIP file to ChatGPT for audit.
"""

export('NB04B')

