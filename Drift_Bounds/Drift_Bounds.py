#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# NB05B — Corrected Budgeted Controller

Debugs the budget-one regression. Always retains a tested incumbent and treats probe budget as extra actions. Uses nested ordered probes, same container and decoder, block-level audit records, and automatic ZIP export. Budget monotonicity is asserted for one fixed block; across several blocks, history-dependent controllers need not be monotonic.
"""

# %pip -q install zstandard lz4



import os, sys, math, time, csv, json, random, hashlib, zlib, bz2, lzma, struct, zipfile, platform
from pathlib import Path
from collections import Counter
try:
 import zstandard as zstd
except ImportError: zstd=None
try:
 import lz4.frame as lz4f
except ImportError: lz4f=None
OUT=Path('sgx_results'); OUT.mkdir(exist_ok=True)
CODECS=['raw','zlib6','bz2','lzma1','zstd1','lz4']
AVAILABLE=[c for c in CODECS if (c!='zstd1' or zstd) and (c!='lz4' or lz4f)]
TRANSFORMS=['identity','delta1','xor1','delta2','context4']
HDR=struct.Struct('<4sBBBBIII')  # magic, version, transform, backend, predictor-param, n, payload, crc
BLOCK=16384

def residual(data, tr):
 if tr=='identity':return data
 out=bytearray(len(data))
 for i,v in enumerate(data):
  pred=(data[i-1] if i>=1 else 0)
  if tr=='delta1':out[i]=(v-pred)&255
  elif tr=='xor1':out[i]=v^pred
  elif tr=='delta2':out[i]=(v-(2*data[i-1]-data[i-2] if i>=2 else pred))&255
  elif tr=='context4':out[i]=(v-(data[i-4] if i>=4 else 0))&255
  else:raise ValueError(tr)
 return bytes(out)
def invert(data,tr):
 if tr=='identity':return data
 out=bytearray(len(data))
 for i,v in enumerate(data):
  pred=(out[i-1] if i>=1 else 0)
  if tr=='delta1':out[i]=(v+pred)&255
  elif tr=='xor1':out[i]=v^pred
  elif tr=='delta2':out[i]=(v+(2*out[i-1]-out[i-2] if i>=2 else pred))&255
  elif tr=='context4':out[i]=(v+(out[i-4] if i>=4 else 0))&255
  else:raise ValueError(tr)
 return bytes(out)
def codec_encode(b,c):
 if c=='raw':return b
 if c=='zlib6':return zlib.compress(b,6)
 if c=='bz2':return bz2.compress(b,compresslevel=5)
 if c=='lzma1':return lzma.compress(b,preset=1)
 if c=='zstd1':return zstd.ZstdCompressor(level=1).compress(b)
 if c=='lz4':return lz4f.compress(b)
 raise ValueError(c)
def codec_decode(b,c):
 if c=='raw':return b
 if c=='zlib6':return zlib.decompress(b)
 if c=='bz2':return bz2.decompress(b)
 if c=='lzma1':return lzma.decompress(b)
 if c=='zstd1':return zstd.ZstdDecompressor().decompress(b)
 if c=='lz4':return lz4f.decompress(b)
 raise ValueError(c)
def encode(data, action):
 tr,c=action; pay=codec_encode(residual(data,tr),c)
 return HDR.pack(b'SGX3',1,TRANSFORMS.index(tr),CODECS.index(c),0,len(data),len(pay),zlib.crc32(data))+pay
def decode(blob):
 if len(blob)<HDR.size:raise ValueError('short header')
 magic,ver,t,c,param,n,k,crc=HDR.unpack_from(blob)
 if magic!=b'SGX3' or ver!=1 or t>=len(TRANSFORMS) or c>=len(CODECS) or param!=0 or k!=len(blob)-HDR.size:raise ValueError('invalid container')
 data=invert(codec_decode(blob[HDR.size:],CODECS[c]),TRANSFORMS[t])
 if len(data)!=n or zlib.crc32(data)!=crc:raise ValueError('corrupt or truncated')
 return data

def datasets(seed=2026):
 rng=random.Random(seed)
 walk=bytearray();x=120
 for i in range(131072):
  x=(x+rng.choice([-1,0,0,0,1]))&255;walk.append(x)
 log=(b'INFO route=/api/items status=200 tenant=42\n'*4000)[:131072]
 jsonl=b''.join((f'{{"id":{i},"region":"APAC","ok":true,"count":{i%13}}}\n').encode() for i in range(3500))[:131072]
 random_bytes=bytes(rng.randrange(256) for _ in range(131072))
 structured=b''.join(struct.pack('<I',i%65536) for i in range(32768))
 mixed=bytes(walk[:32768])+log[:32768]+random_bytes[:32768]+jsonl[:32768]
 d={'sensor_walk':bytes(walk),'logs':log,'jsonl':jsonl,'random':random_bytes,'uint32_structured':structured,'mixed_regimes':mixed}
 directory=Path('sgx_public_data')
 if directory.exists():
  for p in sorted(directory.iterdir()):
   if p.is_file() and 1024<=p.stat().st_size<=16_000_000:d['public_'+p.name]=p.read_bytes()
 return d

def blocks(b):return [b[i:i+BLOCK] for i in range(0,len(b),BLOCK)]
def write_csv(file,records):
 if not records:return
 with (OUT/file).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def save_json(file,obj):(OUT/file).write_text(json.dumps(obj,indent=2,default=str))
def export(tag):
 files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json']
 manifest={'notebook':tag,'python':platform.python_version(),'utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'backend_available':AVAILABLE,'files':[{'path':str(p.relative_to(OUT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files]}
 save_json('manifest.json',manifest)
 target=Path(f'{tag}_ShiftGuardX_results.zip')
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  for p in OUT.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(OUT))
 print('EXPORTED',target.resolve(),target.stat().st_size,'bytes')
 try:
  from google.colab import files as colab_files
  colab_files.download(str(target))
 except ImportError:pass
 return target
print('Available backends',AVAILABLE)


"""
## Run experiment
"""


# Fix regression: ALWAYS encode incumbent before considering alternatives;
# probe budget means ADDITIONAL alternatives, never includes incumbent.
# Candidate schedule is deterministic and nested, so more probes cannot increase
# chosen encoded size on the SAME block. Probe time CAN increase.
fast_backend='zstd1' if 'zstd1' in AVAILABLE else 'zlib6'
incumbent_default=('identity',fast_backend)
POOL=[('identity',c) for c in ('zstd1','zlib6','lz4','raw','bz2','lzma1') if c in AVAILABLE]
POOL +=[(t,c) for t in ['delta1','xor1','delta2','context4'] for c in ('zstd1','zlib6') if c in AVAILABLE]
assert incumbent_default in POOL

def policy_run(source,budget=1,threshold=.5,seed=1):
 items=blocks(source); incumbent=incumbent_default; previous=None
 events=[];encoded=[];t_start=time.perf_counter();n_probes=0
 for index,b in enumerate(items):
  start=time.perf_counter(); counts=Counter(b[:4096]); freq=[counts.get(i,0)/max(1,min(len(b),4096)) for i in range(256)]
  shift=(sum(abs(x-y) for x,y in zip(freq,previous)) if previous is not None else 2.)
  triggered=previous is None or shift>threshold
  cache={}; probe_seconds=0.
  def measure(action,is_probe=False):
   nonlocal probe_seconds,n_probes
   if action not in cache:
    t0=time.perf_counter(); cache[action]=encode(b,action); dt=time.perf_counter()-t0
    if is_probe:probe_seconds+=dt;n_probes+=1
   return cache[action]
  incumbent_blob=measure(incumbent)
  # Deterministic nested ordering, favor fast native backends for low probe budgets.
  # Rotate only the LIST ordering by previous historical winner; no current-block oracle knowledge.
  candidates=[a for a in POOL if a!=incumbent]
  if triggered:
   for action in candidates[:min(budget,len(candidates))]:measure(action,True)
  chosen=min(cache,key=lambda a:(len(cache[a]),a!=incumbent))
  best_blob=cache[chosen]
  assert len(best_blob)<=len(incumbent_blob)
  assert decode(best_blob)==b
  events.append({'block':index,'budget':budget,'threshold':threshold,'drift_l1':shift,'triggered':triggered,'incumbent_before':'/'.join(incumbent),'chosen':'/'.join(chosen),'incumbent_bytes':len(incumbent_blob),'chosen_bytes':len(best_blob),'extra_probes':len(cache)-1,'probe_s':probe_seconds,'block_total_s':time.perf_counter()-start,'roundtrip':True})
  incumbent=chosen;previous=freq;encoded.append(best_blob)
 return {'encoded_bytes':sum(map(len,encoded)),'elapsed_s':time.perf_counter()-t_start,'probes':n_probes,'records':events}

# Controlled single-block invariant: fixed incumbent and nested candidate list.
# A larger budget can never INCREASE compressed bytes in this diagnostic setting.
testdata=datasets()['logs'][:BLOCK]
observed=[]
for budget in [0,1,2,4,8,16]:
 res=policy_run(testdata,budget=budget)
 observed.append(res['encoded_bytes'])
assert all(a>=b for a,b in zip(observed,observed[1:])),observed
print('Single block size by budget:',observed)

results=[];events=[]
for name,data in datasets().items():
 for threshold in [.15,.5,1.0]:
  for budget in [0,1,2,4,8]:
   result=policy_run(data,budget,threshold)
   results.append({'dataset':name,'budget':budget,'threshold':threshold,'raw_bytes':len(data),'encoded_bytes':result['encoded_bytes'],'ratio':len(data)/result['encoded_bytes'],'total_s':result['elapsed_s'],'throughput_mib_s':len(data)/1048576/max(result['elapsed_s'],1e-12),'probes':result['probes'],'verified':True})
   events.extend({'dataset':name,**record} for record in result['records'])
write_csv('NB05B_controller_grid.csv',results)
write_csv('NB05B_block_decisions.csv',events)
save_json('NB05B_protocol.json',{'block_size':BLOCK,'budget_definition':'maximum additional transform+codec encodes per triggered block, above an always-encoded incumbent','drift':'L1 distance between sampled byte-frequency vectors','selection':'minimum measured encoded size including container overhead','caveat':'different prior decisions mean multi-block byte totals need not be monotonic in budget; runtime always includes probes','data_provenance':'synthetic unless public_ prefix'})
print('Rows:',len(results),'decisions:',len(events))
for row in [x for x in results if x['dataset']=='logs' and x['threshold']==.5]:print(row['budget'],row['encoded_bytes'],round(row['throughput_mib_s'],2))


"""
## Export complete results ZIP
"""

export('NB05B')

