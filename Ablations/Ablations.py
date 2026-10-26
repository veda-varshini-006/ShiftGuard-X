#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# NB06 — Compression-Specific Residual Coding

Evaluates predictor residual representations under an IDENTICAL lossless coding backend. Introduces a lag-4/context residual candidate as a diagnostic—not a verified novel algorithm. Includes fixed predictor baselines, per-block oracle (evaluation only), a deployable one-probe controller, exact decoding, exported CSVs, and explicit limitations.
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


# Compression-specific investigation: context-lag residual representation.
# Candidate is context4 (lag-4 byte predictor), vs classical identity, delta1,
# second order delta, XOR. Research hypothesis, NOT a claim of novel predictor.
# All approaches share EXACT same backend and self-describing container.
# We add a mode-switch experiment where one predictor is chosen per block.
# Switching is penalized by full measured container bytes and only occurs if cheaper.
BACKEND='zstd1' if 'zstd1' in AVAILABLE else 'zlib6'
fixed_predictors=['identity','delta1','xor1','delta2','context4']
records=[];block_records=[]
for name,data in datasets().items():
 chunks=blocks(data)
 for tr in fixed_predictors:
  action=(tr,BACKEND);t0=time.perf_counter(); encoded=[encode(b,action) for b in chunks];duration=time.perf_counter()-t0
  assert b''.join(decode(e) for e in encoded)==data
  records.append({'dataset':name,'method':'fixed_'+tr,'backend':BACKEND,'raw_bytes':len(data),'encoded_bytes':sum(map(len,encoded)),'total_s':duration,'bps':8*sum(map(len,encoded))/len(data),'verified':True})
 # Offline per-block minimum: NOT a deployable policy; reports coding headroom.
 t0=time.perf_counter();chosen=[]
 for i,b in enumerate(chunks):
  local=[(tr,encode(b,(tr,BACKEND))) for tr in fixed_predictors]
  tr,blob=min(local,key=lambda x:len(x[1]))
  assert decode(blob)==b
  chosen.append(blob)
  block_records.append({'dataset':name,'block':i,'winning_transform':tr,'best_encoded_bytes':len(blob),'raw_bytes':len(b)})
 elapsed=time.perf_counter()-t0
 records.append({'dataset':name,'method':'oracle_per_block','backend':BACKEND,'raw_bytes':len(data),'encoded_bytes':sum(map(len,chosen)),'total_s':elapsed,'bps':8*sum(map(len,chosen))/len(data),'verified':True})
 # Deployable 1-probe hysteresis policy: try previous and 1 challenger selected
 # from feature heuristic, then measure BOTH (no oracle cheating).
 t0=time.perf_counter();previous='identity';encoded=[]
 for i,b in enumerate(chunks):
  # In a real encoder, feature extraction also costs CPU and is included.
  sample=b[:4096]
  q1=sum(sample[j]==sample[j-1] for j in range(1,len(sample))) / max(1,len(sample)-1)
  q4=sum(sample[j]==sample[j-4] for j in range(4,len(sample))) / max(1,len(sample)-4)
  challenger='context4' if q4>q1+0.04 else 'delta1'
  incumbent_blob=encode(b,(previous,BACKEND))
  alternative_blob=encode(b,(challenger,BACKEND)) if challenger!=previous else incumbent_blob
  if len(alternative_blob)+8<len(incumbent_blob):previous=challenger;blob=alternative_blob
  else:blob=incumbent_blob
  assert decode(blob)==b
  encoded.append(blob)
 elapsed=time.perf_counter()-t0
 records.append({'dataset':name,'method':'context_switch_one_probe','backend':BACKEND,'raw_bytes':len(data),'encoded_bytes':sum(map(len,encoded)),'total_s':elapsed,'bps':8*sum(map(len,encoded))/len(data),'verified':True})
write_csv('NB06_predictor_comparisons.csv',records)
write_csv('NB06_oracle_blocks.csv',block_records)
save_json('NB06_protocol.json',{'backend':BACKEND,'block_size':BLOCK,'principle':'same backend, header, blocks across every predictor','context4':'predict byte from 4 positions earlier modulo 256; reversible; established lag prediction and NOT inherently novel','oracle':'evaluation-only; compresses every candidate, not a deployable comparator','critical_next_question':'whether an original coding mechanism can beat existing predictive + entropy coder baselines on unseen public datasets'})
for name in datasets():
 rr=[x for x in records if x['dataset']==name]
 best=min((x for x in rr if x['method'].startswith('fixed_')),key=lambda z:z['encoded_bytes'])
 proposed=next(x for x in rr if x['method']=='context_switch_one_probe')
 print(name,'best fixed:',best['method'],best['encoded_bytes'],'one-probe:',proposed['encoded_bytes'],'gap:',round(proposed['encoded_bytes']/best['encoded_bytes']-1,3))


"""
## Export complete results ZIP
"""

export('NB06')

