#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX-NB31 — Robust_Calibration_Selector

Precommits selection to the first quarter of each stream and rejects options that do not exceed a calibration safety margin. Evaluates exactly one frozen choice on held-out values.

Run all cells in a fresh Colab CPU runtime. Each notebook is independent and generates its own ZIP. Measurements must not be treated as state-of-the-art evidence without matched specialized baselines and independent validation.
"""

import csv,io,json,hashlib,urllib.request,struct,math,collections,time,statistics,random,sys,platform,zlib,lzma,zipfile,traceback
from pathlib import Path
NB = 31
OUT=Path('/content' if Path('/content').exists() else '/mnt/data')/f'SGX_NB{NB}_outputs'
OUT.mkdir(parents=True,exist_ok=True)
MAX_VALUES=20000
SEED=20261022
errors=[]; provenance=[]
def fetch(url):
 req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (lossless-compression-academic-test)'})
 with urllib.request.urlopen(req,timeout=45) as resp:return resp.read()
def write_rows(filename, rows):
 if not rows:return
 cols=list(dict.fromkeys(k for x in rows for k in x.keys()))
 with (OUT/filename).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
def get_data():
 # New dataset families compared with prior UCI AirQuality/Power exploratory experiments.
 sources=[('seaborn_penguins','https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv',['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']),('seaborn_diamonds','https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv',['carat','depth','table','price','x','y','z']),('seaborn_fmri','https://raw.githubusercontent.com/mwaskom/seaborn-data/master/fmri.csv',['signal'])]
 result=[]
 for dataset,url,fields in sources:
  try:
   blob=fetch(url)
   rd=csv.DictReader(io.StringIO(blob.decode('utf-8-sig')))
   rows=list(rd)
   for field in fields:
    values=[]
    for row in rows:
     if len(values)>=MAX_VALUES:break
     try:v=float(row.get(field,''))
     except (ValueError,TypeError):continue
     if math.isfinite(v):values.append(v)
    if len(values)<100:continue
    raw=struct.pack('<'+'d'*len(values),*values)
    name=dataset+'__'+field
    result.append((name,raw,'REAL_PUBLIC_DATA'))
    provenance.append({'name':name,'url':url,'source_sha256':hashlib.sha256(blob).hexdigest(),'raw_sha256':hashlib.sha256(raw).hexdigest(),'n':len(values),'source_kind':'real observational data','column_order':'as published'})
  except Exception as e:errors.append({'dataset':dataset,'error':repr(e),'url':url})
 (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
 (OUT/'download_errors.json').write_text(json.dumps(errors,indent=2))
 return result
def to_words(raw):return [x[0] for x in struct.iter_unpack('<Q',raw)]
def entropy(vals):
 if not vals:return 0.
 c=collections.Counter(vals);n=len(vals)
 return -sum((x/n)*math.log2(x/n) for x in c.values())
def save_zip():
 (OUT/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'notebook':NB},indent=2))
 output=OUT.parent/f'SGX_NB{NB}_results.zip'
 with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for p in sorted(OUT.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(OUT))
 print('RESULTS ZIP:',output,'bytes=',output.stat().st_size)
 try:
  from google.colab import files
  files.download(str(output))
 except ImportError:pass
 return output


# Experimental reversible layouts. NONE of these building blocks is claimed novel.
MAGIC=b'SGX28V01'; HEAD=8+1+4
LAYOUTS=['identity','byte_shuffle','field_split','xor_shuffle']
def trans(raw,mode):
 n=len(raw)//8
 if mode=='identity':return raw
 if mode=='byte_shuffle':return b''.join(raw[i::8] for i in range(8))
 if mode=='field_split':
  # Sign+exponent, high mantissa, low mantissa lanes of IEEE-754 bytes;
  # fixed layout is invertible, including NaN payloads and signed zeros.
  return b''.join(raw[j::8] for j in (7,6,5,4,3,2,1,0))
 if mode=='xor_shuffle':
  w=to_words(raw);prev=0; a=bytearray()
  for v in w:
   a.extend(struct.pack('<Q',v^prev));prev=v
  a=bytes(a);return b''.join(a[i::8] for i in range(8))
 raise ValueError(mode)
def untrans(encoded,mode):
 n=len(encoded)//8
 if mode=='identity':return encoded
 if mode in ('byte_shuffle','field_split'):
  indices=range(8) if mode=='byte_shuffle' else (7,6,5,4,3,2,1,0)
  out=bytearray(len(encoded))
  for block,j in enumerate(indices):out[j::8]=encoded[block*n:(block+1)*n]
  return bytes(out)
 if mode=='xor_shuffle':
  b=bytearray(len(encoded))
  for j in range(8):b[j::8]=encoded[j*n:(j+1)*n]
  prev=0;out=bytearray()
  for (x,) in struct.iter_unpack('<Q',b):
   prev=x^prev;out.extend(struct.pack('<Q',prev))
  return bytes(out)
 raise ValueError(mode)
def pack(raw,mode='identity',backend='zlib'):
 assert len(raw)%8==0
 t=trans(raw,mode)
 payload=zlib.compress(t,6) if backend=='zlib' else lzma.compress(t,preset=1)
 return MAGIC+bytes([LAYOUTS.index(mode),0 if backend=='zlib' else 1])+struct.pack('<I',len(raw)//8)+payload
def unpack(blob):
 assert blob[:8]==MAGIC
 mode=LAYOUTS[blob[8]];backend=blob[9];n=struct.unpack('<I',blob[10:14])[0]
 p=zlib.decompress(blob[14:]) if backend==0 else lzma.decompress(blob[14:])
 assert len(p)==n*8
 return untrans(p,mode)
# Baseline files retain their library-native container accounting, so we separately
# report 14-byte candidate header and never claim perfectly matched formats.
def base(raw,backend):return zlib.compress(raw,6) if backend=='zlib' else lzma.compress(raw,preset=1)
def base_decode(blob,backend):return zlib.decompress(blob) if backend=='zlib' else lzma.decompress(blob)


inputs=get_data();results=[];measurements=[]
MARGIN=.05
for name,raw,_ in inputs:
 n=len(raw)//8;cut=max(16,n//4);cal=raw[:cut*8];test=raw[cut*8:]
 if len(test)<128:continue
 # Select baseline backend using calibration ONLY, and compare candidate on that same backend.
 backends=['zlib','lzma'];bb=min(backends,key=lambda b:len(base(cal,b)))
 cal_base=len(base(cal,bb));options=[]
 for mode in ['byte_shuffle','field_split','xor_shuffle']:
  b=pack(cal,mode,bb);assert unpack(b)==cal
  options.append((len(b),mode))
 best_size,best_mode=min(options)
 choose=best_size<=cal_base*(1-MARGIN)
 frozen_mode=best_mode if choose else 'identity_native'
 for repeat in range(5):
  t=time.perf_counter();b0=base(test,bb);te0=time.perf_counter()-t
  t=time.perf_counter();assert base_decode(b0,bb)==test;td0=time.perf_counter()-t
  t=time.perf_counter();candidate=pack(test,best_mode,bb) if choose else b0;te1=time.perf_counter()-t
  t=time.perf_counter();decoded=unpack(candidate) if choose else base_decode(candidate,bb);td1=time.perf_counter()-t
  assert decoded==test
  measurements.extend([{'dataset':name,'method':'baseline','repeat':repeat,'size':len(b0),'encode_sec':te0,'decode_sec':td0,'exact':True},{'dataset':name,'method':'policy','repeat':repeat,'size':len(candidate),'encode_sec':te1,'decode_sec':td1,'exact':True}])
 results.append({'dataset':name,'backend':bb,'frozen_choice':frozen_mode,'cal_baseline_bytes':cal_base,'cal_candidate_bytes':best_size,'test_baseline_bytes':len(b0),'test_policy_bytes':len(candidate),'test_savings_percent':100*(len(b0)-len(candidate))/len(b0),'test_n':len(test)//8,'exact':True})
write_rows('frozen_policy.csv',results);write_rows('repeated_measurements.csv',measurements)
(OUT/'decision.json').write_text(json.dumps({'datasets':len(results),'positive':sum(r['test_savings_percent']>0 for r in results),'regression':sum(r['test_savings_percent']<0 for r in results),'margin':MARGIN,'decision':'NEEDS_EXTERNAL_VALIDATION_AND_NOVELTY' if results else 'INCOMPLETE','limitations':['Same families as NB27–29; not independent','Header on candidate included; native baseline uses native wrapper','Frozen fallback avoids size regression only when calibration rejects candidate; no universal guarantee']},indent=2))
print('finished',len(results),'streams')


save_zip()

