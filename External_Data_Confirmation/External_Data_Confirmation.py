#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX-NB33 — External_Data_Confirmation

Runs a predeclared, unchanged selector on a fresh external numeric collection (OpenML/NOAA-style not required); data sources are the UCI Wine Quality red/white datasets never used for SGX NB27–29. Reports failures without synthetic fallback.

Run all cells in a fresh Colab CPU runtime. Each notebook is independent and generates its own ZIP. Measurements must not be treated as state-of-the-art evidence without matched specialized baselines and independent validation.
"""

import csv,io,json,hashlib,urllib.request,struct,math,collections,time,statistics,random,sys,platform,zlib,lzma,zipfile,traceback
from pathlib import Path
NB = 33
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


# Confirmatory gate: distinct independent public data, fixed margin and fixed candidate family.
# Column order is published order, missing/nonfinite omitted, no tuned thresholds.
URLS=[('wine_red','https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'),('wine_white','https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv')]
streams=[]
for group,url in URLS:
 try:
  blob=fetch(url);r=csv.DictReader(io.StringIO(blob.decode('utf-8-sig')),delimiter=';')
  series=collections.defaultdict(list)
  for row in r:
   for k,v in row.items():
    try:
     x=float(v)
     if math.isfinite(x):series[k].append(x)
    except (ValueError,TypeError):pass
  for col,values in series.items():
   if len(values)<100:continue
   raw=struct.pack('<'+'d'*len(values),*values)
   streams.append((group+'__'+col,raw))
   provenance.append({'dataset':group,'column':col,'url':url,'source_sha256':hashlib.sha256(blob).hexdigest(),'raw_sha256':hashlib.sha256(raw).hexdigest(),'n':len(values)})
 except Exception as e:errors.append({'dataset':group,'error':repr(e)})
(OUT/'external_provenance.json').write_text(json.dumps(provenance,indent=2))
(OUT/'external_errors.json').write_text(json.dumps(errors,indent=2))
results=[]
for name,raw in streams:
 n=len(raw)//8;cut=max(16,n//4);cal=raw[:cut*8];test=raw[cut*8:]
 if not test:continue
 bb=min(['zlib','lzma'],key=lambda b:len(base(cal,b)))
 cb=base(cal,bb);opts=[]
 for mode in ['byte_shuffle','field_split','xor_shuffle']:
  x=pack(cal,mode,bb);assert unpack(x)==cal
  opts.append((len(x),mode))
 best,mode=min(opts);selected=best<=.95*len(cb)
 t0=time.perf_counter();baseline=base(test,bb);base_enc_sec=time.perf_counter()-t0
 t0=time.perf_counter();policy=pack(test,mode,bb) if selected else baseline;policy_enc_sec=time.perf_counter()-t0
 assert base_decode(baseline,bb)==test
 assert (unpack(policy) if selected else base_decode(policy,bb))==test
 results.append({'dataset':name,'n_test':len(test)//8,'backend':bb,'selected_mode':mode if selected else 'baseline','baseline_bytes':len(baseline),'policy_bytes':len(policy),'savings_percent':100*(len(baseline)-len(policy))/len(baseline),'exact':True,'baseline_encode_sec':base_enc_sec,'policy_encode_sec':policy_enc_sec})
write_rows('external_confirmatory_results.csv',results)
positive=sum(x['savings_percent']>0 for x in results);regress=sum(x['savings_percent']<0 for x in results)
summary={'real_streams':len(results),'positive':positive,'regressions':regress,'decision':'INCOMPLETE_NO_EXTERNAL_DATA' if not results else ('CONTINUE_BUT_NOVELTY_UNPROVEN' if positive>=max(2,len(results)//4) and regress<=1 else 'STOP_FIELD_SELECTOR_AS_MAIN_METHOD'),'limitations':['Wine quality has relatively small numerical columns','Published datasets may have been seen generally in compression research','Specialized ALP comparison not included','This method combines established coding transformations']}
(OUT/'confirmatory_gate.json').write_text(json.dumps(summary,indent=2));print(summary)


save_zip()

