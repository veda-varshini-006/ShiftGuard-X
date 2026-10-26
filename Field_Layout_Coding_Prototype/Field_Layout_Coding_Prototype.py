#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX-NB28 — Field_Layout_Coding_Prototype

Independent Colab CPU notebook. Data downloads are real-world sources; failures are recorded rather than replaced. Outputs are exploratory. No claim of codec novelty or Q1 acceptance.
"""

import csv,io,json,hashlib,urllib.request,struct,math,collections,time,statistics,random,sys,platform,zlib,lzma,zipfile,traceback
from pathlib import Path
NB = 28
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


rng=random.Random(SEED)
samples=[bytes(rng.randrange(256) for _ in range(8*37)),b'\x00'*8*200,struct.pack('<QQQQQQ',0,1<<63,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000001234,0x7ff0000000000001)]
checks=[]
for i,raw in enumerate(samples):
 for mode in LAYOUTS:
  for backend in ['zlib','lzma']:
   try:
    b=pack(raw,mode,backend);dec=unpack(b);ok=dec==raw;assert ok
    checks.append({'case':i,'mode':mode,'backend':backend,'ok':ok,'compressed_bytes':len(b)})
   except Exception as exc:checks.append({'case':i,'mode':mode,'backend':backend,'ok':False,'error':repr(exc)});raise
write_rows('codec_correctness.csv',checks)
inputs=get_data();results=[]
for name,raw,kind in inputs:
 for mode in LAYOUTS:
  for backend in ['zlib','lzma']:
   try:
    start=time.perf_counter();enc=pack(raw,mode,backend);encode_sec=time.perf_counter()-start
    start=time.perf_counter();dec=unpack(enc);decode_sec=time.perf_counter()-start
    assert dec==raw
    results.append({'dataset':name,'mode':mode,'backend':backend,'n':len(raw)//8,'encoded_bytes':len(enc),'bpv':8*len(enc)/(len(raw)//8),'encode_sec':encode_sec,'decode_sec':decode_sec,'exact':True,'candidate_header_bytes':14})
   except Exception as exc:errors.append({'dataset':name,'mode':mode,'backend':backend,'error':repr(exc)})
write_rows('layout_comparison.csv',results)
(OUT/'runtime_errors.json').write_text(json.dumps(errors,indent=2))
print('Measured',len(results),'layout records on',len(inputs),'real streams. No selection performed.')


(OUT/'prior_art_warning.txt').write_text('Field splitting, byte shuffling and XOR prediction are established. This notebook measures a falsifiable layout hypothesis; it does not establish an original compressor.\n')


save_zip()

