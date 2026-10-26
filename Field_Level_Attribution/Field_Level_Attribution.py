#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX-NB30 — Field_Level_Attribution

Measures how each exponent/sign/fraction lane affects compressibility. This is a diagnostic, not a new codec.

Run all cells in a fresh Colab CPU runtime. Each notebook is independent and generates its own ZIP. Measurements must not be treated as state-of-the-art evidence without matched specialized baselines and independent validation.
"""

import csv,io,json,hashlib,urllib.request,struct,math,collections,time,statistics,random,sys,platform,zlib,lzma,zipfile,traceback
from pathlib import Path
NB = 30
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


inputs=get_data();rows=[];detail=[]
for name,raw,_ in inputs:
 n=len(raw)//8;cut=max(16,int(n*.25));h=raw[cut*8:];n2=len(h)//8
 if n2<20:continue
 lanes=[h[j::8] for j in range(8)]
 vals=to_words(h)
 signexp=[((v>>52)&0xfff) for v in vals]
 fraction_hi=[((v>>32)&0xfffff) for v in vals]
 fraction_lo=[v&0xffffffff for v in vals]
 for lab,items in [('sign_exponent',signexp),('fraction_high20',fraction_hi),('fraction_low32',fraction_lo)]:
  detail.append({'dataset':name,'component':lab,'unique':len(set(items)),'entropy_bits_symbol':entropy(items),'count':len(items)})
 for backend in ['zlib','lzma']:
  for mode in ['identity','byte_shuffle','field_split','xor_shuffle']:
   enc=pack(h,mode,backend)
   assert unpack(enc)==h
   rows.append({'dataset':name,'backend':backend,'mode':mode,'size_bytes':len(enc),'bpv':8*len(enc)/n2,'lossless':True,'comparison':'candidate self-describing format'})
  raw_enc=base(h,backend)
  assert base_decode(raw_enc,backend)==h
  rows.append({'dataset':name,'backend':backend,'mode':'native_identity','size_bytes':len(raw_enc),'bpv':8*len(raw_enc)/n2,'lossless':True,'comparison':'native library stream (no SGX header)'})
 for j,lane in enumerate(lanes):
  detail.append({'dataset':name,'component':f'byte_lane_{j}','unique':len(set(lane)),'entropy_bits_symbol':entropy(lane),'count':len(lane)})
write_rows('layout_attribution.csv',rows);write_rows('field_entropy.csv',detail)
(OUT/'interpretation.json').write_text(json.dumps({'real_streams':len(inputs),'note':'NB28 field_split is byte-lane permutation and not a distinct exponent entropy coder; measure attribution without claiming novelty.','limitations':['Data families used in NB27–29; exploratory','No ALP/Chimp comparison','Entropy marginals do not equal achievable context-dependent code length']},indent=2))
print('datasets',len(inputs),'rows',len(rows))


save_zip()

