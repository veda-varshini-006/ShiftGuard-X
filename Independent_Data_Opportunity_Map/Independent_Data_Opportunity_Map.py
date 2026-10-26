#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX-NB27 — Independent_Data_Opportunity_Map

Independent Colab CPU notebook. Data downloads are real-world sources; failures are recorded rather than replaced. Outputs are exploratory. No claim of codec novelty or Q1 acceptance.
"""

import csv,io,json,hashlib,urllib.request,struct,math,collections,time,statistics,random,sys,platform,zlib,lzma,zipfile,traceback
from pathlib import Path
NB = 27
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


inputs=get_data();audit=[]
for name,raw,kind in inputs:
 w=to_words(raw);n=len(w)
 upper=[(v>>48)&65535 for v in w]
 signs_exp=[(v>>52)&4095 for v in w]
 xor=[a^b for a,b in zip(w[1:],w[:-1])]
 byteplanes=[raw[i::8] for i in range(8)]
 audit.append({'dataset':name,'n':n,'raw_bytes':len(raw),'unique_ratio':len(set(w))/n,'repeat_adjacent_fraction':sum(x==y for x,y in zip(w[1:],w[:-1]))/max(1,n-1),'entropy_upper16':entropy(upper),'entropy_sign_exponent':entropy(signs_exp),'mean_xor_width':statistics.mean([(x.bit_length()+7)//8 for x in xor]) if xor else 0,'byteplane_entropy_avg':statistics.mean([entropy(x) for x in byteplanes]),'baseline_zlib_bpv':8*len(zlib.compress(raw,6))/n,'baseline_lzma_bpv':8*len(lzma.compress(raw,preset=1))/n})
write_rows('independent_dataset_audit.csv',audit)
print('Real datasets:',len(set(x[0].split('__')[0] for x in inputs)),'Streams:',len(inputs))
print('No independent real data loaded' if not inputs else 'Audit completed')


(OUT/'research_hypothesis.json').write_text(json.dumps({'hypothesis':'IEEE-754 field-oriented layout may compress numerical streams more efficiently than unmodified bytes under the same secondary codec.','candidate':'reversible field-wise layout + zlib','null':'No held-out encoded size advantage over matched zlib/lzma baselines','prior_art':'Byte shuffling, field splitting, XOR prediction, zlib and lzma are established; no novelty inferred.','data_is_independent_of_previous_UCI_families':True,'design_stage':'exploratory; avoid tuning on test'},indent=2))


save_zip()

