#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX-NB25 — Dictionary Gain Attribution and Held-Out Generalization

**Goal:** Decide whether NB24's exact-value dictionary coding generalizes to held-out numerical data, after paying for all dictionary headers and file framing.

- Chronological **25% calibration / 75% holdout** split for each real stream. Methods and threshold are fixed **before** examining holdout results.
- Calibration selects the strongest generic codec, and optionally allows a dictionary candidate only if it beats the generic alternative by **at least 2%** on calibration data. Selection stays frozen on holdout.
- Independently record hindsight best baseline and hindsight best dictionary performance **only as diagnostics**, not deployed gains.
- Exact byte-level decoding of all tested held-out streams; no results labeled REAL when downloads fail.
- Dictionary token length, header cost, distinct-symbol ratio, and repetition are measured for attribution; no novelty claim for established methods.
- Official ALP is **not run**; do not interpret these numbers as matched ALP comparisons.

Run **Runtime → Run all** in free Colab CPU. Upload `SGX_NB25_results.zip` when complete.
"""

import os,sys,csv,json,time,hashlib,struct,zlib,lzma,zipfile,io,urllib.request,collections,platform,statistics,traceback,random
from pathlib import Path
OUT=Path('/content/SGX_NB25_outputs') if Path('/content').exists() else Path('SGX_NB25_outputs')
OUT.mkdir(parents=True,exist_ok=True)
TEST_MODE=False  # only set True for a quick synthetic smoke test; default attempts real UCI data
MAX_VALUES=16000  # bounded for free Colab
BLOCK_VALUES=2048
try:
 import zstandard as zstd
except Exception:
 zstd=None
print('Output:',OUT,'zstandard available:',bool(zstd))


# Core format. Every block independently decodable; no reliance on encoder-only state.
MAGIC=b'SGX24A01'
def varint(x):
 assert x>=0
 out=bytearray()
 while x>=128:out.append((x&127)|128);x>>=7
 out.append(x);return bytes(out)
def getvar(b,p):
 x=0;shift=0
 for _ in range(10):
  if p>=len(b):raise ValueError('truncated varint')
  c=b[p];p+=1;x|=(c&127)<<shift
  if not(c&128):return x,p
  shift+=7
 raise ValueError('oversized varint')
def words_from_bytes(raw):
 if len(raw)%8:raise ValueError('not float64 bytes')
 return [raw[i:i+8] for i in range(0,len(raw),8)]
def encode_tokens(words,run_enabled):
 # Dictionary is selected per block. The exact symbols and ordering are transmitted.
 common=collections.Counter(words).most_common(15)
 dictionary=[w for w,c in common if c>=3]
 lookup={w:i+1 for i,w in enumerate(dictionary)}
 body=bytearray();i=0
 while i<len(words):
  w=words[i];j=i+1
  while run_enabled and j<len(words) and words[j]==w:j+=1
  if run_enabled and j-i>=3:
   body.append(255);body+=varint(j-i);body+=w;i=j;continue
  x=lookup.get(w,0);body.append(x)
  if not x:body+=w
  i+=1
 header=bytearray(MAGIC);header+=struct.pack('<I',len(words));header.append(int(run_enabled));header.append(len(dictionary))
 for d in dictionary:header+=d
 return bytes(header)+bytes(body)
def decode_tokens(blob):
 if not blob.startswith(MAGIC) or len(blob)<14:raise ValueError('invalid header')
 p=8;count=struct.unpack_from('<I',blob,p)[0];p+=4
 mode=blob[p];p+=1;k=blob[p];p+=1
 if mode not in (0,1) or k>15:raise ValueError('bad mode/dict')
 if p+8*k>len(blob):raise ValueError('truncated dictionary')
 dictionary=[blob[p+i*8:p+i*8+8] for i in range(k)];p+=8*k
 out=bytearray()
 while len(out)<count*8:
  if p>=len(blob):raise ValueError('truncated symbols')
  code=blob[p];p+=1
  if code==255 and mode:
   run,p=getvar(blob,p)
   if run<3 or p+8>len(blob) or len(out)+8*run>count*8:raise ValueError('bad run')
   w=blob[p:p+8];p+=8;out+=w*run
  elif code==0:
   if p+8>len(blob):raise ValueError('truncated literal')
   out+=blob[p:p+8];p+=8
  elif 1<=code<=k:out+=dictionary[code-1]
  else:raise ValueError('invalid symbol')
 if len(out)!=count*8 or p!=len(blob):raise ValueError('count or trailing bytes')
 return bytes(out)
def backend_pack(data,backend):
 if backend=='none':return b'N'+data
 if backend=='zlib':return b'Z'+zlib.compress(data,6)
 if backend=='lzma':return b'L'+lzma.compress(data,preset=1)
 if backend=='zstd' and zstd:return b'S'+zstd.ZstdCompressor(level=3).compress(data)
 raise ValueError('backend')
def backend_unpack(blob):
 if not blob:raise ValueError('empty')
 tag,p=blob[:1],blob[1:]
 if tag==b'N':return p
 if tag==b'Z':return zlib.decompress(p)
 if tag==b'L':return lzma.decompress(p)
 if tag==b'S' and zstd:return zstd.ZstdDecompressor().decompress(p)
 raise ValueError('unknown backend')
def encode_stream(raw,method):
 if len(raw)%8:raise ValueError('unaligned')
 out=bytearray(b'SGX24STR');n=len(raw)//8;out+=struct.pack('<I',n)
 for start in range(0,n,BLOCK_VALUES):
  part=raw[start*8:min(n,start+BLOCK_VALUES)*8];words=words_from_bytes(part)
  if method=='raw':payload=b'R'+part
  elif method in ('zlib','lzma','zstd'):payload=b'B'+backend_pack(part,method)
  elif method.startswith('dict'):
   run_enabled='run' in method
   backend=method.split('_')[-1]
   payload=b'T'+backend_pack(encode_tokens(words,run_enabled),backend)
  else:raise ValueError('method')
  out+=struct.pack('<I',len(payload))+payload
 return bytes(out)
def decode_stream(blob):
 if not blob.startswith(b'SGX24STR') or len(blob)<12:raise ValueError('bad stream')
 n=struct.unpack_from('<I',blob,8)[0];p=12;out=bytearray()
 while p<len(blob):
  if p+4>len(blob):raise ValueError('truncated block size')
  m=struct.unpack_from('<I',blob,p)[0];p+=4
  if m<1 or p+m>len(blob):raise ValueError('truncated block')
  payload=blob[p:p+m];p+=m
  if payload[:1]==b'R':raw=payload[1:]
  elif payload[:1]==b'B':raw=backend_unpack(payload[1:])
  elif payload[:1]==b'T':raw=decode_tokens(backend_unpack(payload[1:]))
  else:raise ValueError('unknown block')
  if len(raw)%8 or len(raw)>BLOCK_VALUES*8:raise ValueError('bad decoded block')
  out+=raw
 if len(out)!=n*8:raise ValueError('incorrect total size')
 return bytes(out)
METHODS=['raw','zlib','lzma','dict_zlib','dict_run_zlib','dict_lzma','dict_run_lzma']+(['zstd','dict_zstd','dict_run_zstd'] if zstd else [])
print('Methods:',METHODS)


# Real public inputs. No synthetic data will ever be reported as UCI.
def download(url,timeout=45):
 req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 ShiftGuardX-research/1.0'})
 with urllib.request.urlopen(req,timeout=timeout) as resp:return resp.read()
def real_sources():
 specs=[
  ('UCI_Air_Quality','https://archive.ics.uci.edu/static/public/360/air+quality.zip', ['T','RH','CO(GT)']),
  ('UCI_Household_Power','https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip',['Global_active_power','Voltage','Global_intensity'])]
 results=[];src=[]
 for ds,url,fields in specs:
  try:
   payload=download(url)
   z=zipfile.ZipFile(io.BytesIO(payload))
   candidates=[(name,z.read(name)) for name in z.namelist() if name.lower().endswith(('.csv','.txt'))]
   if not candidates:raise RuntimeError('no CSV/TXT inside archive')
   if ds=='UCI_Air_Quality':
    name,body=next(((x,y) for x,y in candidates if 'airqualityuci' in x.lower()),candidates[0]);sep=';'
   else:
    name,body=next(((x,y) for x,y in candidates if 'household_power_consumption' in x.lower()),candidates[0]);sep=';'
   text=body.decode('utf-8-sig',errors='replace')
   reader=csv.DictReader(io.StringIO(text),delimiter=sep)
   columns={key:[] for key in fields}
   for record in reader:
    if all(len(v)>=MAX_VALUES for v in columns.values()):break
    for key in fields:
     if len(columns[key])>=MAX_VALUES:continue
     s=(record.get(key) or '').strip().replace(',','.')
     if not s or s=='?':continue
     try:v=float(s)
     except ValueError:continue
     if ds=='UCI_Air_Quality' and v==-200:continue
     columns[key].append(v)
   for field,values in columns.items():
    if len(values)<100:continue
    raw=struct.pack('<'+'d'*len(values),*values)
    results.append((ds+'__'+field,raw,'REAL_UCI'))
    src.append({'dataset':ds+'__'+field,'kind':'REAL_UCI','url':url,'member':name,'archive_sha256':hashlib.sha256(payload).hexdigest(),'values':len(values),'input_sha256':hashlib.sha256(raw).hexdigest()})
  except Exception as exc:
   src.append({'dataset':ds,'kind':'DOWNLOAD_FAILED','url':url,'error':repr(exc)})
 return results,src
if TEST_MODE:
 datasets=[];provenance=[{'dataset':'ALL','kind':'TEST_MODE','note':'Real download skipped in smoke-test mode'}]
else:
 datasets,provenance=real_sources()
# Synthetic controls are labeled and never counted as real-data evidence.
rng=random.Random(2026)
synthetic=[float((i//32)%23) for i in range(9000)]
synthetic2=[rng.random() for _ in range(9000)]
for name,vals in [('SYNTHETIC_repeated',synthetic),('SYNTHETIC_random',synthetic2)]:
 raw=struct.pack('<'+'d'*len(vals),*vals)
 datasets.append((name,raw,'SYNTHETIC'))
 provenance.append({'dataset':name,'kind':'SYNTHETIC','values':len(vals),'input_sha256':hashlib.sha256(raw).hexdigest()})
with (OUT/'provenance.json').open('w') as f:json.dump(provenance,f,indent=2)
print('Datasets:',[(name,len(raw)//8,kind) for name,raw,kind in datasets])


# Reproduce NB24's end-to-end method framing, plus adversarial bit-exact checks.
checks=[]
patterns={
  'signed_zero': [0x0,0x8000000000000000]*150,
  'nan_payloads': [0x7ff8000000000001,0x7ff8000000000002,0x7ff0000000000001]*100,
  'subnormal': [0x1,0x2,0x8000000000000001]*100,
  'all_repeated': [0x3ff0000000000000]*300,
  'distinct': list(range(300)),
}
for label,words in patterns.items():
 raw=b''.join(struct.pack('<Q',x) for x in words)
 for method in METHODS:
  encoded=encode_stream(raw,method)
  assert decode_stream(encoded)==raw,(label,method)
  checks.append({'case':label,'method':method,'exact_ok':True,'raw_bytes':len(raw),'coded_bytes':len(encoded)})
with (OUT/'correctness.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
print('Special-case tests passed:',len(checks))


# A fixed, preregistered evaluation protocol. Split is by temporal position,
# never random shuffle, and the output series is independently encoded.
CAL_FRACTION=0.25
MIN_CAL_GAIN=0.02
MIN_VALUES=400
BASE=['raw','zlib','lzma']+(['zstd'] if zstd else [])
DICTS=[m for m in METHODS if m.startswith('dict')]
records=[];attribution=[];errors=[];decisions=[]
def measure(raw,method):
 t=time.perf_counter(); encoded=encode_stream(raw,method);enc_secs=time.perf_counter()-t
 t=time.perf_counter(); out=decode_stream(encoded);dec_secs=time.perf_counter()-t
 if out!=raw:raise AssertionError('Byte-exact decoding failed: '+method)
 return {'coded_bytes':len(encoded),'encode_seconds':enc_secs,'decode_seconds':dec_secs,'exact_ok':True}
def stats(raw):
 words=words_from_bytes(raw);n=len(words)
 counts=collections.Counter(words);unique=len(counts)
 adjacent=sum(words[i]==words[i-1] for i in range(1,n))
 top15=sum(c for _,c in counts.most_common(15))/max(1,n)
 tok=encode_tokens(words,False);tok_run=encode_tokens(words,True)
 k=tok[13]
 return {'values':n,'unique_values':unique,'unique_fraction':unique/max(1,n),
         'adjacent_equal_fraction':adjacent/max(1,n-1),'top15_coverage':top15,
         'dictionary_entries':k,'dictionary_entry_bytes':8*k,
         'token_stream_bytes':len(tok),'token_run_stream_bytes':len(tok_run),
         'raw_bytes':len(raw),'dictionary_raw_expansion_bytes':len(tok)-len(raw)}
for dataset, raw,kind in datasets:
 if len(raw)//8<MIN_VALUES:
  errors.append({'dataset':dataset,'stage':'split','error':'insufficient values'});continue
 n=len(raw)//8;ncal=max(100,int(n*CAL_FRACTION))
 cal=raw[:ncal*8];test=raw[ncal*8:]
 scores={}
 try:
  for part,part_raw in [('calibration',cal),('holdout',test)]:
   st=stats(part_raw);attribution.append({'dataset':dataset,'kind':kind,'split':part,**st})
   for method in METHODS:
    met=measure(part_raw,method);scores[(part,method)]=met['coded_bytes']
    records.append({'dataset':dataset,'kind':kind,'split':part,'method':method,
                    'n_values':len(part_raw)//8,'input_bytes':len(part_raw),**met})
  best_base_cal=min(BASE,key=lambda m:scores[('calibration',m)])
  best_dict_cal=min(DICTS,key=lambda m:scores[('calibration',m)])
  bcal=scores[('calibration',best_base_cal)];dcal=scores[('calibration',best_dict_cal)]
  cal_gain=(bcal-dcal)/bcal
  select_dict=(cal_gain>=MIN_CAL_GAIN)
  selected_method=best_dict_cal if select_dict else best_base_cal
  best_base_held=min(BASE,key=lambda m:scores[('holdout',m)])
  best_dict_held=min(DICTS,key=lambda m:scores[('holdout',m)])
  deployed=scores[('holdout',selected_method)]
  fixed_baseline=scores[('holdout',best_base_cal)]
  oracle_base=scores[('holdout',best_base_held)]
  decisions.append({'dataset':dataset,'kind':kind,'cal_n':ncal,'holdout_n':n-ncal,
   'cal_baseline':best_base_cal,'cal_dictionary':best_dict_cal,'cal_dictionary_gain_fraction':cal_gain,
   'frozen_choice':selected_method,'frozen_chose_dictionary':select_dict,
   'frozen_baseline_holdout_bytes':fixed_baseline,'frozen_policy_holdout_bytes':deployed,
   'frozen_policy_gain_fraction':(fixed_baseline-deployed)/fixed_baseline,
   'hindsight_best_baseline':best_base_held,'hindsight_best_baseline_bytes':oracle_base,
   'gain_vs_hindsight_baseline_fraction':(oracle_base-deployed)/oracle_base,
   'hindsight_best_dictionary':best_dict_held,
   'hindsight_best_dictionary_bytes':scores[('holdout',best_dict_held)],
   'exact_ok':True})
 except Exception as e:
  errors.append({'dataset':dataset,'stage':'evaluation','error':repr(e)})
 print(dataset,kind,':',decisions[-1]['frozen_choice'] if decisions and decisions[-1]['dataset']==dataset else 'FAILED')
def save_csv(name,rows):
 if not rows:return
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
save_csv('method_measurements.csv',records)
save_csv('dictionary_attribution.csv',attribution)
save_csv('heldout_decisions.csv',decisions)
(OUT/'errors.json').write_text(json.dumps(errors,indent=2))


# Explicit stop/go rule: at least two independent real series with >2% frozen
# holdout improvement, and no serious regression versus its frozen baseline.
# This heuristic does not establish novelty or statistical significance.
real=[d for d in decisions if d['kind']=='REAL_UCI']
positive=[d for d in real if d['frozen_policy_gain_fraction']>0.02]
regressed=[d for d in real if d['frozen_policy_gain_fraction'] < -0.02]
status='INCOMPLETE_NO_REAL_DATA' if not real else ('CONTINUE_INVESTIGATING' if len(positive)>=2 and not regressed else 'STOP_OR_REDESIGN_DICTIONARY')
summary={'decision':status,'real_streams':len(real),'real_streams_positive_gt_2pct':len(positive),
 'real_streams_regressed_gt_2pct':len(regressed),'tested_methods':METHODS,
 'selection_rule':'25% chronological calibration; dictionary only if >=2% better vs best generic calibration method',
 'note':'ALP not included here. Hindsight best baseline diagnostics are not deployable frozen choices.',
 'real_download_errors':[p for p in provenance if p.get('kind')=='DOWNLOAD_FAILED']}
(OUT/'decision.json').write_text(json.dumps(summary,indent=2))
(OUT/'protocol.json').write_text(json.dumps({'cal_fraction':CAL_FRACTION,'min_cal_gain':MIN_CAL_GAIN,'min_values':MIN_VALUES,
 'dataset_limit_values':MAX_VALUES,'block_values':BLOCK_VALUES,'methods':METHODS,'split':'chronological, no overlap',
 'benchmark_note':'single-run timings descriptive only; no significance claims'},indent=2))
print(json.dumps(summary,indent=2))


# Archive EVERYTHING, even if external UCI downloads were unavailable.
(OUT/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),
 'zstandard_available':bool(zstd),'test_mode':TEST_MODE},indent=2))
files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='sha256_manifest.json')
(OUT/'sha256_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
archive=OUT.parent/'SGX_NB25_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(OUT.iterdir()):
  if p.is_file():z.write(p,arcname=p.name)
print('Results archive:',archive,'bytes:',archive.stat().st_size)
try:
 from google.colab import files as colab_files
 colab_files.download(str(archive))
except ImportError:
 print('Not running in Google Colab; archive saved on disk.')

