#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX-NB26 — Numerical Coding Architecture & Novelty Gate

**Purpose:** determine whether a bounded-block numerical bitstream candidate merits further research after NB25's dictionary approach failed to generalize. This is an **exploratory architecture audit**, not evidence of algorithmic novelty. Uses original UCI input values, byte-exact recovery, calibration-only choice, strictly held-out testing, size including header, and conventional codec comparisons. **Official ALP is not rerun**; prior ALP BPV is for context only and is not treated as a matched claim.

Colab: Runtime → Run all. The final cell writes and offers `SGX_NB26_results.zip` even when external dataset retrieval fails. No paid APIs or GPU required.
"""

import csv,io,json,hashlib,urllib.request,zipfile,struct,collections,math,statistics,random,time,platform,traceback,zlib,lzma,sys
from pathlib import Path
OUT=Path('/content/SGX_NB26_outputs') if Path('/content').is_dir() else Path('/mnt/data/SGX_NB26_outputs')
OUT.mkdir(parents=True,exist_ok=True)
MAX_VALUES=16000; BLOCK=1024; CAL_FRACTION=.25
try:
 import zstandard as zstd
except ImportError:
 zstd=None
PROTOCOL={'max_values_per_series':MAX_VALUES,'block_values':BLOCK,'cal_fraction':CAL_FRACTION,'heldout':'chronological later 75%, zero tuning on holdout','bitwise_comparison':'exact bytes including signed zero and NaN payloads','zstd_available':bool(zstd),'interpretation':'exploratory only; dataset families previously examined'}
(OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
print('Runtime:',sys.version.split()[0],'Zstd:',bool(zstd))


# Public dataset loader. Existing UCI samples have been used in previous SGX experiments;
# these ARE NOT independent confirmatory datasets. Do not substitute synthetic data on download failure.
def retrieve(url):
 req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 SGX-reproducibility'})
 with urllib.request.urlopen(req,timeout=60) as f:return f.read()
def load_real():
 specs=[('UCI_Air_Quality','https://archive.ics.uci.edu/static/public/360/air+quality.zip',['T','RH','CO(GT)']),('UCI_Household_Power','https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip',['Global_active_power','Voltage','Global_intensity'])]
 inputs=[];prov=[];errs=[]
 for ds,url,fields in specs:
  try:
   binary=retrieve(url)
   with zipfile.ZipFile(io.BytesIO(binary)) as z:
    files=[x for x in z.namelist() if x.lower().endswith(('.csv','.txt'))]
    if not files:raise ValueError('No text data found in archive')
    member=next((x for x in files if ('airqualityuci' if ds=='UCI_Air_Quality' else 'household_power_consumption') in x.lower()),files[0]);body=z.read(member)
   reader=csv.DictReader(io.StringIO(body.decode('utf-8-sig',errors='replace')),delimiter=';');cols={k:[] for k in fields}
   for row in reader:
    if all(len(v)>=MAX_VALUES for v in cols.values()):break
    for field in fields:
     if len(cols[field])>=MAX_VALUES:continue
     s=(row.get(field) or '').strip().replace(',','.')
     if s in ('','?'):continue
     try:v=float(s)
     except (TypeError,ValueError):continue
     if not math.isfinite(v) or (ds=='UCI_Air_Quality' and v==-200):continue
     cols[field].append(v)
   for field,values in cols.items():
    if len(values)<400:continue
    raw=struct.pack('<'+'d'*len(values),*values);name=ds+'__'+field
    inputs.append((name,raw,'REAL_UCI'))
    prov.append({'dataset':name,'source_url':url,'archive_member':member,'archive_sha256':hashlib.sha256(binary).hexdigest(),'input_sha256':hashlib.sha256(raw).hexdigest(),'count':len(values),'kind':'REAL_UCI'})
  except Exception as exc:errs.append({'dataset':ds,'stage':'download','error':repr(exc),'url':url})
 return inputs,prov,errs
real,prov,errors=load_real()
rng=random.Random(260026)
synthetic=[('SYNTHETIC_constant',[3.25]*8192),('SYNTHETIC_lowcard',[float(rng.randrange(12)) for _ in range(8192)]),('SYNTHETIC_random',[rng.random() for _ in range(8192)])]
inputs=real+[(name,struct.pack('<'+'d'*len(x),*x),'SYNTHETIC') for name,x in synthetic]
for name,raw,kind in inputs:
 if kind=='SYNTHETIC':prov.append({'dataset':name,'kind':kind,'count':len(raw)//8,'input_sha256':hashlib.sha256(raw).hexdigest()})
(OUT/'provenance.json').write_text(json.dumps(prov,indent=2));(OUT/'errors.json').write_text(json.dumps(errors,indent=2))
print([(name,len(raw)//8,k) for name,raw,k in inputs]);print('Download errors:',errors)


# Candidate representation: independent XOR residual blocks + per-residual byte width
# 0-width residual has a 1-byte tag. This is a diagnostic architecture assembled from
# established XOR prediction and variable-byte coding; NOT claimed to be novel.
MAGIC=b'SGX26XR1'; MASK=(1<<64)-1
def wordlist(raw):
 if len(raw)%8:raise ValueError('input must be aligned to float64')
 return [x[0] for x in struct.iter_unpack('<Q',raw)]
def xr_encode(raw):
 vals=wordlist(raw);out=bytearray(MAGIC);out.extend(struct.pack('<I',len(vals)))
 for start in range(0,len(vals),BLOCK):
  v=vals[start:start+BLOCK];out.extend(struct.pack('<H',len(v)))
  prev=0
  for x in v:
   residual=x^prev;prev=x
   width=(residual.bit_length()+7)//8;out.append(width)
   if width:out.extend(residual.to_bytes(width,'little'))
 return bytes(out)
def xr_decode(data):
 if len(data)<12 or data[:8]!=MAGIC:raise ValueError('bad header')
 n=struct.unpack_from('<I',data,8)[0];pos=12;vals=[]
 while len(vals)<n:
  if pos+2>len(data):raise ValueError('truncated block')
  count=struct.unpack_from('<H',data,pos)[0];pos+=2
  if not (0<count<=BLOCK and len(vals)+count<=n):raise ValueError('invalid block count')
  prev=0
  for _ in range(count):
   if pos>=len(data):raise ValueError('missing width')
   width=data[pos];pos+=1
   if width>8 or pos+width>len(data):raise ValueError('invalid width')
   resid=int.from_bytes(data[pos:pos+width],'little');pos+=width
   value=prev^resid;prev=value;vals.append(value)
 if pos!=len(data):raise ValueError('trailing bytes')
 return b''.join(struct.pack('<Q',v) for v in vals)
# Competitors with complete framing; all recover raw bytes.
def encode(raw,m):
 if m=='raw':return raw
 if m=='zlib':return zlib.compress(raw,6)
 if m=='lzma':return lzma.compress(raw,preset=1)
 if m=='zstd':return zstd.ZstdCompressor(level=3).compress(raw)
 if m=='xor_width':return xr_encode(raw)
 if m=='xor_width_zlib':return zlib.compress(xr_encode(raw),6)
 if m=='xor_width_zstd':return zstd.ZstdCompressor(level=3).compress(xr_encode(raw))
 raise KeyError(m)
def decode(data,m):
 if m=='raw':return data
 if m=='zlib':return zlib.decompress(data)
 if m=='lzma':return lzma.decompress(data)
 if m=='zstd':return zstd.ZstdDecompressor().decompress(data)
 if m=='xor_width':return xr_decode(data)
 if m=='xor_width_zlib':return xr_decode(zlib.decompress(data))
 if m=='xor_width_zstd':return xr_decode(zstd.ZstdDecompressor().decompress(data))
 raise KeyError(m)
METHODS=['raw','zlib','lzma','xor_width','xor_width_zlib']+(['zstd','xor_width_zstd'] if zstd else [])
# Adversarial exact-bit patterns (positive and negative zero, quiet/signalling NaNs, infinities, subnormals)
patterns=[0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,0x7ff0000000000001,0x0000000000000001,0xffffffffffffffff]
edgecases=[]
for name,words in [('special',patterns),('repeat',patterns*64),('alternating',patterns[:2]*2400),('empty',[])]:
 raw=b''.join(struct.pack('<Q',x) for x in words)
 for method in METHODS:
  try:
   enc=encode(raw,method);good=decode(enc,method)==raw
  except Exception as exc:good=False;errors.append({'stage':'adversarial','case':name,'method':method,'error':repr(exc)})
  edgecases.append({'case':name,'method':method,'exact_ok':good})
assert all(x['exact_ok'] for x in edgecases),'Adversarial reconstruction failure'
print('Adversarial exact tests passed:',len(edgecases))


# Predictive diagnostic measurements; avoids implying a novel coder has been shown.
def features(raw):
 w=wordlist(raw);n=len(w);freq=collections.Counter(w)
 xor=[w[i]^w[i-1] for i in range(1,n)]
 width=collections.Counter((x.bit_length()+7)//8 for x in xor)
 probs=[v/max(1,n) for v in freq.values()]
 return {'count':n,'unique_ratio':len(freq)/max(1,n),'adjacent_repeat_fraction':sum(x==0 for x in xor)/max(1,len(xor)), 'mean_xor_width_bytes':sum((x.bit_length()+7)//8 for x in xor)/max(1,len(xor)), 'zero_xor_fraction':width.get(0,0)/max(1,len(xor)), 'empirical_value_entropy_bits':-sum(p*math.log2(p) for p in probs if p>0)}
measurements=[];diagnostics=[];choices=[]
for name,raw,kind in inputs:
 try:
  n=len(raw)//8;cal_n=max(100,int(n*CAL_FRACTION));cal=raw[:8*cal_n];hold=raw[8*cal_n:]
  if len(hold)<8:raise ValueError('no holdout')
  sizes={}
  for split,part in [('calibration',cal),('holdout',hold)]:
   diagnostics.append({'dataset':name,'kind':kind,'split':split,**features(part)})
   for m in METHODS:
    t=time.perf_counter();encoded=encode(part,m);enc_t=time.perf_counter()-t
    t=time.perf_counter();restored=decode(encoded,m);dec_t=time.perf_counter()-t
    if restored!=part:raise AssertionError('exact decode failed: '+name+'/'+m)
    sizes[split,m]=len(encoded)
    measurements.append({'dataset':name,'kind':kind,'split':split,'method':m,'input_bytes':len(part),'coded_bytes':len(encoded),'bpv':8*len(encoded)/(len(part)//8),'encode_seconds':enc_t,'decode_seconds':dec_t,'exact_ok':True})
  baseline_methods=[x for x in METHODS if not x.startswith('xor_')]
  candidate_methods=[x for x in METHODS if x.startswith('xor_')]
  base=min(baseline_methods,key=lambda m:sizes['calibration',m]);cand=min(candidate_methods,key=lambda m:sizes['calibration',m]);
  select=cand if sizes['calibration',cand]<sizes['calibration',base]*0.98 else base
  fixed_base=sizes['holdout',base];held_size=sizes['holdout',select]
  hindsight=min(baseline_methods,key=lambda m:sizes['holdout',m]);hindsight_bytes=sizes['holdout',hindsight]
  choices.append({'dataset':name,'kind':kind,'calibration_baseline':base,'calibration_candidate':cand,'frozen_method':select,'frozen_baseline_holdout_bytes':fixed_base,'frozen_method_holdout_bytes':held_size,'savings_vs_frozen_baseline_fraction':(fixed_base-held_size)/fixed_base,'hindsight_best_baseline':hindsight,'hindsight_best_baseline_bytes':hindsight_bytes,'savings_vs_hindsight_baseline_fraction':(hindsight_bytes-held_size)/hindsight_bytes,'exact_ok':True})
 except Exception as exc:errors.append({'stage':'benchmark','dataset':name,'error':repr(exc),'traceback':traceback.format_exc()[-1800:]})
 print(name,':',choices[-1]['frozen_method'] if choices and choices[-1]['dataset']==name else 'FAILED')
print('Rows:',len(measurements),'Decisions:',len(choices))


# Explicit literature-claim boundary and a pre-stated gate, NOT a novelty certificate.
prior_art=[
 {'idea':'XOR/predictive floating-point coding','examples':'Gorilla (Pelkonen et al., VLDB 2015); Chimp (Li et al., VLDB 2022)','overlap':'XOR with earlier values and variable-length residual representation are established','reference':'https://www.vldb.org/pvldb/vol8/p1816-teller.pdf'},
 {'idea':'Floating-point ALP encoding','examples':'ALP (Afroozeh et al., SIGMOD 2024)','overlap':'specialized numerical encoding, adaptive choices and packing already studied','reference':'https://github.com/cwida/ALP'},
 {'idea':'Dictionary/RLE/block packing','examples':'widely used in columnar storage and compression systems','overlap':'not novel alone; NB25 CO gain does not prove novelty','reference':'https://parquet.apache.org/docs/file-format/data-pages/encodings/'},
 {'idea':'General-purpose compression','examples':'Zstandard, LZMA, zlib','overlap':'entropy backend is prior art','reference':'https://github.com/facebook/zstd'}]
real_choices=[r for r in choices if r['kind']=='REAL_UCI']
real_wins=[r for r in real_choices if r['savings_vs_frozen_baseline_fraction']>=.02 and r['savings_vs_hindsight_baseline_fraction']>=.02]
gate={'status':'INCOMPLETE_REAL_DATA' if len(real_choices)<6 else 'STOP_OR_RETHINK' if len(real_wins)<2 else 'FOLLOWUP_REQUIRED',
      'real_streams':len(real_choices),'real_stream_wins_2percent_vs_both':len(real_wins),
      'minimum_criterion':'at least 2/6 real streams >=2% heldout reduction versus both frozen and hindsight baselines; no exact failures',
      'novelty_established':False,'independent_confirmation':False,'official_ALP_matched':False,
      'interpretation':'even if size criterion passes, this is not a novel published coding mechanism; independent datasets and matched ALP/Chimp/Gorilla still required'}
(OUT/'prior_art_audit.json').write_text(json.dumps(prior_art,indent=2))
(OUT/'research_gate.json').write_text(json.dumps(gate,indent=2))
print(json.dumps(gate,indent=2))


# Reliable persistence: always export logs/metadata, including incomplete and failed runs.
def save_csv(filename,rows):
 if rows:
  cols=list(dict.fromkeys(k for r in rows for k in r))
  with (OUT/filename).open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader();writer.writerows(rows)
save_csv('measurements.csv',measurements);save_csv('diagnostics.csv',diagnostics);save_csv('heldout_decisions.csv',choices);save_csv('adversarial_correctness.csv',edgecases)
(OUT/'errors.json').write_text(json.dumps(errors,indent=2));(OUT/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'zstandard_version':getattr(zstd,'__version__',None)},indent=2))
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}
(OUT/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2))
archive=OUT.parent/'SGX_NB26_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(OUT.iterdir()):
  if p.is_file():z.write(p,arcname=p.name)
print('Wrote',archive,'bytes:',archive.stat().st_size)
try:
 from google.colab import files
 files.download(str(archive))
except ImportError:
 print('Not running in Colab; results ZIP is saved at',archive)
except Exception as exc:
 print('Auto-download unavailable:',repr(exc),'ZIP still saved at',archive)

