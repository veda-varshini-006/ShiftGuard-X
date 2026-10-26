#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX-NB24 — Structure-Aware Numerical Symbol Coding

**Goal:** Test whether a small exact-value dictionary and zero-error run coding outperform a matched baseline on genuine numerical streams. **This is an exploratory candidate, not a claimed novel codec.**

- No paid API or GPU. Run **Runtime → Run all** in Colab.
- Exact bitwise preservation of the input IEEE-754 `float64` arrays (including signed zero and NaN payloads in synthetic tests).
- Real UCI downloads are attempted. If they fail, results are explicitly marked synthetic/diagnostic, never presented as real.
- Baselines: raw, zlib, lzma, optional zstd, plus delta+zstd where available. Report size and measured encode/decode time, always including headers.
- Outputs: `SGX_NB24_results.zip` and machine-readable CSV/JSON with per-dataset evidence.
"""

import os,sys,csv,json,time,hashlib,struct,zlib,lzma,zipfile,io,urllib.request,collections,platform,statistics,traceback,random
from pathlib import Path
OUT=Path('/content/SGX_NB24_outputs') if Path('/content').exists() else Path('SGX_NB24_outputs')
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


# Mandatory correctness and corruption tests before benchmarking
special=[0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,0x7ff0000000000001,0x0000000000000001,0x7fefffffffffffff]
raw_special=b''.join(struct.pack('<Q',v) for v in special*500)
raw_patterns=b''.join(struct.pack('<Q',i%17) for i in range(5000))
raw_random=random.Random(41).randbytes(16*1024)
raw_random=raw_random[:len(raw_random)//8*8]
checks=[]
for label,raw in [('special',raw_special),('pattern',raw_patterns),('random_bits',raw_random),('empty',b'')]:
 for method in METHODS:
  enc=encode_stream(raw,method);dec=decode_stream(enc)
  assert dec==raw,(label,method)
  checks.append({'input':label,'method':method,'exact_ok':True,'nbytes':len(raw),'coded_bytes':len(enc)})
for bad in (b'',b'not_a_stream',encode_stream(raw_patterns,'dict_run_zlib')[:-3]):
 try:decode_stream(bad);raise AssertionError('corrupt stream accepted')
 except ValueError:pass
 except (zlib.error,lzma.LZMAError):pass
with (OUT/'correctness.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
print('Round trips passed:',len(checks))


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


# Benchmarks use actual encoded stream byte count (headers included). Save after each dataset.
bench=[];fail=[]
fields=['dataset','kind','method','n_values','input_bytes','coded_bytes','bits_per_value','encode_seconds','decode_seconds','encode_mib_s','decode_mib_s','exact_ok']
for ds,raw,kind in datasets:
 for method in METHODS:
  try:
   t=time.perf_counter();encoded=encode_stream(raw,method);et=time.perf_counter()-t
   t=time.perf_counter();decoded=decode_stream(encoded);dt=time.perf_counter()-t
   if decoded!=raw:raise AssertionError('decoder mismatch')
   bench.append(dict(dataset=ds,kind=kind,method=method,n_values=len(raw)//8,input_bytes=len(raw),coded_bytes=len(encoded),bits_per_value=8*len(encoded)/max(1,len(raw)//8),encode_seconds=et,decode_seconds=dt,encode_mib_s=len(raw)/1048576/max(et,1e-9),decode_mib_s=len(raw)/1048576/max(dt,1e-9),exact_ok=True))
  except Exception as exc:fail.append({'dataset':ds,'method':method,'error':repr(exc)})
 with (OUT/'benchmarks.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(bench)
 with (OUT/'errors.json').open('w') as f:json.dump(fail,f,indent=2)
 print('Completed',ds,'records',len(bench))
assert bench,'No benchmark records'


# Evidence-led summary. Compare to strongest fixed measured reference; no hindsight-selected method claimed deployable.
summary=[]
for ds in sorted(set(x['dataset'] for x in bench)):
 group=[x for x in bench if x['dataset']==ds]
 baseline=min((x for x in group if x['method'] in ('zlib','lzma','zstd','raw')),key=lambda x:x['coded_bytes'])
 candidate=min((x for x in group if x['method'].startswith('dict')),key=lambda x:x['coded_bytes'])
 saving=100*(baseline['coded_bytes']-candidate['coded_bytes'])/baseline['coded_bytes']
 summary.append({'dataset':ds,'kind':baseline['kind'],'baseline_method':baseline['method'],'baseline_bytes':baseline['coded_bytes'],'best_candidate_method_hindsight':candidate['method'],'candidate_bytes':candidate['coded_bytes'],'candidate_size_saving_pct':round(saving,4),'candidate_encode_mib_s':round(candidate['encode_mib_s'],4),'baseline_encode_mib_s':round(baseline['encode_mib_s'],4)})
with (OUT/'gap_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
real=[x for x in summary if x['kind']=='REAL_UCI'];improved=[x for x in real if x['candidate_size_saving_pct']>0]
assessment={'status':'EXPLORATORY_ONLY','real_datasets_count':len(real),'real_datasets_candidate_smaller_count':len(improved),'research_novelty':'NOT_VERIFIED','specialized_ALP_same_protocol':'NOT_RUN','explanation':'Dictionary/run coding has extensive prior art; positive size differences do not establish originality or top-journal readiness.'}
with (OUT/'decision.json').open('w') as f:json.dump(assessment,f,indent=2)
print(json.dumps(assessment,indent=2));print(*summary,sep='\n')


# Always archive every generated record. Colab download is optional outside Colab.
import zipfile
with (OUT/'environment.json').open('w') as f:json.dump({'python':sys.version,'platform':platform.platform(),'methods':METHODS,'max_values':MAX_VALUES,'block_values':BLOCK_VALUES,'test_mode':TEST_MODE},f,indent=2)
files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='sha256_manifest.json')
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
with (OUT/'sha256_manifest.json').open('w') as f:json.dump(manifest,f,indent=2)
archive=OUT.parent/'SGX_NB24_results.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in OUT.iterdir():
  if p.is_file():z.write(p,p.name)
print('Created',archive, 'bytes',archive.stat().st_size)
try:
 from google.colab import files
 files.download(str(archive))
except ImportError:pass

