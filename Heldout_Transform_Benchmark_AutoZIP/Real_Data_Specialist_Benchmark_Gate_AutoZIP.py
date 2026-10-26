#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB116 — Real-data, Bit-Exact Benchmark Gate

**No old result ZIP and no manual file uploads required.** The notebook downloads two independently sourced real datasets from the UCI repository.

**Predeclared question:** Does the *unchanged* NB115 two-parent codec (block size 1,024; Zstandard level 6) outperform the identically framed one-parent codec and competitive lossless baselines on real numeric float64 arrays? All results are byte-accounted and round-trip checked.

**Important boundaries:** The target is the canonical IEEE-754 float64 matrix obtained by parsing publicly available CSV files, **not the original CSV text**. This experiment is a generalization and competitiveness test, **not a proof of novelty**. zfp and fpzip are attempted as specialist baselines; if either cannot install or fails exact reconstruction, that failure is explicitly recorded. ALP, Corra, Virtual, Chimp, and Gorilla are **not evaluated**, so no state-of-the-art or publication claim can be made.
"""

"""
### 1. Imports and dependency installation — Colab
"""

import sys,subprocess,importlib,io,os,json,struct,csv,time,hashlib,zipfile,lzma,bz2,urllib.request,platform,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from google.colab import files

def ensure(module, package):
 try:return importlib.import_module(module),None
 except ImportError:
  try:
   subprocess.check_call([sys.executable,'-m','pip','-q','install',package]);return importlib.import_module(module),None
  except Exception as e:return None,repr(e)
zstd,ze=ensure('zstandard','zstandard')
if zstd is None:raise RuntimeError('Zstandard is required for the matched comparison: '+str(ze))
zfpy,zfe=ensure('zfpy','zfpy')
fpzip,fpe=ensure('fpzip','fpzip')
print('Optional specialists: zfp',zfpy is not None,'fpzip',fpzip is not None)
print('NumPy',np.__version__,'pandas',pd.__version__,'zstandard',getattr(zstd,'__version__','unknown'))
np.seterr(over='ignore')


"""
### 2. Exact NB115 reversible encoder/decoder — no algorithmic changes
"""

MAGIC=b'SGX11501';C=zstd.ZstdCompressor(level=6);D=zstd.ZstdDecompressor()
# mode 0 raw, 1 XOR 1-parent, 2 subtract 1-parent, 3 subtract two parents (mod 2**64)
# 7-byte descriptor: mode:u8, source1:u8, source2:u8, compressed size:u32
# all transforms are reversible for arbitrary IEEE-754 bit patterns

def pack_block(w,enable_two=True):
 cols,rows=w.shape;out=bytearray(MAGIC+struct.pack('<HI',cols,rows))
 for target in range(cols):
  t=w[target];options=[]
  def choice(mode,s1,s2,r):
   comp=C.compress(r.astype('<u8',copy=False).tobytes())
   options.append((len(comp)+7,mode,s1,s2,comp))
  choice(0,255,255,t)
  for s in range(target):
   choice(1,s,255,np.bitwise_xor(t,w[s]))
   choice(2,s,255,t-w[s])
  if enable_two:
   for s1 in range(target):
    for s2 in range(s1+1,target):
     choice(3,s1,s2,t-w[s1]-w[s2])
  _,mode,s1,s2,comp=min(options,key=lambda x:(x[0],x[1],x[2],x[3]))
  out.extend(struct.pack('<BBBI',mode,s1,s2,len(comp)));out.extend(comp)
 return bytes(out)

def unpack_block(blob):
 view=memoryview(blob)
 if len(view)<14 or view[:8].tobytes()!=MAGIC:raise ValueError('Bad block header')
 nc,nrows=struct.unpack_from('<HI',view,8);pos=14;out=[]
 if not 1<=nc<=255:raise ValueError('Invalid dimensions')
 for target in range(nc):
  if pos+7>len(view):raise ValueError('Truncated descriptor')
  mode,s1,s2,length=struct.unpack_from('<BBBI',view,pos);pos+=7
  if pos+length>len(view):raise ValueError('Truncated payload')
  raw=D.decompress(view[pos:pos+length]);pos+=length
  if len(raw)!=8*nrows:raise ValueError('Wrong decoded length')
  r=np.frombuffer(raw,dtype='<u8').copy()
  if mode==0 and s1==255 and s2==255:t=r
  elif mode in (1,2) and s1<target and s2==255:
   t=np.bitwise_xor(r,out[s1]) if mode==1 else r+out[s1]
  elif mode==3 and s1<s2<target:t=r+out[s1]+out[s2]
  else:raise ValueError('Illegal parent descriptor')
  out.append(t)
 if pos!=len(view):raise ValueError('Trailing bytes')
 return np.stack(out)

def encode(w,enable_two=True,block_rows=1024):
 chunks=[pack_block(w[:,i:i+block_rows],enable_two) for i in range(0,w.shape[1],block_rows)]
 return b'SGX115DS'+struct.pack('<II',w.shape[1],len(chunks))+b''.join(struct.pack('<I',len(x))+x for x in chunks)
def decode(blob):
 v=memoryview(blob)
 if len(v)<16 or v[:8].tobytes()!=b'SGX115DS':raise ValueError('Bad dataset magic')
 rows,n=struct.unpack_from('<II',v,8);pos=16;blocks=[]
 for i in range(n):
  if pos+4>len(v):raise ValueError('Missing block length')
  size=struct.unpack_from('<I',v,pos)[0];pos+=4
  if pos+size>len(v):raise ValueError('Missing block body')
  block=unpack_block(v[pos:pos+size]);pos+=size
  if blocks and block.shape[0]!=blocks[0].shape[0]:raise ValueError('Changed column count')
  blocks.append(block)
 if pos!=len(v):raise ValueError('Trailing dataset bytes')
 out=np.concatenate(blocks,axis=1)
 if out.shape[1]!=rows:raise ValueError('Wrong number of rows')
 return out


"""
### 3. Source download, provenance, parsing, and independent real datasets
"""

URLS={
 'bike':'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip',
 'beijing':'https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip'
}
BIKE_COLS=['casual','registered','cnt','temp','atemp','hum','windspeed','hr','weekday']
BEIJING_COLS=['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP']

def get_source(url):
 req=urllib.request.Request(url,headers={'User-Agent':'ShiftGuard-X-research-benchmark/1.0'})
 try:
  with urllib.request.urlopen(req,timeout=120) as r:data=r.read()
 except Exception as ex:
  raise RuntimeError('Official UCI download failed. Use Runtime > Run all again once internet is available. No synthetic fallback is permitted. URL='+url) from ex
 if not zipfile.is_zipfile(io.BytesIO(data)):raise RuntimeError('Unexpected non-ZIP response from official source: '+url)
 return data

def find_csv_zip(data, fragment):
 queue=[('uci_outer',data)]
 while queue:
  label,content=queue.pop(0)
  with zipfile.ZipFile(io.BytesIO(content)) as z:
   matches=[name for name in z.namelist() if name.lower().endswith('.csv') and fragment.lower() in name.lower()]
   if matches:return z.read(sorted(matches)[0]), label+' / '+sorted(matches)[0]
   for name in z.namelist():
    if name.lower().endswith('.zip'):queue.append((label+' / '+name,z.read(name)))
 raise RuntimeError('Expected CSV not found in source ZIP. Search term: '+fragment)

def make_data(source_key,fragment,cols):
 data=get_source(URLS[source_key]);csv_bytes,path=find_csv_zip(data,fragment)
 frame=pd.read_csv(io.BytesIO(csv_bytes),low_memory=False)
 missing=[name for name in cols if name not in frame.columns]
 if missing:raise RuntimeError(f'Unexpected dataset schema; missing {missing}, has {list(frame.columns)}')
 # Counts and decimal CSV numbers are converted to a canonical numpy.float64 matrix.
 # No rows or missing values are dropped; all NaNs are represented and encoded as IEEE words.
 num=frame[cols].apply(pd.to_numeric,errors='coerce').to_numpy(dtype='<f8',copy=True)
 words=np.ascontiguousarray(num.T).view('<u8')
 assert words.shape==(len(cols),len(frame))
 source={ 'dataset':source_key,'url':URLS[source_key], 'download_sha256':hashlib.sha256(data).hexdigest(),
          'selected_csv':path,'selected_csv_sha256':hashlib.sha256(csv_bytes).hexdigest(),
          'matrix_sha256':hashlib.sha256(words.tobytes()).hexdigest(),
          'columns':cols,'rows':len(frame),'nan_cells':int(np.isnan(num).sum()),
          'dtype':'IEEE754 float64, little-endian canonical parse',
          'scope_note':'exact reconstruction of parsed numeric array, not of CSV source text'}
 return words,source

DATA=[]
for name,fragment,cols in [('bike','hour.csv',BIKE_COLS),('beijing','Aotizhongxin',BEIJING_COLS)]:
 matrix,provenance=make_data(name,fragment,cols)
 DATA.append((name,matrix,provenance))
 print(name,'shape',matrix.shape,'missing numeric cells',provenance['nan_cells'],'SHA',provenance['matrix_sha256'])
assert len(DATA)==2


"""
### 4. Shared schema wrapper, reversible generic compressor baselines, and round-trip checks
"""

# Identical schema is embedded in EVERY scored file. Codecs are compared on same 9xN float64 bit matrix.
# Header names and dtypes matter, even if the specialized codec already knows shape.
def schema_prefix(prov):
 b=json.dumps({'dataset':prov['dataset'],'columns':prov['columns'],'dtype':'<f8',
               'cols':len(prov['columns']),'rows':prov['rows']},sort_keys=True,separators=(',',':')).encode('utf-8')
 return b'SGX116SC'+struct.pack('<I',len(b))+b

def generic(method,w):
 nc,nr=w.shape;raw=np.ascontiguousarray(w,dtype='<u8').tobytes()
 hdr=struct.pack('<8sII',b'SGX11601',nc,nr)
 if method=='zstd_matrix':return hdr+C.compress(raw)
 if method=='zstd_row_major':return hdr+C.compress(np.ascontiguousarray(w.T,dtype='<u8').tobytes())
 if method=='zstd_byte_shuffle':
  arr=np.frombuffer(raw,dtype=np.uint8).reshape(-1,8)
  return hdr+C.compress(arr.T.copy().tobytes())
 if method=='zstd_columnwise':
  pieces=[C.compress(np.ascontiguousarray(w[j],dtype='<u8').tobytes()) for j in range(nc)]
  return hdr+b''.join(struct.pack('<I',len(p)) for p in pieces)+b''.join(pieces)
 if method=='zstd_temporal_xor':
  x=w.copy();x[:,1:]=np.bitwise_xor(w[:,1:],w[:,:-1]);return hdr+C.compress(x.tobytes())
 if method=='lzma_matrix':return hdr+lzma.compress(raw,preset=6)
 if method=='bz2_matrix':return hdr+bz2.compress(raw,compresslevel=9)
 raise ValueError(method)

def generic_decode(method,payload):
 assert payload[:8]==b'SGX11601','Unexpected generic frame header'
 nc,nr=struct.unpack_from('<II',payload,8);body=payload[16:]
 if method=='lzma_matrix':raw=lzma.decompress(body)
 elif method=='bz2_matrix':raw=bz2.decompress(body)
 elif method=='zstd_columnwise':
  cuts=struct.unpack_from('<'+'I'*nc,body,0);pos=nc*4;decoded=[]
  for size in cuts:
   decoded.append(D.decompress(body[pos:pos+size]));pos+=size
  assert pos==len(body)
  raw=b''.join(decoded)
 elif method=='zstd_row_major':
  raw=np.frombuffer(D.decompress(body),dtype='<u8').reshape(nr,nc).T.copy().tobytes()
 elif method=='zstd_byte_shuffle':
  x=np.frombuffer(D.decompress(body),dtype=np.uint8)
  raw=x.reshape(8,-1).T.copy().tobytes()
 elif method=='zstd_temporal_xor':
  z=np.frombuffer(D.decompress(body),dtype='<u8').copy().reshape(nc,nr)
  raw=np.bitwise_xor.accumulate(z,axis=1).tobytes()
 elif method=='zstd_matrix':raw=D.decompress(body)
 else:raise ValueError(method)
 assert len(raw)==nc*nr*8
 return np.frombuffer(raw,dtype='<u8').reshape(nc,nr).copy()

def wrap(payload,prov):return schema_prefix(prov)+payload

def exact_float_from_words(w):
 # rows x columns, contiguous float64 without arithmetic transformations
 return np.ascontiguousarray(w.T,dtype='<u8').view('<f8')

GENERIC=['zstd_matrix','zstd_row_major','zstd_byte_shuffle','zstd_columnwise',
         'zstd_temporal_xor','lzma_matrix','bz2_matrix']
print('NB115 codec and seven generic comparators defined')


"""
### 5. Adversarial IEEE-754 regression and exact full-file benchmark
"""

rng=np.random.default_rng(1162026)
ADV=rng.integers(0,2**64,size=(9,137),dtype=np.uint64)
ADV[:,0:9]=np.array([0,0x8000000000000000,1,0x8000000000000001,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000001234,0x7ff0000000001234,0xfff8000000007777],dtype=np.uint64)
for two in [False,True]:assert np.array_equal(decode(encode(ADV,two,128)),ADV)
for method in GENERIC:
 assert np.array_equal(generic_decode(method,generic(method,ADV)),ADV),method
print('All NB115 and generic codecs pass arbitrary special IEEE bit-pattern checks')

OUT=Path('/content/SGX_NB116_outputs');OUT.mkdir(parents=True,exist_ok=True)
RESULTS=[];PROVENANCE={};FAILURES={};PAYLOADS={}
for name,w,prov in DATA:
 schema=schema_prefix(prov);PROVENANCE[name]=prov
 def add(method,payload,recon,available=True,error=''):
  full=wrap(payload,prov);ok=np.array_equal(w,recon)
  RESULTS.append({'dataset':name,'method':method,'bytes':len(full),'rows':w.shape[1],
                  'columns':w.shape[0],'schema_bytes':len(schema),'bit_exact':bool(ok),
                  'status':'PASS' if ok else 'BITWISE_MISMATCH','error':error})
  if ok:PAYLOADS[(name,method)]=full
  print(' ',method, len(full),'PASS' if ok else 'BITWISE_MISMATCH')

 print('Dataset:',name)
 for method,two in [('sgx115_one_parent',False),('sgx115_two_parent',True)]:
  t0=time.perf_counter();payload=encode(w,two,1024)
  add(method,payload,decode(payload));print('    elapsed_sec',round(time.perf_counter()-t0,2))
 for method in GENERIC:
  payload=generic(method,w);add(method,payload,generic_decode(method,payload))
 floats=exact_float_from_words(w)
 for method,mod,err in [('zfp_reversible',zfpy,zfe),('fpzip_lossless',fpzip,fpe)]:
  if mod is None:
   RESULTS.append({'dataset':name,'method':method,'bytes':None,'rows':w.shape[1],
                   'columns':w.shape[0],'schema_bytes':len(schema),'bit_exact':False,
                   'status':'NOT_AVAILABLE','error':str(err)[:300]})
   continue
  try:
   if method=='zfp_reversible':
    payload=mod.compress_numpy(floats) # defaults to reversible mode; header on
    restored=mod.decompress_numpy(payload)
   else:
    payload=mod.compress(floats,precision=0,order='C')
    restored=mod.decompress(payload,order='C')
   if restored.shape!=floats.shape:raise RuntimeError('Specialist altered array shape')
   # Bitwise check (not np.array_equal on floating values: NaN payload and negative zero matter)
   rebuilt=np.ascontiguousarray(restored,dtype='<f8').view('<u8').T.copy()
   add(method,payload,rebuilt)
  except Exception as ex:
   RESULTS.append({'dataset':name,'method':method,'bytes':None,'rows':w.shape[1],
                   'columns':w.shape[0],'schema_bytes':len(schema),'bit_exact':False,
                   'status':'ERROR','error':repr(ex)[:300]})

 # Save only the two SGX codecs and the strongest VALID non-SGX baseline, not all large files.
 for method in ['sgx115_one_parent','sgx115_two_parent']:
  (OUT/f'{name}_{method}.bin').write_bytes(PAYLOADS[(name,method)])
 winners=sorted([(len(v),meth) for (nm,meth),v in PAYLOADS.items() if nm==name and not meth.startswith('sgx115_')])
 if winners:
  best_method=winners[0][1]
  (OUT/f'{name}_best_baseline_{best_method}.bin').write_bytes(PAYLOADS[(name,best_method)])

with (OUT/'comparison.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(RESULTS[0]));writer.writeheader();writer.writerows(RESULTS)
(OUT/'provenance.json').write_text(json.dumps(PROVENANCE,indent=2))
print('Completed',len(RESULTS),'attempted methods')


"""
### 6. Conservative, predeclared research decision and downloadable evidence ZIP
"""

import pandas as pd
results_df=pd.DataFrame(RESULTS)
summary={};success=[]
for name,_,_ in DATA:
 r=results_df[(results_df.dataset==name)&(results_df.status=='PASS')]
 b1=int(r.loc[r.method=='sgx115_one_parent','bytes'].iloc[0])
 b2=int(r.loc[r.method=='sgx115_two_parent','bytes'].iloc[0])
 competition=r[~r.method.str.startswith('sgx115_')].sort_values('bytes')
 if competition.empty:raise RuntimeError('No valid non-SGX comparator; abort research gate')
 best=competition.iloc[0]
 winner={'two_parent_bytes':b2,'one_parent_bytes':b1,'best_valid_baseline':str(best['method']),
         'best_valid_baseline_bytes':int(best['bytes']),
         'two_vs_one_pct':round(100*(1-b2/b1),4),
         'two_vs_best_baseline_pct':round(100*(1-b2/int(best['bytes'])),4),
         'valid_competitors':competition.method.tolist(),
         'missing_specialist_benchmarks':results_df[(results_df.dataset==name)&(results_df.status!='PASS')].method.tolist()}
 summary[name]=winner
 success.append((b2<b1) and (b2<int(best['bytes'])))
 print(name,json.dumps(winner,indent=2))
decision={'status':'COMPLETE','scope':'float64 matrices canonically parsed from real UCI CSV datasets',
          'datasets':summary,'two_parent_strictly_beats_one_parent_and_best_valid_baseline_in_both':bool(all(success)),
          'novelty_proven':False,'state_of_art_claim_allowed':False,
          'prior_art_gap':'ALP, Corra, Virtual, Chimp, Gorilla, and implementation-matched specialist comparisons are still needed.',
          'method_known':'NB115 tests known multi-parent modular prediction and adaptive choice; no original rule theorem proven.',
          'caution':'Even success on these 2 sources is not algorithmic novelty; failure is sufficient to reject blanket superiority claims.',
          'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'zstandard':getattr(zstd,'__version__','unknown'),
                      'zfpy':getattr(zfpy,'__version__',None) if zfpy else None,'fpzip':getattr(fpzip,'__version__',None) if fpzip else None}}
(OUT/'decision.json').write_text(json.dumps(decision,indent=2))
(OUT/'limitations.txt').write_text("""NB116 is not a novelty demonstration. Two-parent modular residuals and adaptive transform selection have precedent.
All sizes include the identical schema prefix (column order, float64 dtype and shape). Native zfp/fpzip headers retained.
Original CSV text, column types, decimal formatting, categorical strings, and missing tokens cannot be reconstructed; only the parsed float64 matrix.
No ALP/Corra/Virtual/Chimp/Gorilla baseline in this notebook. Optional zfp/fpzip not available or not bit-exact is recorded as such.
Per-block transform selection is performed at encoding time; its complete headers, descriptions and streams count in the SGX file size.
Original methods' parameter and format differences limit interpretation. No training/learned selection, no tuning on the datasets.
""")
ARCH=Path('/content/SGX_NB116_results.zip')
with zipfile.ZipFile(ARCH,'w',zipfile.ZIP_DEFLATED) as archive:
 for f in sorted(OUT.rglob('*')):
  if f.is_file():archive.write(f,arcname=f.relative_to(OUT))
with zipfile.ZipFile(ARCH) as archive:assert archive.testzip() is None
print('Verified:',ARCH,ARCH.stat().st_size,'bytes')
files.download(str(ARCH))

