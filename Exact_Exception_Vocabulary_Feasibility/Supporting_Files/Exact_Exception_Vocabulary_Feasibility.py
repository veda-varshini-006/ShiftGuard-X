#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB85 — Exact-Exception Vocabulary Feasibility

**Q1 scientific gate, NOT a new algorithm claim.** Tests whether a shared dictionary of exact IEEE-754 exceptional bit patterns across 512-value blocks offers savings after all metadata and full-file baselines.

Relevant prior art: [ALP / Apache Parquet exception specification](https://parquet.apache.org/docs/file-format/data-pages/alpencoding/), [ALP (SIGMOD 2024)](https://ir.cwi.nl/pub/33334/), [C3](https://github.com/cwida/C3), [Corra](https://arxiv.org/abs/2403.17229). Dictionary coding is itself established.

**Primary threshold:** predefined scale=100, at least 3% aggregate and at least 2 real dataset families each >=3% savings over the strongest **tested** noncandidate method per stream; <=25% regressions, no codec errors. Even success would only justify further investigation.

Tested inputs: scikit-learn bundled public data and UCI Wine Quality data if reachable. All are development/previously used families; synthetic edge cases are reported separately. A cached user-owned NB70 result archive can supply UCI CSV bytes without network. Exact floating-point bit patterns, including signed zero and NaN payloads, are preserved. Missing files/downloads are reported.

Run all cells from the top. The result is downloaded automatically as `SGX_NB85_results.zip`.
"""

import importlib.util, subprocess, sys, os
if importlib.util.find_spec('zstandard') is None and os.path.exists('/content'):
    p = subprocess.run([sys.executable,'-m','pip','install','-q','zstandard'], capture_output=True, text=True, timeout=150)
    print('Optional zstandard install:', 'OK' if p.returncode==0 else 'UNAVAILABLE (continuing)', p.stderr[-400:])


import os,sys,io,csv,json,math,hashlib,struct,time,statistics,zipfile,zlib,lzma,urllib.request,platform,traceback
from pathlib import Path
import numpy as np
from sklearn.datasets import load_wine,load_diabetes,load_breast_cancer,load_digits
try:
 import zstandard as zstd
except ImportError:
 zstd=None
SEED=20261024
BLOCK=512
SCALES={0:1,1:100}
MODE_NAMES={0:'raw_plain',1:'raw_zlib',2:'raw_lzma',3:'global_exception_dictionary',4:'block_exception_dictionary',5:'byte_shuffle_zlib',6:'raw_zstd'}
MODES=[0,1,2,3,4,5]+([6] if zstd else [])
MAGIC=b'EX85'
HEAD=struct.Struct('<4sBBHIII')  # magic / mode / scale-id / blocksize / count / core bytes / side bytes
G_MODES={'zlib6':0,'lzma1':1,'shuffle_zlib6':2,'zstd3':3}
G_MAGIC=b'GB85'

def f64bytes(values): return np.ascontiguousarray(np.asarray(values,dtype='<f8')).tobytes()

def make_int_and_exceptions(raw,scale):
 if len(raw)%8:raise ValueError('input not 8-byte aligned')
 floats=np.frombuffer(raw,dtype='<f8'); word=np.frombuffer(raw,dtype='<u8');n=len(floats)
 with np.errstate(over='ignore',invalid='ignore'):
  possible=np.isfinite(floats)&(np.abs(floats) < (2**31-1)/scale)
  k=np.zeros(n,dtype='<i4')
  if np.any(possible): k[possible]=np.rint(floats[possible]*scale).astype('<i4')
  rec=k.astype('<f8')/scale
  match=possible & (rec.view('<u8')==word)
 k[~match]=0
 pos=np.flatnonzero(~match).astype('<u4')
 exc=word[~match].astype('<u8')
 return k,pos,exc

def _pairbytes(pos,vals):
 pairs=np.empty(len(pos),dtype=[('position','<u4'),('word','<u8')])
 pairs['position']=pos;pairs['word']=vals
 return pairs.tobytes()

def _get_pairs(data):
 if len(data)%12: raise ValueError('bad exception tuple length')
 arr=np.frombuffer(data,dtype=[('position','<u4'),('word','<u8')])
 return arr['position'].copy(),arr['word'].copy()

def _idwidth(k): return 1 if k<=256 else (2 if k<=65536 else 4)
def _dtype_for_width(w): return {1:'<u1',2:'<u2',4:'<u4'}[w]

def side_pack(mode,pos,words,n,blocksz):
 m=len(pos)
 if mode==0: return _pairbytes(pos,words)
 if mode==1: return zlib.compress(_pairbytes(pos,words),6)
 if mode==2: return lzma.compress(_pairbytes(pos,words),preset=1)
 if mode==6:
  if zstd is None:raise ValueError('zstandard unavailable')
  return zstd.ZstdCompressor(level=3).compress(_pairbytes(pos,words))
 if mode==3:
  unique,ids=np.unique(words,return_inverse=True);w=_idwidth(len(unique))
  payload=pos.astype('<u4').tobytes()+ids.astype(_dtype_for_width(w)).tobytes()
  compressed=zlib.compress(payload,6)
  return struct.pack('<IBII',len(unique),w,m,len(compressed))+unique.astype('<u8').tobytes()+compressed
 if mode==4:
  out=bytearray();blocks=(n+blocksz-1)//blocksz
  for b in range(blocks):
   mask=(pos>=b*blocksz)&(pos<(b+1)*blocksz)
   localpos=(pos[mask]-(b*blocksz)).astype('<u2')
   localval=words[mask];unique,ids=np.unique(localval,return_inverse=True);w=_idwidth(len(unique))
   payload=localpos.tobytes()+ids.astype(_dtype_for_width(w)).tobytes()
   zipped=zlib.compress(payload,6)
   out.extend(struct.pack('<HBII',len(unique),w,len(localpos),len(zipped)))
   out.extend(unique.astype('<u8').tobytes());out.extend(zipped)
  return bytes(out)
 if mode==5:
  zpos=zlib.compress(pos.astype('<u4').tobytes(),6)
  shuffled=words.view(np.uint8).reshape(-1,8).T.copy().tobytes() if m else b''
  zval=zlib.compress(shuffled,6)
  return struct.pack('<II',len(zpos),len(zval))+zpos+zval
 raise ValueError('bad side mode')

def side_unpack(mode,data,n,blocksz):
 if mode==0:return _get_pairs(data)
 if mode==1:return _get_pairs(zlib.decompress(data))
 if mode==2:return _get_pairs(lzma.decompress(data))
 if mode==6:
  if zstd is None:raise ValueError('zstandard unavailable')
  return _get_pairs(zstd.ZstdDecompressor().decompress(data))
 if mode==3:
  if len(data)<13:raise ValueError('truncated global dictionary')
  k,w,m,l=struct.unpack('<IBII',data[:13]);off=13
  if w not in (1,2,4) or k>n or m>n:raise ValueError('bad dictionary header')
  if len(data)!=13+8*k+l:raise ValueError('global dictionary size mismatch')
  dictionary=np.frombuffer(data,dtype='<u8',count=k,offset=off);off+=8*k
  d=zlib.decompress(data[off:]);expect=m*(4+w)
  if len(d)!=expect:raise ValueError('dictionary indices/positions wrong size')
  pos=np.frombuffer(d,dtype='<u4',count=m,offset=0).copy()
  ids=np.frombuffer(d,dtype=_dtype_for_width(w),count=m,offset=4*m)
  if np.any(ids>=k):raise ValueError('invalid dictionary index')
  return pos,dictionary[ids].copy()
 if mode==4:
  positions=[];values=[];p=0
  for b in range((n+blocksz-1)//blocksz):
   if len(data)-p<11:raise ValueError('truncated block dictionary header')
   k,w,m,l=struct.unpack('<HBII',data[p:p+11]);p+=11
   if w not in (1,2,4) or k>m or m>blocksz:raise ValueError('invalid local dictionary')
   if len(data)-p<8*k+l:raise ValueError('truncated local payload')
   table=np.frombuffer(data,dtype='<u8',count=k,offset=p).copy();p+=8*k
   payload=zlib.decompress(data[p:p+l]);p+=l
   if len(payload)!=(2+w)*m:raise ValueError('local indices/positions size mismatch')
   local=np.frombuffer(payload,dtype='<u2',count=m).astype('<u4')
   idx=np.frombuffer(payload,dtype=_dtype_for_width(w),count=m,offset=2*m)
   if np.any(local>=blocksz) or np.any(idx>=k):raise ValueError('bad local index')
   positions.append(local+b*blocksz);values.append(table[idx])
  if p!=len(data):raise ValueError('trailing local dictionary bytes')
  return (np.concatenate(positions) if positions else np.array([],dtype='<u4'),np.concatenate(values) if values else np.array([],dtype='<u8'))
 if mode==5:
  if len(data)<8:raise ValueError('shuffled header')
  a,b=struct.unpack('<II',data[:8]);z=data[8:]
  if len(z)!=a+b:raise ValueError('shuffled sections')
  pos=np.frombuffer(zlib.decompress(z[:a]),dtype='<u4').copy()
  e=zlib.decompress(z[a:])
  if len(e)!=8*len(pos):raise ValueError('shuffled value length')
  vals=np.frombuffer(e,dtype=np.uint8).reshape(8,-1).T.copy().tobytes()
  return pos,np.frombuffer(vals,dtype='<u8').copy()
 raise ValueError('bad side mode')

def encode_exception(raw,mode=3,scale_id=1):
 if mode not in MODES or scale_id not in SCALES:raise ValueError('unsupported mode')
 scale=SCALES[scale_id];q,pos,words=make_int_and_exceptions(raw,scale);n=len(q)
 core=bytearray()
 for start in range(0,n,BLOCK):
  c=zlib.compress(q[start:start+BLOCK].astype('<i4').tobytes(),6)
  core.extend(struct.pack('<I',len(c)));core.extend(c)
 side=side_pack(mode,pos,words,n,BLOCK)
 return HEAD.pack(MAGIC,mode,scale_id,BLOCK,n,len(core),len(side))+core+side

def decode_exception(blob):
 if len(blob)<HEAD.size:raise ValueError('short frame')
 magic,mode,sc_id,blocksz,n,cl,sl=HEAD.unpack(blob[:HEAD.size])
 if magic!=MAGIC or blocksz!=BLOCK or sc_id not in SCALES or mode not in MODES:raise ValueError('invalid frame header')
 if len(blob)!=HEAD.size+cl+sl:raise ValueError('frame size mismatch')
 core=blob[HEAD.size:HEAD.size+cl];side=blob[HEAD.size+cl:];p=0;parts=[]
 for start in range(0,n,blocksz):
  if len(core)-p<4:raise ValueError('truncated int core')
  size=struct.unpack('<I',core[p:p+4])[0];p+=4
  if size>len(core)-p:raise ValueError('invalid int section')
  d=zlib.decompress(core[p:p+size]);p+=size
  expect=min(blocksz,n-start)
  if len(d)!=4*expect:raise ValueError('wrong inlier length')
  parts.append(np.frombuffer(d,dtype='<i4'))
 if p!=len(core):raise ValueError('trailing core bytes')
 q=np.concatenate(parts) if parts else np.empty(0,dtype='<i4')
 with np.errstate(over='ignore',invalid='ignore'):
  out=(q.astype('<f8')/SCALES[sc_id]).view('<u8').copy()
 pos,vals=side_unpack(mode,side,n,blocksz)
 if len(pos)!=len(vals) or np.any(pos>=n) or len(np.unique(pos))!=len(pos):raise ValueError('exception position invalid')
 out[pos]=vals
 return out.astype('<u8').tobytes()

def generic_encode(raw,mode):
 if mode=='zlib6':payload=zlib.compress(raw,6)
 elif mode=='lzma1':payload=lzma.compress(raw,preset=1)
 elif mode=='shuffle_zlib6':payload=zlib.compress(np.frombuffer(raw,dtype=np.uint8).reshape(-1,8).T.copy().tobytes() if len(raw) else b'',6)
 elif mode=='zstd3':payload=zstd.ZstdCompressor(level=3).compress(raw)
 else:raise ValueError('bad generic mode')
 return G_MAGIC+struct.pack('<BII',G_MODES[mode],len(raw)//8,len(payload))+payload

def generic_decode(blob):
 if blob[:4]!=G_MAGIC or len(blob)<13:raise ValueError('generic frame')
 mode,n,k=struct.unpack('<BII',blob[4:13]);p=blob[13:]
 if len(p)!=k:raise ValueError('generic frame length')
 if mode==0:raw=zlib.decompress(p)
 elif mode==1:raw=lzma.decompress(p)
 elif mode==2:
  dec=zlib.decompress(p)
  if len(dec)%8:raise ValueError('shuffled length')
  raw=np.frombuffer(dec,dtype=np.uint8).reshape(8,-1).T.copy().tobytes() if n else b''
 elif mode==3:raw=zstd.ZstdDecompressor().decompress(p)
 else:raise ValueError('unknown generic mode')
 if len(raw)!=8*n:raise ValueError('generic reconstructed length')
 return raw

def measure(data,encode,decode,reps=2):
 enc=[];dec=[];size=None
 for it in range(reps+1):
  st=time.perf_counter_ns();compressed=encode(data);et=(time.perf_counter_ns()-st)/1e6
  st=time.perf_counter_ns();rebuilt=decode(compressed);dt=(time.perf_counter_ns()-st)/1e6
  if rebuilt!=data:raise AssertionError('NOT bit-exact')
  if it:size=len(compressed);enc.append(et);dec.append(dt)
 return size,float(statistics.median(enc)),float(statistics.median(dec))

def negative_data():
 rng=np.random.default_rng(SEED)
 base=(rng.integers(-400,400,size=4096)/100).astype('<f8')
 mask=rng.random(len(base))<0.3
 patterns=np.array([0x8000000000000000,0x7ff8000000000001,0x7ff8000000000011,0xfff0000000000000],dtype='<u8')
 words=base.view('<u8').copy();words[mask]=rng.choice(patterns,size=int(mask.sum()))
 repeated=words.view('<f8').copy()
 words2=base.view('<u8').copy();unique_pos=np.flatnonzero(mask)
 for p in unique_pos:words2[p]=np.uint64(0x7ff8000000000000 | (int(p)+1))
 unique=words2.view('<f8').copy()
 # Rows are deliberately shuffled only in the exception locations for a control
 swapped=words.copy();swapped[mask]=rng.permutation(swapped[mask]);permuted=swapped.view('<f8').copy()
 return {'synthetic_repeated_exceptions':repeated,'synthetic_unique_nan_payloads':unique,'synthetic_exception_order_permuted':permuted,'synthetic_all_decimal':base,'synthetic_empty':np.array([],dtype='<f8')}

def development_datasets():
 families=[('sklearn_wine',load_wine().data),('sklearn_diabetes',load_diabetes().data),('sklearn_breast_cancer',load_breast_cancer().data),('sklearn_digits',load_digits().data)]
 for family,vals in families:
  for j in range(min(6,vals.shape[1])):
   yield family+'__col'+str(j).zfill(2),np.asarray(vals[:,j],dtype='<f8'),'sklearn_bundled',family
 # UCI used as known development data, not labelled blind external confirmation
 urls={
  'uci_wine_red':'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
  'uci_wine_white':'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
 }
 archived_sources={}
 for path in (Path('/content/SGX_NB70_results.zip'),Path('/mnt/data/SGX_NB70_results.zip')):
  if path.exists():
   try:
    with zipfile.ZipFile(path) as archive:
     for fam in ('uci_wine_red','uci_wine_white'):
      archived_sources[fam]=archive.read('wine_'+fam.rsplit('_',1)[-1]+'_source.csv')
    break
   except Exception as e:
    REMOTE_ERRORS.append({'family':'NB70_ARCHIVE','url':str(path),'error':repr(e)})
 for family,url in urls.items():
  try:
   if family in archived_sources:
    content=archived_sources[family]
   else:
    with urllib.request.urlopen(url,timeout=12) as resp:content=resp.read()
   lines=io.StringIO(content.decode('utf-8-sig'));rows=list(csv.DictReader(lines,delimiter=';'))
   if not rows:raise ValueError('empty remote dataset')
   for col in ['fixed acidity','total sulfur dioxide','chlorides','alcohol']:
    values=np.asarray([float(row[col]) for row in rows],dtype='<f8')
    yield family+'__'+col.replace(' ','_'),values,hashlib.sha256(content).hexdigest(),family
  except Exception as err:
   REMOTE_ERRORS.append({'family':family,'url':url,'error':repr(err)})

OUT=Path('/content' if Path('/content').exists() else '/mnt/data')/'SGX_NB85_outputs';OUT.mkdir(parents=True,exist_ok=True)
REMOTE_ERRORS=[]

def writecsv(name,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def export():
 p=OUT.parent/'SGX_NB85_results.zip'
 with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
  for q in sorted(OUT.rglob('*')):
   if q.is_file():z.write(q,q.relative_to(OUT))
 print('RESULT_ZIP',p,'SIZE',p.stat().st_size)
 try:
  from google.colab import files;files.download(str(p))
 except ImportError:pass
 return p

def run():
 edge=[]
 samples=negative_data()
 samples['special_bits']=np.asarray([0x7ff8000000000001,0x7ff8000000003333,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x0000000000000001,0xffffffffffffffff],dtype='<u8').view('<f8')
 for label,arr in samples.items():
  b=arr.tobytes()
  for scale_id in SCALES:
   for m in MODES:
    try:
     c=encode_exception(b,m,scale_id);d=decode_exception(c)
     edge.append({'case':label,'scale':SCALES[scale_id],'method':MODE_NAMES[m],'pass':d==b,'bytes':len(c)})
    except Exception as e:edge.append({'case':label,'scale':SCALES[scale_id],'method':MODE_NAMES[m],'pass':False,'error':repr(e)})
 assert all(x['pass'] for x in edge),'edge roundtrip failed'
 writecsv('bit_exact_edge_cases.csv',edge)
 
 rows=[];summary=[];errors=[];provenance=[]
 streams=list(development_datasets())
 for name,vals,source,family in streams:
  # 25% chronological calibration, 75% heldout; no method is selected using test
  if len(vals)<60:continue
  cut=max(1,len(vals)//4);raw=f64bytes(vals[cut:]);dig=hashlib.sha256(raw).hexdigest()
  provenance.append({'stream':name,'family':family,'n':len(vals),'heldout_n':len(vals)-cut,'heldout_sha256':dig,'source_id':source})
  for sid,scale in SCALES.items():
   q,pos,exc=make_int_and_exceptions(raw,scale)
   k=len(np.unique(exc));m=len(pos)
   config=[]
   for method in MODES:
    label=MODE_NAMES[method]
    try:
     size,et,dt=measure(raw,lambda b,mm=method,ss=sid:encode_exception(b,mm,ss),decode_exception)
     r={'stream':name,'family':family,'scale':scale,'mode':label,'scope':'heldout','n':len(q),'n_exceptions':m,'unique_exceptions':k,'bytes':size,'enc_ms':et,'dec_ms':dt,'exact':True,'sha256':dig}
     rows.append(r);config.append(r)
    except Exception as e:errors.append({'stream':name,'scale':scale,'mode':label,'error':repr(e)})
   # full-file generic baselines are identical per scale, but recorded for each comparison
   generic=[]
   for label in list(G_MODES):
    if label=='zstd3' and zstd is None:continue
    try:
     size,et,dt=measure(raw,lambda b,ll=label:generic_encode(b,ll),generic_decode)
     r={'stream':name,'family':family,'scale':scale,'mode':label,'scope':'heldout','n':len(q),'n_exceptions':m,'unique_exceptions':k,'bytes':size,'enc_ms':et,'dec_ms':dt,'exact':True,'sha256':dig}
     rows.append(r);generic.append(r)
    except Exception as e:errors.append({'stream':name,'scale':scale,'mode':label,'error':repr(e)})
   dictionary=next((r for r in config if r['mode']=='global_exception_dictionary'),None)
   noncandidate=[r for r in config if r['mode']!='global_exception_dictionary']+generic
   if dictionary and noncandidate:
    best=min(noncandidate,key=lambda x:x['bytes'])
    summary.append({'stream':name,'family':family,'scale':scale,'n':len(q),'n_exceptions':m,'unique_exceptions':k,'dictionary_bytes':dictionary['bytes'],'best_other_bytes':best['bytes'],'best_other_method':best['mode'],'gain_percent':100*(best['bytes']-dictionary['bytes'])/best['bytes'],'is_win':dictionary['bytes']<best['bytes'],'sha256':dig})
 writecsv('method_measurements.csv',rows);writecsv('real_stream_decisions.csv',summary);writecsv('dataset_provenance.csv',provenance);writecsv('download_errors.csv',REMOTE_ERRORS);writecsv('codec_errors.csv',errors)
 controls=[]
 for name,vals in negative_data().items():
  raw=f64bytes(vals);scaleid=1
  m=len(make_int_and_exceptions(raw,SCALES[scaleid])[1]);u=len(np.unique(make_int_and_exceptions(raw,SCALES[scaleid])[2]))
  method_sizes={MODE_NAMES[i]:len(encode_exception(raw,i,scaleid)) for i in MODES}
  method_sizes.update({p:len(generic_encode(raw,p)) for p in G_MODES if p!='zstd3' or zstd})
  controls.append({'case':name,'n':len(vals),'n_exceptions':m,'unique_exception_values':u,'info_theoretic_dictionary_break_even_uncompressed':64*u + m*(math.ceil(math.log2(max(u,1)))) < 64*m if m else False,**method_sizes})
 writecsv('synthetic_negative_controls.csv',controls)
 primary=[r for r in summary if r['scale']==100]
 families=sorted({r['family'] for r in primary})
 tot_c=sum(r['dictionary_bytes'] for r in primary);tot_b=sum(r['best_other_bytes'] for r in primary)
 byfamily=[]
 for fam in families:
  rr=[r for r in primary if r['family']==fam]
  aa=sum(r['dictionary_bytes'] for r in rr);bb=sum(r['best_other_bytes'] for r in rr)
  byfamily.append({'family':fam,'streams':len(rr),'dict_bytes':aa,'best_other_bytes':bb,'savings_percent':100*(bb-aa)/bb if bb else None,'wins':sum(r['is_win'] for r in rr)})
 writecsv('family_summary.csv',byfamily)
 gain=100*(tot_b-tot_c)/tot_b if tot_b else 0
 passes=(len(families)>=2 and gain>=3 and sum(f['savings_percent']>=3 for f in byfamily)>=2 and sum(r['gain_percent']<0 for r in primary)<=len(primary)//4 and not errors)
 decision={'research_phase':'minimal_feasibility_not_novelty_proof','candidate':'cross_block_exact_exception_dictionary','method_prior_art':'Classical dictionary coding and prior ALP exception streams; this mechanism is NOT claimed novel','primary_predefined_scale':100,'families':len(families),'streams':len(primary),'exception_dictionary_total_bytes':tot_c,'strongest_non_candidate_per_stream_total_bytes':tot_b,'aggregate_gain_percent':gain,'candidate_wins':sum(r['is_win'] for r in primary),'candidate_losses':sum(r['gain_percent']<0 for r in primary),'rejection_threshold':'At least 3% aggregate and at least two dataset families at 3%, at most 25% losing streams, and zero codec errors; additionally competitive native ALP and a literature-distinct mechanism are mandatory before a paper claim.','feasibility_gate':'CONTINUE_ONLY_TO_PRIOR_ART_NOVELTY_CHECK' if passes else 'REJECT_MECHANISM','native_ALP_matched':'NOT_RUN','external_blind_confirmation':'NOT_RUN','zstd_available':zstd is not None,'errors':len(errors),'download_failures':REMOTE_ERRORS,'sources':'sklearn bundled (previously used), UCI Wine Quality (previously used); development only'}
 (OUT/'decision.json').write_text(json.dumps(decision,indent=2))
 (OUT/'environment.json').write_text(json.dumps({'python':sys.version,'numpy':np.__version__,'sklearn':__import__('sklearn').__version__,'zstandard':getattr(zstd,'__version__',None),'platform':platform.platform(),'seed':SEED,'block_size':BLOCK,'scale_primary':100,'measurement_note':'sum of per-stream median encoding/decoding times; calibration cost not included; primary metric file bytes includes custom header and sections'},indent=2))
 print('MAIN_DECISION',json.dumps(decision,indent=2));export()




run()

