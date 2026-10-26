#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# NB65 — Exception-Mask Coupling Feasibility Test

**Falsifiable hypothesis:** Exact-decimal reconstruction exceptions sometimes co-occur across related columns. Row-synchronous compression of their **exception masks** might save bytes compared with independent per-column masks, even after all scale factors, integer values, bitwise exceptions, compressed sections, and metadata are counted.

**Ablation controls:** identical integer and raw-exception payloads with (A) independent masks, (B) jointly bit-packed masks, and (C) per-row mask patterns, plus a row-shifted negative control. Compare with calibration-selected per-column specialized and row-major generic baselines. On the first 25% of rows, choose all scales, baseline methods, and candidate mode; freeze these choices on the last 75%. A new mode must beat both the baseline (5% margin) and its own independent-mask ablation (2% margin) **on calibration**. Negative/no-gain results count as scientific outcomes.

**Prior art alert:** ALP already handles decimal reconstruction and exceptions; bitmap and inter-column entropy coding predate this work. A successful mask-coupling demonstration would not alone prove originality.

Run the single code cell to generate `SGX_NB65_results.zip`.
"""

import os,sys,io,csv,json,math,struct,time,hashlib,zipfile,platform,statistics,zlib,lzma,urllib.request,importlib.util,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_wine, load_breast_cancer, load_digits

NB_NUMBER = 65
SMOKE_TEST = bool(int(os.environ.get('SGX_SMOKE_TEST','0')))
ROOT=Path('/content' if Path('/content').exists() else '/mnt/data')
OUT=ROOT/(f'SGX_NB{NB_NUMBER}_SMOKE_outputs' if SMOKE_TEST else f'SGX_NB{NB_NUMBER}_outputs')
OUT.mkdir(parents=True,exist_ok=True)
def save_csv(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row.keys()))
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
def save_json(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,default=str),encoding='utf-8')
def finish():
    name=f'SGX_NB{NB_NUMBER}'+('_SMOKE' if SMOKE_TEST else '')+'_results.zip'
    dest=ROOT/name
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(OUT))
    print('RESULT_ZIP',dest,'size',dest.stat().st_size)
    try:
        from google.colab import files
        files.download(str(dest))
    except ImportError:pass
    except Exception as ex:print('Please download manually:',ex)
    return dest
def environment():
    save_json('environment.json',{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform(),'smoke_test':SMOKE_TEST,'zstandard_present':importlib.util.find_spec('zstandard') is not None,'codec_reference':'SGX NB52 exact decimal lattice + specialized alternatives','claim':'exploratory proof-of-feasibility, no novelty established'})
def bytes_matrix(x):
    return np.ascontiguousarray(x,dtype='<f8').tobytes()
def bytes_column(x,j):
    return np.ascontiguousarray(x[:,j],dtype='<f8').tobytes()
def matrix_raw(x):
    return np.frombuffer(bytes_matrix(x),dtype='<u8').copy()
def development_data():
    full=[('diabetes',load_diabetes().data),('wine_classification',load_wine().data),('breast_cancer',load_breast_cancer().data),('digits',load_digits().data)]
    return [(name,np.ascontiguousarray(a[:,:min(10,a.shape[1])],dtype='<f8')) for name,a in full]
def sha(blob):return hashlib.sha256(blob).hexdigest()
def safe_run(tag,fn,*args):
    try:return fn(*args),None
    except (ValueError, OverflowError, struct.error) as e:return None,f'{type(e).__name__}: {e}'

SCALES=[1,10,100,1000,10000]
MAG=b'QPK1'
def bitpack(values,width):
  acc=0;pending=0;out=bytearray()
  for v in values:
   v=int(v)
   if v<0 or (width and v >= (1<<width)) or (not width and v):raise ValueError('Out-of-range packed value')
   acc|=v<<pending;pending+=width
   while pending>=8:
    out.append(acc&255);acc>>=8;pending-=8
  if pending:out.append(acc&255)
  return bytes(out)
def bitunpack(buf,width,count):
  acc=0;pending=0;it=iter(buf);out=[]
  for _ in range(count):
   while pending<width:
    try:acc|=next(it)<<pending
    except StopIteration:raise ValueError('Truncated packed values')
    pending+=8
   out.append(acc&((1<<width)-1));acc>>=width;pending-=width
  if any(it):raise ValueError('Trailing nonzero packed bytes')
  return out
def lattice_pack(r,scale):
 if len(r)%8:raise ValueError('misaligned input')
 a=np.frombuffer(r,dtype='<f8'); n=len(a)
 # Word equality rather than numeric equality handles -0.0 and NaN payloads.
 original=np.frombuffer(r,dtype='<u8')
 present=[];integers=[];exceptions=[]
 for i,v in enumerate(a):
  v=float(v);accept=False;k=0
  if math.isfinite(v) and -2147483648 <= v*scale <= 2147483647:
   candidate=int(round(v*scale))
   if -2147483648<=candidate<=2147483647:
    restored=float(candidate)/scale
    if struct.pack('<d',restored)==r[i*8:i*8+8]:accept=True;k=candidate
  if accept:present.append(1);integers.append(k)
  else:present.append(0);exceptions.append(r[i*8:i*8+8])
 mn=min(integers) if integers else 0
 shifted=[v-mn for v in integers]
 width=max(shifted,default=0).bit_length()
 packed=bitpack(shifted,width)
 marker=bitpack(present,1)
 sections=[zlib.compress(marker,6),zlib.compress(packed,6),zlib.compress(b''.join(exceptions),6)]
 # 4 magic, 1 scale index, 4 n, 4 accepted, 8 min, 1 width, 12 compressed section lengths
 header=MAG+struct.pack('<BIIqBIII',SCALES.index(scale),n,len(integers),mn,width,*[len(x) for x in sections])
 return header+b''.join(sections)
def lattice_unpack(blob):
 if blob[:4]!=MAG or len(blob)<34:raise ValueError('Bad header')
 k,n,naccept,mn,width,a,b,c=struct.unpack('<BIIqBIII',blob[4:34]);scale=SCALES[k]
 if n>100_000_000 or naccept>n or width>64:raise ValueError('Invalid parameters')
 if len(blob)!=34+a+b+c:raise ValueError('Length mismatch')
 p=34;marker=zlib.decompress(blob[p:p+a]);p+=a
 packed=zlib.decompress(blob[p:p+b]);p+=b
 rawexceptions=zlib.decompress(blob[p:p+c]);p+=c
 mask=bitunpack(marker,1,n)
 if sum(mask)!=naccept:raise ValueError('Mask count mismatch')
 numbers=bitunpack(packed,width,naccept)
 if len(rawexceptions)!=(n-naccept)*8:raise ValueError('Exception size mismatch')
 it=iter(numbers);e=0;out=bytearray()
 for tag in mask:
  if tag:
   val=(next(it)+mn)/scale
   out.extend(struct.pack('<d',val))
  else:out.extend(rawexceptions[e:e+8]);e+=8
 return bytes(out)
def generic_pack(mode,r):
 if mode=='zlib6':return zlib.compress(r,6)
 if mode=='lzma1':return lzma.compress(r,preset=1)
 if mode=='zstd3':
  import zstandard as zstd
  return zstd.ZstdCompressor(level=3).compress(r)
 raise ValueError(mode)
def generic_unpack(mode,b):
 if mode=='zlib6':return zlib.decompress(b)
 if mode=='lzma1':return lzma.decompress(b)
 if mode=='zstd3':
  import zstandard as zstd
  return zstd.ZstdDecompressor().decompress(b)
 raise ValueError(mode)

# Stronger complete-file alternatives; outer container header always included for every method.
METHODS=['zlib6','lzma1','zstd3','shuffle_zlib6','xor_zlib6','int_for','int_for_zlib','dict_bitpack','dict_zlib','float32_zlib']+['lattice_'+str(i) for i in range(5)]
METHODS=[m for m in METHODS if m!='zstd3' or importlib.util.find_spec('zstandard')]
NONLATTICE=[m for m in METHODS if not m.startswith('lattice_')]
CANDIDATES=[m for m in METHODS if m.startswith('lattice_')]
MIDS={name:i for i,name in enumerate(METHODS)}
INV={i:name for name,i in MIDS.items()}
FRAME=b'SGX5'
def _shuffle(r):return np.frombuffer(r,dtype=np.uint8).reshape(-1,8).T.copy().tobytes()
def _unshuffle(b):return np.frombuffer(b,dtype=np.uint8).reshape(8,-1).T.copy().tobytes()
def _xor(r):
 a=np.frombuffer(r,dtype='<u8');b=a.copy()
 if len(b)>1:b[1:]=a[1:]^a[:-1]
 return b.astype('<u8').tobytes()
def _unxor(r):
 a=np.frombuffer(r,dtype='<u8').copy();shift=1
 while shift<len(a):a[shift:]^=a[:-shift].copy();shift*=2
 return a.astype('<u8').tobytes()
def _integrals(r):
 a=np.frombuffer(r,dtype='<f8')
 if not np.all(np.isfinite(a)):raise ValueError('has nonfinite values')
 if np.any(a<-(2**31)) or np.any(a>(2**31-1)):raise ValueError('outside int32')
 v=a.astype('<i4')
 if v.astype('<f8').tobytes()!=r:raise ValueError('not bit-exact int32 (including signed zeros)')
 return v.astype(np.int64)
def _int_for_pack(r, zipped):
 a=_integrals(r);mn=int(a.min()) if len(a) else 0
 shift=(a-mn).astype(np.uint64);width=int(shift.max()).bit_length() if len(shift) else 0
 packed=bitpack(shift,width);return struct.pack('<qB',mn,width)+(zlib.compress(packed,6) if zipped else packed)
def _int_for_unpack(b,n,zipped):
 mn,w=struct.unpack('<qB',b[:9]);buf=zlib.decompress(b[9:]) if zipped else b[9:]
 vals=bitunpack(buf,w,n);x=np.array([mn+int(k) for k in vals],dtype=np.int64)
 if np.any(x<-(2**31)) or np.any(x>(2**31-1)):raise ValueError('invalid int')
 return x.astype('<f8').tobytes()
def _dict_pack(r,zipped):
 a=np.frombuffer(r,dtype='<u8');unique,ids=np.unique(a,return_inverse=True)
 if len(unique)>65535:raise ValueError('dictionary > 65535')
 w=max(0,(len(unique)-1).bit_length())
 p=struct.pack('<IB',len(unique),w)+unique.astype('<u8').tobytes()+bitpack(ids,w)
 return zlib.compress(p,6) if zipped else p
def _dict_unpack(b,n,zipped):
 data=zlib.decompress(b) if zipped else b
 k,w=struct.unpack('<IB',data[:5]);d=np.frombuffer(data,dtype='<u8',count=k,offset=5)
 idx=bitunpack(data[5+8*k:],w,n)
 if any(i>=k for i in idx):raise ValueError('bad dict id')
 return d[np.asarray(idx,dtype=np.int64)].astype('<u8').tobytes()
def encode_payload(m,r):
 if m.startswith('lattice_'):return lattice_pack(r,SCALES[int(m.split('_')[1])])
 if m in ('zlib6','lzma1','zstd3'):return generic_pack(m,r)
 if m=='shuffle_zlib6':return zlib.compress(_shuffle(r),6)
 if m=='xor_zlib6':return zlib.compress(_xor(r),6)
 if m=='int_for':return _int_for_pack(r,False)
 if m=='int_for_zlib':return _int_for_pack(r,True)
 if m=='dict_bitpack':return _dict_pack(r,False)
 if m=='dict_zlib':return _dict_pack(r,True)
 if m=='float32_zlib':
  a=np.frombuffer(r,dtype='<f8');b=a.astype('<f4')
  if b.astype('<f8').tobytes()!=r:raise ValueError('not bit-exact float32')
  return zlib.compress(b.tobytes(),6)
 raise ValueError('unknown method')
def decode_payload(m,b,n):
 if m.startswith('lattice_'):return lattice_unpack(b)
 if m in ('zlib6','lzma1','zstd3'):return generic_unpack(m,b)
 if m=='shuffle_zlib6':return _unshuffle(zlib.decompress(b))
 if m=='xor_zlib6':return _unxor(zlib.decompress(b))
 if m=='int_for':return _int_for_unpack(b,n,False)
 if m=='int_for_zlib':return _int_for_unpack(b,n,True)
 if m=='dict_bitpack':return _dict_unpack(b,n,False)
 if m=='dict_zlib':return _dict_unpack(b,n,True)
 if m=='float32_zlib':return np.frombuffer(zlib.decompress(b),dtype='<f4').astype('<f8').tobytes()
 raise ValueError('unknown method')
def encode(m,r):
 if len(r)%8:raise ValueError('unaligned')
 p=encode_payload(m,r)
 return FRAME+struct.pack('<BIQ',MIDS[m],len(r)//8,len(p))+p  # 17 bytes inclusive
# Fully self-describing stream; checked without external method context.
def decode(f):
 if f[:4]!=FRAME or len(f)<17:raise ValueError('frame')
 mid,n,nbytes=struct.unpack('<BIQ',f[4:17]);p=f[17:]
 if mid not in INV or len(p)!=nbytes:raise ValueError('body length / mode')
 b=decode_payload(INV[mid],p,n)
 if len(b)!=8*n:raise ValueError('decoded length')
 return b
def measured(m,r,reps=3):
 size=None;enc=[];dec=[]
 # warmup and take medians of measured real work; no cached encoding or selection time counted.
 for j in range(reps+1):
  st=time.perf_counter_ns();f=encode(m,r);a=(time.perf_counter_ns()-st)/1e6
  st=time.perf_counter_ns();b=decode(f);d=(time.perf_counter_ns()-st)/1e6
  if b!=r:raise AssertionError('BIT MISMATCH '+m)
  if j:size=len(f);enc.append(a);dec.append(d)
 return {'bytes':size,'bpv':8*size/(len(r)//8),'encode_ms_median':statistics.median(enc),'decode_ms_median':statistics.median(dec),'exact':True}
def all_edges():
 bits=np.array([0,0x8000000000000000,0x7ff8000000000001,0x7ff8000000000033,0x7ff0000000000000,0xfff0000000000000,0x0000000000000001,0xffffffffffffffff],dtype='<u8')
 examples={'empty':b'','special':bits.tobytes(),'repeated_special':bits.tobytes()*4,'integers':raw(np.arange(-20,20)),'decimal':raw(np.arange(50)/10)}
 out=[]
 for label,b in examples.items():
  for m in METHODS:
   try:
    c=encode(m,b);ok=decode(c)==b;out.append({'input':label,'mode':m,'status':'PASS' if ok else 'FAIL'})
   except ValueError as e:out.append({'input':label,'mode':m,'status':'INAPPLICABLE','reason':str(e)})
 assert not any(x['status']=='FAIL' for x in out)
 return out

# Hypothesis: row-wise co-occurrence in exact-decimal exception masks might enable better
# entropy coding than independent per-column masks. This is NOT claimed to be original.
# Prototypes contain all scales, mask lengths, original exception bits, integer values,
# metadata and complete decoder, with no external model state.
MASKS=b'MCE1'; COLUMN=b'SGC1'; ROW=b'SGR1'
MASK_MODES={'independent_mask':0,'joint_mask':1,'rowpattern_mask':2}
BACK_MODES={v:k for k,v in MASK_MODES.items()}
ROW_MODES={'zlib6':0,'lzma1':1,'zstd3':2}
INV_ROW_MODES={v:k for k,v in ROW_MODES.items()}
SPECIALIZED=['zlib6','lzma1','shuffle_zlib6','int_for_zlib','dict_zlib','float32_zlib']+(['zstd3'] if importlib.util.find_spec('zstandard') else [])+[f'lattice_{i}' for i in range(len(SCALES))]
AVAILABLE_ROW=['zlib6','lzma1']+(['zstd3'] if importlib.util.find_spec('zstandard') else [])

def exact_decimal_col(data,scale):
    if len(data)%8:raise ValueError('bad float64 column')
    a=np.frombuffer(data,dtype='<f8')
    ok=np.zeros(len(a),dtype=np.uint8)
    nums=[];exceptions=[]
    for i,num in enumerate(a):
        v=float(num)
        if math.isfinite(v):
            scaled=v*scale
            if math.isfinite(scaled) and -2147483648<=scaled<=2147483647:
                k=int(round(scaled))
                if -2147483648<=k<=2147483647 and struct.pack('<d',float(k)/scale)==data[8*i:8*i+8]:
                    ok[i]=1;nums.append(k);continue
        exceptions.append(data[8*i:8*i+8])
    return ok,np.array(nums,dtype='<i4').tobytes(),b''.join(exceptions)

def scale_for_calibration(cal_col):
    # Chosen without access to held-out data, with self-describing reference lattice codec.
    return int(min(range(len(SCALES)),key=lambda j:len(lattice_pack(cal_col,SCALES[j]))))

def masks_for_matrix(x,scale_indices):
    n,c=x.shape
    if c!=len(scale_indices):raise ValueError('invalid scale count')
    masks=np.zeros((n,c),dtype=np.uint8);payload=[]
    for col,idx in enumerate(scale_indices):
        if idx<0 or idx>=len(SCALES):raise ValueError('invalid decimal scale')
        m,ints,exceptions=exact_decimal_col(bytes_column(x,col),SCALES[idx]);masks[:,col]=m
        payload.append((ints,exceptions))
    return masks,payload

def mask_compressed_sections(masks,mode):
    n,c=masks.shape
    if mode=='independent_mask':
        return [zlib.compress(np.packbits(masks[:,j],bitorder='little').tobytes(),6) for j in range(c)]
    if mode=='joint_mask':
        return [zlib.compress(np.packbits(masks.reshape(-1),bitorder='little').tobytes(),6)]
    if mode=='rowpattern_mask':
        return [zlib.compress(np.packbits(masks,axis=1,bitorder='little').tobytes(),6)]
    raise KeyError(mode)

def decode_masks(sections,mode,n,c):
    if mode=='independent_mask':
        if len(sections)!=c:raise ValueError('mask columns')
        masks=np.stack([np.unpackbits(np.frombuffer(zlib.decompress(s),dtype=np.uint8),bitorder='little')[:n] for s in sections],axis=1)
    elif mode=='joint_mask':
        if len(sections)!=1:raise ValueError('mask sections')
        masks=np.unpackbits(np.frombuffer(zlib.decompress(sections[0]),dtype=np.uint8),bitorder='little')[:n*c].reshape(n,c)
    elif mode=='rowpattern_mask':
        if len(sections)!=1:raise ValueError('mask sections')
        width=(c+7)//8
        packed=np.frombuffer(zlib.decompress(sections[0]),dtype=np.uint8)
        if len(packed)!=n*width:raise ValueError('invalid rowpattern length')
        masks=np.unpackbits(packed.reshape(n,width),axis=1,bitorder='little')[:,:c]
    else:raise ValueError('mask mode')
    if masks.shape!=(n,c):raise ValueError('truncated masks')
    return masks.astype(np.uint8)

def encode_mask_matrix(x,scale_indices,mode):
    x=np.asarray(x,dtype='<f8');n,c=x.shape
    if c<1 or c>255 or n>2**32-1:raise ValueError('matrix size')
    masks,data=masks_for_matrix(x,scale_indices)
    masks_compressed=mask_compressed_sections(masks,mode)
    values=[];columns=[]
    for si,(ints,exc) in zip(scale_indices,data):
        ib=zlib.compress(ints,6);eb=zlib.compress(exc,6)
        columns.append((si,len(ib),len(eb)))
        values.extend((ib,eb))
    out=bytearray(MASKS+struct.pack('<BIBB',MASK_MODES[mode],n,c,len(masks_compressed)))
    out.extend(struct.pack('<'+'I'*len(masks_compressed),*[len(q) for q in masks_compressed]))
    for sc,il,el in columns:out.extend(struct.pack('<BII',sc,il,el))
    for q in masks_compressed:out.extend(q)
    for q in values:out.extend(q)
    return bytes(out)

def decode_mask_matrix(blob):
    if blob[:4]!=MASKS or len(blob)<11:raise ValueError('bad mask matrix header')
    modeidx,n,c,ns=struct.unpack('<BIBB',blob[4:11]);p=11
    if modeidx not in BACK_MODES or c==0 or ns==0:raise ValueError('bad mode or column count')
    if (modeidx==0 and ns!=c) or (modeidx!=0 and ns!=1):raise ValueError('mask section count')
    sizes=struct.unpack('<'+'I'*ns,blob[p:p+4*ns]);p+=4*ns
    entries=[]
    for j in range(c):
        scale,ilen,elen=struct.unpack('<BII',blob[p:p+9]);p+=9
        if scale>=len(SCALES):raise ValueError('invalid scale')
        entries.append((SCALES[scale],ilen,elen))
    masks=[]
    for length in sizes:masks.append(blob[p:p+length]);p+=length
    recovered=decode_masks(masks,BACK_MODES[modeidx],n,c)
    columns=[]
    for j,(scale,ilen,elen) in enumerate(entries):
        ib=zlib.decompress(blob[p:p+ilen]);p+=ilen
        eb=zlib.decompress(blob[p:p+elen]);p+=elen
        good=recovered[:,j].astype(bool)
        if len(ib)!=int(np.sum(good))*4 or len(eb)!=int(np.sum(~good))*8:raise ValueError('value section lengths')
        integers=np.frombuffer(ib,dtype='<i4');vals=np.empty(n,dtype='<u8')
        fixed=np.asarray(integers,dtype=np.float64)/scale
        vals[good]=fixed.view('<u8')
        vals[~good]=np.frombuffer(eb,dtype='<u8')
        columns.append(vals)
    if p!=len(blob):raise ValueError('trailing compressed bytes')
    return np.ascontiguousarray(np.column_stack(columns),dtype='<u8').tobytes(),n,c

def encode_column_baseline(x,methods):
    n,c=x.shape
    if len(methods)!=c:raise ValueError('mode count')
    parts=[encode(methods[j],bytes_column(x,j)) for j in range(c)]
    return COLUMN+struct.pack('<IB',n,c)+b''.join(struct.pack('<I',len(q))+q for q in parts)

def decode_column_baseline(blob):
    if blob[:4]!=COLUMN:raise ValueError('bad column baseline header')
    n,c=struct.unpack('<IB',blob[4:9]);p=9;cols=[]
    for j in range(c):
        length=struct.unpack('<I',blob[p:p+4])[0];p+=4
        b=decode(blob[p:p+length]);p+=length
        if len(b)!=n*8:raise ValueError('bad decoded column length')
        cols.append(np.frombuffer(b,dtype='<u8').copy())
    if p!=len(blob):raise ValueError('trailing column baseline bytes')
    return np.ascontiguousarray(np.column_stack(cols),dtype='<u8').tobytes(),n,c

def encode_row_baseline(x,mode):
    n,c=x.shape
    raw=bytes_matrix(x);body=generic_pack(mode,raw)
    return ROW+struct.pack('<IBBI',n,c,ROW_MODES[mode],len(body))+body

def decode_row_baseline(blob):
    if blob[:4]!=ROW:raise ValueError('bad row baseline header')
    n,c,mi,size=struct.unpack('<IBBI',blob[4:14]);body=blob[14:]
    if len(body)!=size:raise ValueError('bad row baseline size')
    raw=generic_unpack(INV_ROW_MODES[mi],body)
    if len(raw)!=n*c*8:raise ValueError('bad decoded size')
    return raw,n,c

def roundtrip_encoded(kind,x,scale_indices=None,col_methods=None):
    if kind in MASK_MODES:
        encoded=encode_mask_matrix(x,scale_indices,kind);raw,n,c=decode_mask_matrix(encoded)
    elif kind=='column_specialized':
        encoded=encode_column_baseline(x,col_methods);raw,n,c=decode_column_baseline(encoded)
    elif kind.startswith('row_'):
        encoded=encode_row_baseline(x,kind[4:]);raw,n,c=decode_row_baseline(encoded)
    else:raise KeyError(kind)
    if (n,c)!=x.shape or raw!=bytes_matrix(x):raise AssertionError('NOT BIT-EXACT: '+kind)
    return encoded

def measured_kind(kind,x,scale_indices=None,col_methods=None,reps=2):
    timings=[];sizes=[]
    for i in range(reps+1):
        t=time.perf_counter_ns();encoded=roundtrip_encoded(kind,x,scale_indices,col_methods);e=(time.perf_counter_ns()-t)/1e6
        if i:
            timings.append(e);sizes.append(len(encoded))
    assert len(set(sizes))==1
    return {'bytes':sizes[0],'encode_roundtrip_ms':float(statistics.median(timings)),'exact':True}

def cal_choices(cal):
    scales=[];methods=[]
    for j in range(cal.shape[1]):
        col=bytes_column(cal,j)
        scales.append(scale_for_calibration(col))
        valid=[]
        for mode in SPECIALIZED:
            try: valid.append((len(encode(mode,col)),mode))
            except (ValueError,OverflowError):continue
        if not valid:raise AssertionError('No valid column fallback')
        methods.append(min(valid)[1])
    return scales,methods

def mask_diagnostic(x,scale_indices,seed=9102):
    masks,_=masks_for_matrix(x,scale_indices)
    n,c=masks.shape
    indep=sum(len(q) for q in mask_compressed_sections(masks,'independent_mask'))
    joint=sum(len(q) for q in mask_compressed_sections(masks,'joint_mask'))
    rowpat=sum(len(q) for q in mask_compressed_sections(masks,'rowpattern_mask'))
    shifted=masks.copy();rng=np.random.default_rng(seed)
    for j in range(c):
        if n>1:shifted[:,j]=np.roll(masks[:,j],int(rng.integers(0,n)))
    neg=sum(len(q) for q in mask_compressed_sections(shifted,'joint_mask'))
    return {'mask_independent_bytes':indep,'mask_joint_bytes':joint,'mask_rowpattern_bytes':rowpat,'mask_shifted_control_bytes':neg,'acceptance_rate':float(np.mean(masks)),'joint_vs_independent_saving_bytes':indep-joint,'alignment_advantage_vs_shift_bytes':neg-joint}

def edge_cases():
    patterns=np.array([0x0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000021,0x0000000000000001,0xffffffffffffffff],dtype='<u8')
    cases={'empty':np.empty((0,3),dtype='<f8'),'special_bits':np.tile(patterns,4)[:27].reshape(9,3).view('<f8'),'decimal':(np.arange(48,dtype='<f8').reshape(12,4)-15)/100,'signed_zero':np.array([[0.0,-0.0,1.0],[2.5,3.75,-0.0]],dtype='<f8')}
    output=[]
    for name,x in cases.items():
        scales=[2]*x.shape[1];methods=['zlib6']*x.shape[1]
        for kind in list(MASK_MODES)+['column_specialized','row_zlib6','row_lzma1']:
            try:
                b=roundtrip_encoded(kind,x,scales,methods)
                output.append({'case':name,'codec':kind,'result':'PASS','bytes':len(b)})
            except Exception as ex:output.append({'case':name,'codec':kind,'result':'FAIL','error':repr(ex)})
    assert all(q['result']=='PASS' for q in output),[q for q in output if q['result']!='PASS']
    return output

environment();save_csv('edge_cases.csv',edge_cases())
metrics=[];frozen=[];controls=[];errors=[]
for name,x in development_data():
    if SMOKE_TEST:x=x[:min(len(x),120),:min(x.shape[1],5)]
    n=x.shape[0];cut=max(20,n//4);train=x[:cut];hold=x[cut:]
    try:
        scales,methods=cal_choices(train)
        codec_kinds=['column_specialized']+['row_'+m for m in AVAILABLE_ROW]+list(MASK_MODES)
        cal={mode:measured_kind(mode,train,scales,methods,reps=1)['bytes'] for mode in codec_kinds}
        held={mode:measured_kind(mode,hold,scales,methods,reps=2)['bytes'] for mode in codec_kinds}
        base=min(['column_specialized']+['row_'+m for m in AVAILABLE_ROW],key=lambda t:cal[t])
        candid=min(['joint_mask','rowpattern_mask'],key=lambda t:cal[t])
        # A candidate must improve its independent-mask ablation too, not merely a weak conventional comparator.
        qualify=(cal[candid]<=0.95*cal[base] and cal[candid]<=0.98*cal['independent_mask'])
        selected=candid if qualify else base
        for mode in codec_kinds:
            metrics.append({'dataset':name,'mode':mode,'calibration_bytes':cal[mode], 'heldout_bytes':held[mode],
              'original_bytes_holdout':len(bytes_matrix(hold)), 'bit_exact':True})
        diagnostic=mask_diagnostic(hold,scales)
        controls.append({'dataset':name,'n_heldout':len(hold),'columns':x.shape[1],**diagnostic})
        hindsight_base=min(['column_specialized']+['row_'+m for m in AVAILABLE_ROW],key=lambda t:held[t])
        frozen.append({'dataset':name,'n_calibration':len(train),'n_heldout':len(hold),
           'frozen_baseline':base,'candidate_mode':candid,'candidate_qualified':qualify,
           'selected':selected,'selected_bytes':held[selected],'deployable_baseline_bytes':held[base],
           'best_baseline_hindsight_bytes_diagnostic_only':held[hindsight_base],
           'gain_vs_deployable_baseline_pct':100*(held[base]-held[selected])/held[base],
           'gain_vs_hindsight_baseline_pct_diagnostic_only':100*(held[hindsight_base]-held[selected])/held[hindsight_base],
           'all_methods_bit_exact':True,'sha256_holdout':sha(bytes_matrix(hold))})
    except Exception as ex:errors.append({'dataset':name,'error':repr(ex),'traceback':traceback.format_exc()[-3000:]})
save_csv('ablation_comparisons.csv',metrics);save_csv('frozen_decisions.csv',frozen)
save_csv('negative_controls.csv',controls);save_csv('errors.csv',errors)
if frozen:
    baseline=sum(x['deployable_baseline_bytes'] for x in frozen);selected=sum(x['selected_bytes'] for x in frozen)
else:baseline=selected=0
summary={'status':'ERROR' if errors else 'COMPLETE','datasets':len(frozen),'selected_candidate_count':sum(x['selected'] in ('joint_mask','rowpattern_mask') for x in frozen),
 'total_deployable_baseline_bytes':baseline,'total_frozen_policy_bytes':selected,
 'aggregate_savings_pct':100*(baseline-selected)/baseline if baseline else None,
 'errors':errors,'novelty_warning':'Row-pattern/exception-mask entropy coding has extensive prior art; scientific novelty NOT established.',
 'stop_rule':'Retire hypothesis if no candidate selected and no mask-control advantage; require an external replication and native ALP later if positive.'}
save_json('decision.json',summary);print('NB65 decision',json.dumps(summary,indent=2)[:2500]);finish()

