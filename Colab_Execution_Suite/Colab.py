#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X — NB95F: Official Virtual bitwise-parity repair

**Rerun NB95 with the original published Virtual 0.2.2 implementation.** This version fixes a verification mistake in NB95: column order is not a valid reason to reject an otherwise correct decoded table. It now matches columns by name and compares the exact IEEE-754 float64 bits for every row. It rejects missing/renamed columns and all bit differences. It prints the decoded columns and any specific mismatch. No benchmark numbers have been invented.

It retains NB93 matched-source hashes, calibration-frozen ShiftGuard-X, full encoded file byte accounting, original Virtual and Parquet comparisons, and output audit. This is **not** a new codec or a claim of novelty. Run all cells in a fresh Google Colab CPU runtime.
"""

# Colab setup: mandatory packages, plus published Virtual implementation.
# Any installation failure is explicitly logged; never substitute synthetic data or a proxy.
import sys,subprocess,importlib,importlib.util,platform
INSTALL={}
PINS={'zstandard':'0.25.0','pcodec':'1.0.3','pyarrow':'20.0.0',
      'pandas':'2.2.2','numpy':'2.0.2','virtual-parquet':'0.2.2'}
for package,version in PINS.items():
    module={'virtual-parquet':'virtual'}.get(package,package)
    if importlib.util.find_spec(module) is not None:
        INSTALL[package]={'status':'already_present'}
    else:
        try:
            r=subprocess.run([sys.executable,'-m','pip','install','-q',f'{package}=={version}'],
                 capture_output=True,text=True,timeout=300)
            INSTALL[package]={'status':'installed' if r.returncode==0 else 'INSTALL_FAILED',
                              'returncode':r.returncode,'stderr_tail':r.stderr[-900:]}
        except Exception as e:INSTALL[package]={'status':'INSTALL_FAILED','error':repr(e)}
print(INSTALL)


import os, io, csv, json, time, lzma, struct, hashlib, zipfile, urllib.request, platform, traceback
from pathlib import Path
import numpy as np, pandas as pd
import zstandard as zstd
try:
    import pcodec
    from pcodec import standalone as pcs
except Exception:
    pcodec=pcs=None
import pyarrow.parquet as pq
ROOT=Path('/content') if Path('/content').exists() else Path('/mnt/data')
OUT=ROOT/'SGX_NB95F_outputs'; OUT.mkdir(exist_ok=True)
CACHE=ROOT/'SGX_NB95F_source_cache'; CACHE.mkdir(exist_ok=True)
ERR=[]; DATA=[]; PROVENANCE=[]
UCI_URL='https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'
NYC_URL='https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet'
MAX_TAXI_ROWS=120000

def download_exact(url, dest):
    if not dest.exists() or dest.stat().st_size == 0:
        req=urllib.request.Request(url,headers={'User-Agent':'ShiftGuardX-Research/93 (noncommercial academic benchmark)'})
        with urllib.request.urlopen(req,timeout=100) as src, open(dest,'wb') as dst:
            while True:
                b=src.read(1024*1024)
                if not b:break
                dst.write(b)
    b=dest.read_bytes()
    return hashlib.sha256(b).hexdigest(),len(b)

def load_bike():
    path=CACHE/'bike_sharing_uci275.zip'
    sha,sz=download_exact(UCI_URL,path)
    with zipfile.ZipFile(path) as z:
        member=next(s for s in z.namelist() if s.lower().endswith('hour.csv'))
        df=pd.read_csv(io.BytesIO(z.read(member)))
    names=['casual','registered','cnt']
    assert all(s in df.columns for s in names)
    m={s:np.ascontiguousarray(df[s].to_numpy(dtype='<f8')) for s in names}
    PROVENANCE.append({'family':'UCI-Bike-275','url':UCI_URL,'sha256':sha,'source_bytes':sz,'n_rows':len(df),'source_member':member,'transform':'Original integer-valued columns converted exactly to IEEE float64; not CSV-byte restoration'})
    return {'family':'UCI-Bike-275','sources':names[:-1], 'target':names[-1], 'values':m}

def load_taxi():
    path=CACHE/'yellow_tripdata_2024-01.parquet'
    sha,sz=download_exact(NYC_URL,path)
    pqfile=pq.ParquetFile(path)
    available={s.lower():s for s in pqfile.schema_arrow.names}
    required=['fare_amount','extra','mta_tax','tip_amount','tolls_amount','improvement_surcharge','congestion_surcharge','airport_fee','total_amount']
    missing=[c for c in required if c not in available]
    if missing:raise ValueError('Source Parquet missing required fields: '+str(missing))
    actual=[available[k] for k in required]
    # Bounded memory: only columns needed, and stop after first MAX_TAXI_ROWS.
    tables=[];nrows=0
    for batch in pqfile.iter_batches(batch_size=25000,columns=actual):
        if nrows>=MAX_TAXI_ROWS:break
        b=batch.to_pandas().iloc[:MAX_TAXI_ROWS-nrows]
        tables.append(b);nrows+=len(b)
    df=pd.concat(tables,ignore_index=True)
    m={canon:np.ascontiguousarray(pd.to_numeric(df[available[canon]],errors='coerce').to_numpy(dtype='<f8')) for canon in required}
    PROVENANCE.append({'family':'NYC-TLC-Yellow-2024-01','url':NYC_URL,'sha256':sha,'source_bytes':sz,'n_rows':len(df),'source_member':'first 120000 parquet rows; selected numeric columns','transform':'Parquet-decoded float64 bit patterns; null conversion to numpy NaN, not original parquet-file bytes'})
    return {'family':'NYC-TLC-Yellow-2024-01','sources':required[:-1], 'target':required[-1], 'values':m}

for label,fn in [('UCI-Bike-275',load_bike),('NYC-TLC-Yellow-2024-01',load_taxi)]:
    try:
        d=fn();DATA.append(d)
        print(label,'loaded rows:',len(d['values'][d['target']]))
    except Exception as e:
        ERR.append({'family':label,'stage':'data_loading','error':repr(e)})
        print(label,'INCOMPLETE:',repr(e))


# Complete-file coder; every method carries its decoding metadata.
MAGIC=b'SG93'; C=zstd.ZstdCompressor(level=3);D=zstd.ZstdDecompressor()

def as_f8(arr):return np.asarray(arr,dtype='<f8').copy()
def fbits(arr):return as_f8(arr).view('<u8')
def as_u8(data):return np.frombuffer(data,dtype='<u8').copy()

def split8(b):
    x=np.frombuffer(b,dtype=np.uint8)
    if len(x)%8:raise ValueError('nonaligned byte stream')
    return x.reshape(-1,8).T.copy().tobytes()
def unsplit8(b):
    x=np.frombuffer(b,dtype=np.uint8)
    if len(x)%8:raise ValueError('nonaligned split stream')
    return x.reshape(8,-1).T.copy().tobytes()

def enc_bytes(raw,codec):
    if codec=='zstd':return C.compress(raw)
    if codec=='zstd_split':return C.compress(split8(raw))
    if codec=='lzma':return lzma.compress(raw,preset=1)
    raise ValueError(codec)
def dec_bytes(payload,codec):
    if codec=='zstd':return D.decompress(payload)
    if codec=='zstd_split':return unsplit8(D.decompress(payload))
    if codec=='lzma':return lzma.decompress(payload)
    raise ValueError(codec)

def enc_column(v,method):
    if method=='pcodec':
        if pcs is None:raise RuntimeError('pcodec unavailable')
        return bytes(pcs.simple_compress(as_f8(v),pcodec.ChunkConfig()))
    return enc_bytes(as_f8(v).tobytes(),method)
def dec_column(p,method,n):
    if method=='pcodec':
        if pcs is None:raise RuntimeError('pcodec unavailable')
        v=as_f8(pcs.simple_decompress(p))
    else:
        raw=dec_bytes(p,method)
        if len(raw)!=8*n:raise ValueError('wrong decoded byte count')
        v=np.frombuffer(raw,dtype='<f8').copy()
    if len(v)!=n:raise ValueError('wrong decoded element count')
    return v

METHODS=['zstd','zstd_split','lzma']+(['pcodec'] if pcs is not None else [])
def frozen_codec(calib):
    outcomes=[]
    for method in METHODS:
        try:
            t=time.perf_counter();p=enc_column(calib,method)
            assert dec_column(p,method,len(calib)).tobytes()==as_f8(calib).tobytes()
            outcomes.append((len(p),method,time.perf_counter()-t))
        except Exception as e:ERR.append({'stage':'codec_trial','codec':method,'error':repr(e)})
    if not outcomes:raise RuntimeError('no bit-exact compressor available')
    return min(outcomes)[1],outcomes

def predictors(values,sources):
    n=len(values[sources[0]])
    with np.errstate(all='ignore'):
        raw=np.zeros(n,dtype='<f8')
        for name in sources:raw=np.add(raw,values[name],dtype='<f8')
        cents=np.zeros(n,dtype=np.int64)
        valid=np.ones(n,dtype=bool)
        for name in sources:
            v=as_f8(values[name])
            okay=np.isfinite(v)&(np.abs(v)<1e12)
            valid &= okay
            vv=np.where(okay,v,0)
            cents+=np.rint(vv*100.0).astype(np.int64)
        cents_pred=cents.astype('<f8')/100.0
        cents_pred[~valid]=np.nan
    return raw,cents_pred

def masks(values,sources,target,mode):
    truth=fbits(values[target]); a,b=predictors(values,sources)
    a_bits=fbits(a);b_bits=fbits(b)
    if mode=='mask_raw':
        code=np.where(truth==a_bits,0,2).astype(np.uint8)
    elif mode=='mask_cents':
        code=np.where(truth==b_bits,0,2).astype(np.uint8)
    elif mode=='mask_tri':
        code=np.where(truth==a_bits,0,np.where(truth==b_bits,1,2)).astype(np.uint8)
    else:raise ValueError(mode)
    return code

def pack_mode(code,tri):
    if tri:
        n=len(code);pad=(-n)%4
        if pad:code=np.pad(code,(0,pad),constant_values=3)
        x=code.reshape(-1,4)
        return ((x[:,0]<<6)|(x[:,1]<<4)|(x[:,2]<<2)|x[:,3]).astype('u1').tobytes()
    return np.packbits(code==2,bitorder='big').tobytes()

def unpack_mode(raw,n,tri):
    if tri:
        x=np.frombuffer(raw,dtype='u1');out=np.empty(len(x)*4,dtype='u1')
        out[0::4]=x>>6;out[1::4]=(x>>4)&3;out[2::4]=(x>>2)&3;out[3::4]=x&3
        return out[:n].copy()
    return (np.unpackbits(np.frombuffer(raw,dtype='u1'),bitorder='big')[:n]*2).astype('u1')

def table_hash(values,names):
    h=hashlib.sha256()
    for name in names:h.update(as_f8(values[name]).tobytes())
    return h.hexdigest()

def frame_pack(meta,chunks):
    assert len(meta['chunks'])==len(chunks)
    m=json.dumps(meta,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')
    return MAGIC+struct.pack('<I',len(m))+m+b''.join(chunks)

def frame_unpack(p):
    if len(p)<8 or p[:4]!=MAGIC:raise ValueError('frame truncated or invalid')
    mlen=struct.unpack_from('<I',p,4)[0]
    if mlen>1024*1024 or len(p)<8+mlen:raise ValueError('truncated metadata')
    meta=json.loads(p[8:8+mlen]);pos=8+mlen;chunks=[]
    for info in meta['chunks']:
        length=int(info['nbytes'])
        if length<0 or pos+length>len(p):raise ValueError('truncated chunk')
        chunks.append(p[pos:pos+length]);pos+=length
    if pos!=len(p):raise ValueError('trailing frame bytes')
    return meta,chunks

def encode(values,sources,target,mode,frozen_methods):
    n=len(values[target]); names=sources+[target];parts=[];tags=[]
    def add(kind,p):
        tags.append({'kind':kind,'nbytes':len(p)});parts.append(p)
    for src in sources:add('col:'+src,enc_column(values[src],frozen_methods[src]))
    targetv=as_f8(values[target]);truth=targetv.view('<u8')
    if mode=='raw':
        add('target',enc_column(targetv,frozen_methods[target]))
    elif mode in ('mask_raw','mask_cents','mask_tri'):
        code=masks(values,sources,target,mode)
        add('mask',C.compress(pack_mode(code,mode=='mask_tri')))
        add('exceptions',C.compress(truth[code==2].astype('<u8').tobytes()))
    elif mode=='xor_raw':
        pred,_=predictors(values,sources)
        add('xor',C.compress(split8((truth^fbits(pred)).astype('<u8').tobytes())))
    else:raise ValueError(mode)
    meta={'schema':'SG93-v1','n':n,'sources':sources,'target':target,'mode':mode,'frozen_methods':frozen_methods,'raw_sha256':table_hash(values,names),'chunks':tags}
    return frame_pack(meta,parts)

def decode(frame):
    meta,parts=frame_unpack(frame);n=int(meta['n']);sources=meta['sources'];target=meta['target'];mode=meta['mode'];methods=meta['frozen_methods']
    values={s:dec_column(p,methods[s],n) for s,p in zip(sources,parts[:len(sources)])}
    extra=parts[len(sources):]
    if mode=='raw':
        assert len(extra)==1
        values[target]=dec_column(extra[0],methods[target],n)
    elif mode in ('mask_raw','mask_cents','mask_tri'):
        if len(extra)!=2:raise ValueError('bad masks')
        tri=(mode=='mask_tri');mraw=D.decompress(extra[0]);expected=(n+3)//4 if tri else (n+7)//8
        if len(mraw)!=expected:raise ValueError('bad mask byte count')
        code=unpack_mode(mraw,n,tri)
        if np.any(code>2) or (not tri and np.any(code==1)):raise ValueError('invalid mask code')
        pred_a,pred_b=predictors(values,sources)
        words=fbits(pred_a if mode!='mask_cents' else pred_b)
        if tri:words=np.where(code==1,fbits(pred_b),words)
        exceptions=as_u8(D.decompress(extra[1]))
        if len(exceptions)!=int(np.sum(code==2)):raise ValueError('wrong exception count')
        words[code==2]=exceptions
        values[target]=words.view('<f8').copy()
    elif mode=='xor_raw':
        if len(extra)!=1:raise ValueError('bad xor format')
        words=as_u8(unsplit8(D.decompress(extra[0])))
        if len(words)!=n:raise ValueError('bad xor values')
        a,_=predictors(values,sources)
        values[target]=(words^fbits(a)).view('<f8').copy()
    else:raise ValueError('unrecognized mode')
    if table_hash(values,sources+[target])!=meta['raw_sha256']:raise ValueError('checksum mismatch')
    return values


# Local algorithmic smoke test only. NEVER call these data "real dataset results".
words=np.array([0x0000000000000000,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,0x7ff80000000000ab,0x0000000000000001,0x7fefffffffffffff],dtype='<u8')
odd=words.view('<f8').copy(); parents={'a':odd,'b':np.array([1.,-2.,3.,4.,5.,6.,7.,8.],dtype='<f8')}
y=odd.copy();values={**parents,'y':y}
for mode in ['raw','mask_raw','mask_cents','mask_tri','xor_raw']:
    methods={k:'zstd' for k in values}
    buf=encode(values,['a','b'],'y',mode,methods)
    restored=decode(buf)
    assert all(fbits(restored[k]).tobytes()==fbits(values[k]).tobytes() for k in values)
    try:decode(buf[:-1]);raise AssertionError('truncation was accepted')
    except (ValueError,zstd.ZstdError,lzma.LZMAError):pass
print('PASS: all five modes, signed zeros, payload-distinct NaNs, infinities, subnormal, truncated frames')


"""
## Matched held-out experiment

The NB93 source-loading, reversible encoding and decoding routines above are **identical to the earlier implementation**, apart from output/cache folder names. NB95 uses calibration to select the SGX encoding mode and its per-column compression methods; held-out data never changes those choices. Virtual runs from its **official `virtual.to_format` / `virtual.from_format` public API** and is measured as a native transductive table encoder. Full on-disk output sizes are used; checksum and strict row/column/bitwise validation are mandatory. Any different dataset hash blocks claim of matching NB93. No post-test switching of the retained SGX choice.
"""

import tempfile,os,importlib.metadata
import pyarrow as pa,pyarrow.parquet as pq
try:
    import virtual as official_virtual
    virtual_status='IMPORTED_OFFICIAL_PACKAGE'
except Exception as e:
    official_virtual=None;virtual_status='UNAVAILABLE';ERR.append({'family':'ALL','stage':'Virtual_import','error':repr(e)})
print('Virtual package:',virtual_status)
try:VIRTUAL_VERSION=importlib.metadata.version('virtual-parquet')
except Exception:VIRTUAL_VERSION='UNAVAILABLE'

# Frozen modes / full-float-column input parity, following NB93.
MODES=['raw','mask_raw','mask_cents','mask_tri','xor_raw']
EXPECTED_NB93_HOLD={
 'UCI-Bike-275':'710e66352e1d341f53797dc9535f6436589ee99bb7765f979209ee474edf1693',
 'NYC-TLC-Yellow-2024-01':'3d0fc4346b1b4d3457aaa1e9941fc2729c67a76ca405bc1fb6829d7e22f8f651'}
ROWS=[];CAL=[];RAW_BASELINES=[];VIRTUAL_ATTEMPTS=[]
def subset(v,a,b):return {k:v[k][a:b].copy() for k in v}
def bits_identical(original, got, names):
    """Verify raw IEEE-754 bits by *column name*, not Parquet display order.

    A reordered schema is acceptable if all named columns and every row are
    preserved. Missing/extra columns, row reordering, and changes in NaN
    payloads or signed-zero bits are NOT acceptable.
    """
    if not isinstance(got,pd.DataFrame):
        return False,'wrong decoded object type: '+type(got).__name__
    expected=list(names)
    actual_columns=list(got.columns)
    if len(got)!=len(original[expected[0]]):
        return False,f'row count differs: expected={len(original[expected[0]])}, got={len(got)}'
    if got.columns.has_duplicates:
        return False,'duplicate decoded column names: '+repr(actual_columns)
    if len(actual_columns)!=len(expected) or set(actual_columns)!=set(expected):
        missing=[k for k in expected if k not in actual_columns]
        extra=[k for k in actual_columns if k not in expected]
        return False,f'column membership differs; missing={missing!r}; extra={extra!r}; actual={actual_columns!r}'
    for k in expected:
        try:
            actual=np.ascontiguousarray(got[k].to_numpy(dtype='<f8'))
        except Exception as e:
            return False,f'{k}: dtype conversion failed: {e!r}'
        aa=fbits(actual)
        bb=fbits(original[k])
        if not np.array_equal(aa,bb):
            j=int(np.flatnonzero(aa!=bb)[0])
            count=int(np.count_nonzero(aa!=bb))
            return False,(f'{k}: IEEE-754 mismatch count={count}, first_row={j}, '
                          f'expected_bits=0x{int(bb[j]):016x}, got_bits=0x{int(aa[j]):016x}')
    return True,('BIT_EXACT' if actual_columns==expected else 'BIT_EXACT_COLUMNS_REORDERED')

# Local verifier tests: schema reordering must pass; bit corruption must fail.
_verifier_names=['a','b']
_verifier_orig={'a':np.array([0x8000000000000000,0x7ff8000000000001],dtype='<u8').view('<f8'),
                'b':np.array([0x0000000000000000,0x7ff80000000000ab],dtype='<u8').view('<f8')}
_verifier_df=pd.DataFrame({'b':_verifier_orig['b'],'a':_verifier_orig['a']})
assert bits_identical(_verifier_orig,_verifier_df,_verifier_names)[0]
_verifier_bad=_verifier_df.copy()
_verifier_bad['a']=np.array([0x0000000000000000,0x7ff8000000000001],dtype='<u8').view('<f8')
assert not bits_identical(_verifier_orig,_verifier_bad,_verifier_names)[0]
print('PASS: parity validator accepts reordered columns and rejects modified signed-zero bits')

def frame_from_parquet(values,names,where):
    df=pd.DataFrame({k:as_f8(values[k]) for k in names})
    t0=time.perf_counter()
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),str(where),
                   compression='zstd',compression_level=3,use_dictionary=True)
    enc_s=time.perf_counter()-t0
    t1=time.perf_counter();restored=pq.read_table(str(where)).to_pandas();dec_s=time.perf_counter()-t1
    ok,why=bits_identical(values,restored,names)
    return {'bytes':where.stat().st_size,'encode_s':enc_s,'decode_s':dec_s,'bits_verified':ok,
            'reason':why,'artifact':str(where)}

def official_virtual_test(values,names,work_dir):
    # Capture all files left by the official encoder in its isolated workdir;
    # do not assume the .parquet file is the only output required to decode.
    if official_virtual is None:return {'status':'UNAVAILABLE','reason':'virtual-parquet import failed'}
    df=pd.DataFrame({k:as_f8(values[k]) for k in names})
    work_dir.mkdir(exist_ok=True,parents=True)
    original_dir=Path.cwd()
    try:
        os.chdir(work_dir)
        t0=time.perf_counter()
        official_virtual.to_format(df,'virtual_data.parquet')
        enc_s=time.perf_counter()-t0
        p=Path('virtual_data.parquet')
        if not p.exists():raise RuntimeError('Virtual did not create virtual_data.parquet')
        # Avoid importing/calling a substitute; the read must be official API.
        t1=time.perf_counter()
        restored=official_virtual.from_format('virtual_data.parquet')
        dec_s=time.perf_counter()-t1
        decoded_columns=list(restored.columns) if isinstance(restored,pd.DataFrame) else []
        ok,why=bits_identical(values,restored,names)
        print('Virtual decoded columns:',decoded_columns,'verification:',why)
        outputs=[x for x in Path('.').rglob('*') if x.is_file()]
        return {'status':'BIT_EXACT' if ok else 'NOT_BIT_EXACT','reason':why,
                'bytes':sum(x.stat().st_size for x in outputs),
                'decoded_columns':decoded_columns,
                'main_parquet_bytes':p.stat().st_size,
                'side_files':[str(x) for x in outputs if str(x)!='virtual_data.parquet'],
                'encode_s':enc_s,'decode_s':dec_s,'bits_verified':ok,'output_files':len(outputs)}
    except Exception as e:
        return {'status':'UNAVAILABLE_OR_FAILED','reason':repr(e),'bits_verified':False}
    finally:os.chdir(original_dir)

for d in DATA:
    fam=d['family'];sources=d['sources'];target=d['target'];names=sources+[target]
    values=d['values'];n=len(values[target]);cut=max(50,int(n*.25))
    if n-cut<100:
        ERR.append({'family':fam,'stage':'eval','error':'too few holdout values'});continue
    train=subset(values,0,cut);hold=subset(values,cut,n)
    held_sha=table_hash(hold,names);same_as_93=(held_sha==EXPECTED_NB93_HOLD.get(fam))
    if not same_as_93:
        ERR.append({'family':fam,'stage':'NB93_input_match','error':f'holdout SHA mismatch: {held_sha}'})
    try:
        cal_start=time.perf_counter()
        methods={k:frozen_codec(train[k])[0] for k in names}
        cal_scores=[]
        for mode in MODES:
            enc=encode(train,sources,target,mode,methods)
            check=decode(enc)
            assert all(fbits(check[k]).tobytes()==fbits(train[k]).tobytes() for k in names)
            cal_scores.append({'family':fam,'mode':mode,'cal_bytes':len(enc)})
        chosen=min(cal_scores,key=lambda r:r['cal_bytes'])['mode']
        for r in cal_scores:CAL.append({**r,'chosen':r['mode']==chosen})
        cal_seconds=time.perf_counter()-cal_start
        run_start=time.perf_counter();buf=encode(hold,sources,target,chosen,methods);sg_enc=time.perf_counter()-run_start
        run_start=time.perf_counter();sg_dec=decode(buf);sg_dec_s=time.perf_counter()-run_start
        sg_ok=all(fbits(sg_dec[k]).tobytes()==fbits(hold[k]).tobytes() for k in names)
        if not sg_ok:raise RuntimeError('SGX bitwise check failed')
        # Explicit simple Corra-style arithmetic dependency **baseline**, not actual Corra.
        # Compare all known NB93 modes post-test only as ablation/oracle, NOT deployable policy.
        mode_sizes={}
        for mode in MODES:
            tmp=encode(hold,sources,target,mode,methods)
            res=decode(tmp)
            if not all(fbits(res[k]).tobytes()==fbits(hold[k]).tobytes() for k in names):
                raise RuntimeError('NB93 mode bitwise decode failure: '+mode)
            mode_sizes[mode]=len(tmp)
            RAW_BASELINES.append({'family':fam,'mode':mode,'bytes':len(tmp),'source':'NB93 original implemented mode, NOT published Corra'})
        with tempfile.TemporaryDirectory(prefix='sgx95_') as dirname:
            dpath=Path(dirname)
            parq=frame_from_parquet(hold,names,dpath/'heldout_zstd.parquet')
            # distinct directory, no Parquet baseline files counted against Virtual
            virt=official_virtual_test(hold,names,dpath/'virtual')
        if not parq['bits_verified']:
            ERR.append({'family':fam,'stage':'parquet_bitwise','error':parq['reason']})
        if not virt.get('bits_verified'):
            ERR.append({'family':fam,'stage':'virtual_parity','error':virt.get('reason','unavailable')})
        VIRTUAL_ATTEMPTS.append({'family':fam,**virt})
        rec={'family':fam,'n_cal':cut,'n_hold':n-cut,'n_columns':len(names),'heldout_sha256':held_sha,
             'matches_NB93_heldout':same_as_93,'frozen_mode':chosen,'sgx_bytes':len(buf),
             'sgx_verified':sg_ok,'sgx_encode_s':sg_enc,'sgx_decode_s':sg_dec_s,'calibration_seconds':cal_seconds,
             'frozen_source_methods':json.dumps(methods,sort_keys=True),
             'best_known_NB93_mode':min(mode_sizes,key=mode_sizes.get),
             'best_known_NB93_bytes_oracle':min(mode_sizes.values()),
             'parquet_zstd_bytes':parq['bytes'],'parquet_verified':parq['bits_verified'],
             'parquet_encode_s':parq['encode_s'],'parquet_decode_s':parq['decode_s'],
             'virtual_status':virt['status'],'virtual_verified':virt.get('bits_verified',False),
             'virtual_bytes':virt.get('bytes'), 'virtual_encode_s':virt.get('encode_s'),
             'virtual_decode_s':virt.get('decode_s'), 'virtual_reason':virt.get('reason'),
             'virtual_main_parquet_bytes':virt.get('main_parquet_bytes'),
             'virtual_side_files':json.dumps(virt.get('side_files',[])),
             'virtual_decoded_columns':json.dumps(virt.get('decoded_columns',[])),
             'sgx_saving_vs_virtual_pct':(100*(virt['bytes']-len(buf))/virt['bytes']) if virt.get('bits_verified') and virt.get('bytes',0)>0 else None,
             'sgx_saving_vs_parquet_pct':100*(parq['bytes']-len(buf))/parq['bytes'] if parq['bits_verified'] else None,
             'sgx_saving_vs_simple_best_pct':100*(min(mode_sizes.values())-len(buf))/min(mode_sizes.values())}
        ROWS.append(rec)
        print(fam, 'mode',chosen,'SGX',len(buf),'Virtual',virt.get('bytes',virt['status']),
              'Parquet',parq['bytes'],'correct',virt.get('bits_verified'))
    except Exception as e:
        ERR.append({'family':fam,'stage':'experiment','error':repr(e)})
        print(fam,'INCOMPLETE experiment:',repr(e))


"""
## Decision gate and exported results

**Evidence gate:** Both families must load identically to NB93, SGX must decode bitwise, **official** Virtual and Parquet both must decode bitwise, and total bytes must be accounted for. A favorable size comparison is an interesting result only; originality is **not** established. We do not claim a direct comparison to Corra or C3 because their official implementations have not been executed on identical complete inputs. If the Virtual library cannot decode exactly, its size is *not* scored as a valid lossless comparator. The output includes all failures and missing evidence.
"""

def write_csv(name,rows):
    headers=list(dict.fromkeys(k for r in rows for k in r)) if rows else ['status']
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=headers,extrasaction='ignore')
        w.writeheader()
        if rows:w.writerows(rows)
        else:w.writerow({'status':'NO_RECORDS'})
for name,data in [('matched_comparison.csv',ROWS),('calibration_frozen_modes.csv',CAL),
                  ('original_NB93_mode_ablations.csv',RAW_BASELINES),('virtual_attempts.csv',VIRTUAL_ATTEMPTS),
                  ('source_provenance.csv',PROVENANCE),('run_issues.csv',ERR)]:write_csv(name,data)
full=(len(ROWS)==2 and all(x['matches_NB93_heldout'] and x['sgx_verified'] and x['parquet_verified'] and x['virtual_verified'] for x in ROWS) and not ERR)
size_better=bool(full and all(x['sgx_saving_vs_virtual_pct'] is not None and x['sgx_saving_vs_virtual_pct']>=5 for x in ROWS))
strong_simple=bool(full and all(x['sgx_saving_vs_simple_best_pct']>=5 for x in ROWS))
DECISION={'notebook':'NB95F','status':'COMPLETE' if full else 'INCOMPLETE',
          'real_data_families':len(ROWS),'official_virtual_version':VIRTUAL_VERSION,
          'official_virtual_bitwise_for_both':bool(full),
          'SGX_size_5pct_better_than_Virtual_on_both':size_better,
          'SGX_size_5pct_better_than_all_original_NB93_variants_on_both':strong_simple,
          'direct_actual_Corra_benchmark':'NOT_RUN',
          'direct_actual_C3_benchmark':'NOT_RUN',
          'virtual_calibration_frozen':'NO_PUBLIC_API_ASSUMPTION: normal Virtual to_format observes encoded input',
          'article_level_novel_algorithm_established':False,'q1_submission_ready':False,
          'research_verdict':'Still no demonstrated distinct algorithmic rule, even if size gates pass.' if full else 'Prior-art parity incomplete: do not draw performance or novelty conclusions.',
          'important_scope':'Exact decoded selected float64 columns, not source CSV/Parquet bytes; formats differ; development datasets previously examined.',
          'run_issues_count':len(ERR)}
(OUT/'decision.json').write_text(json.dumps(DECISION,indent=2,default=str))
(OUT/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),
 'pandas':pd.__version__,'numpy':np.__version__,'pyarrow':pa.__version__,
 'pcodec_available':pcs is not None,'virtual_version':VIRTUAL_VERSION,'install_log':INSTALL},indent=2,default=str))
result_path=ROOT/'SGX_NB95F_results.zip'
with zipfile.ZipFile(result_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.iterdir()):
        if f.is_file():z.write(f,arcname=f.name)
print(json.dumps(DECISION,indent=2))
print('Download or retrieve:',result_path)
try:
    from google.colab import files
    files.download(str(result_path))
except ImportError:pass

