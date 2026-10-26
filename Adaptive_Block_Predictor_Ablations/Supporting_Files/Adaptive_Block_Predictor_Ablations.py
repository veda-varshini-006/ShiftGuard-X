#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X — Notebook 47: Adaptive_Block_Predictor_Ablations

**Purpose:** Test a falsifiable, fully reversible per-block predictor bank and compare with Zlib/LZMA/Zstandard and ablations on identical UCI input bytes; include selection/tag metadata and encoder selection cost.

All code is standalone. Run **Runtime → Run all** in Google Colab (CPU). Each notebook writes a results ZIP to `/content` and attempts an automatic download. If external ALP dependencies or UCI downloads fail, it exports a diagnostic ZIP rather than inventing scores.

**Research integrity:** A native ALP estimated bits/value figure is *not* directly comparable to an encoded file containing its header; benchmarking alone does not establish novelty or Q1 journal readiness.
"""

import sys,os,io,csv,json,zipfile,time,hashlib,subprocess,shutil,struct,zlib,lzma,platform,traceback,statistics,urllib.request
from pathlib import Path
import numpy as np
ROOT=Path('/content') if Path('/content').exists() else Path('/mnt/data')
OUT=ROOT/'SGX_NB47_outputs';OUT.mkdir(parents=True,exist_ok=True)
TEST_MODE = os.environ.get('SGX_LOCAL_SMOKE_ONLY','0') == '1' # leave FALSE in Colab

def dump_json(name,x): (OUT/name).write_text(json.dumps(x,indent=2,default=str))
def dump_csv(name,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for row in rows for k in row.keys()))
    with (OUT/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def result_zip(no):
    path=ROOT/f'SGX_NB{no}_results.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob('*')):
            if f.is_file(): z.write(f,f.relative_to(OUT))
    print('RESULTS ZIP',path,'bytes',path.stat().st_size)
    try:
        from google.colab import files
        files.download(str(path))
    except ImportError: pass
    except Exception as e: print('Manual download:',path,e)
    return path
def run(args,timeout=360,cwd=None,env=None):
    t=time.monotonic()
    try:
        p=subprocess.run(args,cwd=cwd,env=env,text=True,capture_output=True,timeout=timeout,check=False)
        return {'cmd':args,'returncode':p.returncode,'seconds':round(time.monotonic()-t,2),'stdout_tail':p.stdout[-12000:],'stderr_tail':p.stderr[-12000:]}
    except Exception as e:return {'cmd':args,'returncode':-1,'seconds':round(time.monotonic()-t,2),'exception':repr(e)}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def stream_bytes(a):return np.ascontiguousarray(a,dtype='<f8').tobytes()


def real_streams():
    # UCI Wine Quality: external real observations; no zero-padding or imputation.
    # For official ALP bench vector alignment, use exact prefix multiple of 1024.
    # This discards an explicitly recorded tail and is applied to ALL codecs.
    if TEST_MODE:
        from sklearn.datasets import load_digits,load_breast_cancer
        for name,d,cols in [('digits_fixture',load_digits().data,[0,10,25,50]),('breast_fixture',load_breast_cancer().data,[0,2,10,20])]:
            for j in cols:
                x=np.asarray(d[:,j],dtype='<f8')
                yield {'dataset':name,'stream':f'{name}__c{j:02d}','source_url':'sklearn local smoke only','original_n':len(x),'n':len(x),'truncated_n':0,'raw':stream_bytes(x),'test_mode':True}
        return
    locations={
      'red_wine':'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
      'white_wine':'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'}
    for dataset,url in locations.items():
        with urllib.request.urlopen(url,timeout=75) as response:
            raw_csv=response.read().decode('utf-8-sig')
        reader=csv.DictReader(io.StringIO(raw_csv),delimiter=';')
        records=list(reader)
        assert records and len(records)>=1024,'Not enough rows: '+dataset
        cols=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
        length=(len(records)//1024)*1024
        for j,column in enumerate(cols):
            values=np.asarray([float(r[column]) for r in records[:length]],dtype='<f8')
            stream=f'{dataset}__c{j:02d}'
            yield {'dataset':dataset,'stream':stream,'source_url':url,'column':column,'original_n':len(records),'n':length,'truncated_n':len(records)-length,'raw':stream_bytes(values),'test_mode':False}


TAG={0:'zlib',1:'shuffle_zlib',2:'xor1_zlib',3:'xor2_zlib',4:'xor4_zlib',5:'xor8_zlib'}
BS=256

def do_xor(raw,lag):
    a=np.frombuffer(raw,dtype='<u8'); v=a.copy()
    if len(a)>lag: v[lag:]=a[lag:] ^ a[:-lag]
    return v.astype('<u8').tobytes()
def undo_xor(raw,lag):
    a=np.frombuffer(raw,dtype='<u8').copy()
    for i in range(lag,len(a)):a[i]^=a[i-lag]
    return a.astype('<u8').tobytes()
def shuf(raw):
    return np.frombuffer(raw,dtype=np.uint8).reshape(-1,8).T.copy().tobytes()
def unshuf(raw):
    return np.frombuffer(raw,dtype=np.uint8).reshape(8,-1).T.copy().tobytes()
def do_block(raw,tag):
    if tag==0:return zlib.compress(raw,6)
    if tag==1:return zlib.compress(shuf(raw),6)
    lag={2:1,3:2,4:4,5:8}[tag]
    return zlib.compress(do_xor(raw,lag),6)
def undo_block(blob,tag):
    raw=zlib.decompress(blob)
    if tag==0:return raw
    if tag==1:return unshuf(raw)
    return undo_xor(raw,{2:1,3:2,4:4,5:8}[tag])

def sgx_encode(raw,enabled_tags=(0,1,2,3,4,5),block_size=BS):
    if len(raw)%8:raise ValueError('Input not float64-aligned')
    n=len(raw)//8
    blocks=(n+block_size-1)//block_size
    out=bytearray(b'SGX8'+struct.pack('<QHI',n,block_size,blocks))
    modes=[]
    for i in range(blocks):
        b=raw[i*block_size*8:min((i+1)*block_size*8,len(raw))]
        variants=[(len(payload),tag,payload) for tag in enabled_tags for payload in [do_block(b,tag)]]
        _,tag,payload=min(variants,key=lambda r:(r[0],r[1]))
        out.extend(struct.pack('<BI',tag,len(payload)));out.extend(payload);modes.append(tag)
    return bytes(out),modes

def sgx_decode(blob):
    if len(blob)<18 or blob[:4]!=b'SGX8':raise ValueError('Bad header')
    n,bs,nblock=struct.unpack('<QHI',blob[4:18]);p=18
    if bs==0 or nblock!=(n+bs-1)//bs:raise ValueError('Bad block count')
    pieces=[]
    for i in range(nblock):
        if p+5>len(blob):raise ValueError('Missing block header')
        tag,size=struct.unpack('<BI',blob[p:p+5]);p+=5
        if tag not in TAG or p+size>len(blob):raise ValueError('Invalid block')
        raw=undo_block(blob[p:p+size],tag);p+=size
        expected=min(bs,n-i*bs)*8
        if len(raw)!=expected:raise ValueError('Bad block length')
        pieces.append(raw)
    if p!=len(blob):raise ValueError('Trailing encoded data')
    out=b''.join(pieces)
    if len(out)!=n*8:raise ValueError('Size mismatch')
    return out

def bench_encode_decode(name,raw,reps=3):
    measures=[];selection=[]
    for _ in range(reps):
        t=time.perf_counter_ns()
        if name=='sgx_adaptive': buf,selection=sgx_encode(raw)
        elif name=='sgx_zlib_only': buf,selection=sgx_encode(raw,(0,))
        elif name=='sgx_shuffle_only': buf,selection=sgx_encode(raw,(1,))
        elif name=='sgx_xor1_only': buf,selection=sgx_encode(raw,(2,))
        elif name=='zlib6':buf=zlib.compress(raw,6)
        elif name=='lzma1':buf=lzma.compress(raw,preset=1)
        elif name=='zstd3':
            import zstandard as zstd
            buf=zstd.ZstdCompressor(level=3).compress(raw)
        else:raise ValueError(name)
        enc_ms=(time.perf_counter_ns()-t)/1e6
        t=time.perf_counter_ns()
        if name.startswith('sgx_'):decoded=sgx_decode(buf)
        elif name=='zlib6':decoded=zlib.decompress(buf)
        elif name=='lzma1':decoded=lzma.decompress(buf)
        else:
            import zstandard as zstd
            decoded=zstd.ZstdDecompressor().decompress(buf)
        dec_ms=(time.perf_counter_ns()-t)/1e6
        assert decoded==raw,f'BIT-EXACT VIOLATION: {name}'
        measures.append((len(buf),enc_ms,dec_ms))
    return {'bytes':measures[0][0],'enc_ms':float(statistics.median(x[1] for x in measures)),'dec_ms':float(statistics.median(x[2] for x in measures)),'mode_counts':{TAG[t]:selection.count(t) for t in sorted(set(selection))} if name=='sgx_adaptive' else {}}

def edge_tests():
    bits=np.array([0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,0x0000000000000001,0xffffffffffffffff],dtype='<u8')
    raw=bits.tobytes()*100
    for m in ['sgx_adaptive','sgx_zlib_only','sgx_shuffle_only','sgx_xor1_only']:
        assert bench_encode_decode(m,raw,1)['bytes']>0
    return 'PASS (0,-0,Inf,-Inf,NaN payload,subnormal,all-ones)'


status={'dataset_source':'UCI Wine Quality' if not TEST_MODE else 'LOCAL_SMOKE_ONLY','special_float_edge_checks':None,'completion':'STARTED','novelty':'NOT ESTABLISHED; per-block predictor bank, XOR and shuffle have substantial prior art','method':'per-block minimum-length selection among 6 reversible zlib transforms; selection cost paid at encoding','size_accounting':'SGX mode tags, block lengths, stream header INCLUDED; generic native library format headers INCLUDED'}
rows=[];failures=[];choices=[]
try:
    status['special_float_edge_checks']=edge_tests()
    methods=['zlib6','lzma1','sgx_adaptive','sgx_zlib_only','sgx_shuffle_only','sgx_xor1_only']
    try:import zstandard;methods.append('zstd3')
    except ImportError:pass
    for item in real_streams():
        raw=item['raw'];h=sha(raw)
        for mode in methods:
            try:
                met=bench_encode_decode(mode,raw,3)
                rows.append({'stream':item['stream'],'dataset':item['dataset'],'n':item['n'],'source_original_n':item['original_n'],'truncated_n':item['truncated_n'],'input_sha256':h,'mode':mode,'size_bytes':met['bytes'],'bpv':8*met['bytes']/item['n'],'encoding_ms_median':met['enc_ms'],'decoding_ms_median':met['dec_ms'],'bit_exact':True,'smoke_only':TEST_MODE})
                if mode=='sgx_adaptive':choices.append({'stream':item['stream'],'mode_counts':met['mode_counts']})
            except Exception as e:failures.append({'stream':item['stream'],'mode':mode,'error':repr(e)})
    if not rows:raise RuntimeError('No benchmark rows produced')
    status.update({'completion':'COMPLETE' if not failures else 'PARTIAL','number_of_streams':len({r['stream'] for r in rows}),'rows':len(rows),'failures':len(failures)})
except Exception as e:status.update({'completion':'ERROR','error':repr(e),'traceback':traceback.format_exc()[-5000:]})
dump_csv('candidate_ablation.csv',rows);dump_csv('errors.csv',failures);dump_json('selector_counts.json',choices);dump_json('status.json',status)
print(json.dumps(status,indent=2));result_zip(47)

