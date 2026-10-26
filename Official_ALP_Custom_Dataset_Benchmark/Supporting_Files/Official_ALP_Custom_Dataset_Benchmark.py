#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X — Notebook 46: Official_ALP_Custom_Dataset_Benchmark

**Purpose:** Actually attempt official ALP bench_your_dataset on matched, SHA-256-identified UCI Wine Quality float64 columns. ALP native BPV is recorded separately from general-purpose native compressed-file bytes. It is NOT claimed as an end-to-end fair file-size comparison.

All code is standalone. Run **Runtime → Run all** in Google Colab (CPU). Each notebook writes a results ZIP to `/content` and attempts an automatic download. If external ALP dependencies or UCI downloads fail, it exports a diagnostic ZIP rather than inventing scores.

**Research integrity:** A native ALP estimated bits/value figure is *not* directly comparable to an encoded file containing its header; benchmarking alone does not establish novelty or Q1 journal readiness.
"""

import sys,os,io,csv,json,zipfile,time,hashlib,subprocess,shutil,struct,zlib,lzma,platform,traceback,statistics,urllib.request
from pathlib import Path
import numpy as np
ROOT=Path('/content') if Path('/content').exists() else Path('/mnt/data')
OUT=ROOT/'SGX_NB46_outputs';OUT.mkdir(parents=True,exist_ok=True)
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


CONFIG={'repository':'https://github.com/cwida/ALP.git','pinned_commit':'31ca0ed11c93c99d3f5b5c30e01a3e1c3832d3ce','estimated_alp_size_only':True,'official_benchmark_name':'bench_your_dataset','dataset_list':'benchmarks/your_own_dataset.csv'}
repo=ROOT/'SGX_ALP_OFFICIAL_NB46';logs=[];status={'stage':'STARTED','test_mode':TEST_MODE,'alp_benchmark_ran':False,'alp_decode_bitwise_verified':False,'alp_full_container_bytes_available':False,'comparison_accounting':'INCOMPATIBLE: ALP dry estimated BPV vs generic compressed native-file sizes'}
try:
    samples=list(real_streams())
    assert samples,'No real input data'
    data_dir=ROOT/'SGX_NB46_input_columns'; data_dir.mkdir(exist_ok=True)
    manifest=[]; config_rows=[]; generic_rows=[]
    for idx,item in enumerate(samples):
        fp=data_dir/(item['stream']+'.f64');fp.write_bytes(item['raw'])
        manifest.append({k:v for k,v in item.items() if k!='raw'} | {'sha256':sha(item['raw']),'binary_path':str(fp),'source_kind':'REAL_UCI' if not TEST_MODE else 'SKLEARN_SMOKE'})
        config_rows.append({'id':idx,'column_name':item['stream'],'data_type':'double','path':str(fp),'file_type':'binary'})
        for mode in ('zlib6','lzma1','zstd3'):
            try:
                result=bench_encode_decode(mode,item['raw'],2)
                generic_rows.append({'stream':item['stream'],'input_sha256':sha(item['raw']),'n':item['n'],'mode':mode,'bpv':8*result['bytes']/item['n'],'native_file_bytes':result['bytes'],'decode_exact':True})
            except Exception as e:generic_rows.append({'stream':item['stream'],'mode':mode,'error':repr(e)})
    dump_json('input_manifest.json',manifest);dump_csv('generic_native_sizes.csv',generic_rows)
    status.update({'n_streams':len(samples),'streams_provenance':'UCI Wine Quality' if not TEST_MODE else 'LOCAL_SMOKE_ONLY','input_manifest_written':True})
except Exception as e:
    status.update({'stage':'DOWNLOAD_OR_DATA_FAILED','error':repr(e),'traceback':traceback.format_exc()[-5500:]})
if status.get('input_manifest_written') and not TEST_MODE:
    try:
        apt=run(['bash','-lc','apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq clang clang-tidy cmake ninja-build git'],timeout=900);logs.append(apt)
        if not (repo/'.git').exists():logs.append(run(['git','clone','--depth','1',CONFIG['repository'],str(repo)],timeout=300))
        if not (repo/'.git').exists():raise RuntimeError('ALP cloning failed: inspect build_logs.json')
        # Shallow checkout of pin may require fetch; record actual commit and reject mismatch.
        head=run(['git','-C',str(repo),'rev-parse','HEAD']);logs.append(head)
        status['commit']=head.get('stdout_tail','').strip()
        if status['commit']!=CONFIG['pinned_commit']:
            logs.append(run(['git','-C',str(repo),'fetch','--depth','1','origin',CONFIG['pinned_commit']],timeout=180))
            logs.append(run(['git','-C',str(repo),'checkout','--detach',CONFIG['pinned_commit']],timeout=90))
            status['commit']=run(['git','-C',str(repo),'rev-parse','HEAD']).get('stdout_tail','').strip()
        if status['commit']!=CONFIG['pinned_commit']:raise RuntimeError('Pinned ALP commit unavailable; no unpinned benchmark accepted')
        config_path=repo/CONFIG['dataset_list']
        if not config_path.parent.exists():raise RuntimeError('ALP dataset config parent missing')
        with config_path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['id','column_name','data_type','path','file_type']);writer.writeheader();writer.writerows(config_rows)
        (OUT/'submitted_alp_dataset_config.csv').write_bytes(config_path.read_bytes())
        build=repo/'sgx_bench_build'
        cc=shutil.which('clang');cxx=shutil.which('clang++')
        if not cc or not cxx:raise RuntimeError('Clang or Clang++ missing')
        flags=['cmake','-S',str(repo),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DALP_BUILD_BENCHMARKING=ON','-DALP_ENABLE_CLANG_TIDY=OFF','-DCMAKE_C_COMPILER='+cc,'-DCMAKE_CXX_COMPILER='+cxx]
        logs.append(run(flags,timeout=600));status['configure_ok']=logs[-1]['returncode']==0
        if not status['configure_ok']:raise RuntimeError('ALP benchmark CMake configuration failed')
        exe=build/'benchmarks'/'bench_your_dataset'
        logs.append(run(['cmake','--build',str(build),'--target','bench_your_dataset','--parallel','2'],timeout=1200));status['benchmark_target_built']=logs[-1]['returncode']==0
        if not status['benchmark_target_built']:raise RuntimeError('Official ALP custom benchmark target failed')
        if not exe.exists():raise RuntimeError(f'Expected benchmark executable not found: {exe}')
        result_file=repo/'benchmarks'/'your_own_dataset_result.csv'
        if result_file.exists():result_file.unlink()
        logs.append(run([str(exe)],cwd=build,timeout=900));status['benchmark_returncode']=logs[-1]['returncode']
        if result_file.exists():
            (OUT/'alp_native_results.csv').write_bytes(result_file.read_bytes())
            with result_file.open(newline='') as f:alp_rows=list(csv.DictReader(f))
            status['alp_rows']=len(alp_rows)
            status['alp_benchmark_ran']=bool(alp_rows) and logs[-1]['returncode']==0
            dump_json('alp_raw_table_preview.json',alp_rows[:8])
        if not status['alp_benchmark_ran']:raise RuntimeError('No nonempty official ALP benchmark output; inspect logs')
        status['stage']='ALP_NATIVE_BPV_MEASURED_ACCOUNTING_UNMATCHED'
    except Exception as e:status.update({'stage':'ALP_RUN_FAILED','error':repr(e),'traceback':traceback.format_exc()[-5500:]})
elif TEST_MODE:status['stage']='LOCAL_SMOKE_ONLY_NO_ALP'
dump_json('benchmark_status.json',status);dump_json('build_logs.json',logs);dump_json('configuration.json',CONFIG)
print('NB46 STATUS',json.dumps(status,indent=2)[:7000]);result_zip(46)

