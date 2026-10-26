#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX NB21 — Real-data ALP gap forensics

Research diagnostic: downloads public streams independently and profiles temporal repetition, raw representation distributions, and estimated compressibility. It does **not** claim ALP internal residual traces. Upload the NB20 ZIP optionally to attach its official aggregate BPV; absence never becomes fabricated evidence. Synthetic controls are labeled separately. `SGX_NB21_results.zip`.
"""

OUTPUT_NAME='SGX_NB21_outputs'
import os,sys,io,json,csv,time,hashlib,zipfile,urllib.request,pathlib,platform,math,traceback,zlib,lzma,struct
import numpy as np
ROOT=pathlib.Path('/content') if pathlib.Path('/content').exists() else pathlib.Path('/mnt/data')
OUT=ROOT/OUTPUT_NAME;OUT.mkdir(exist_ok=True)
STATUS=[]
def save_csv(name, rows):
  keys=list(dict.fromkeys(k for row in rows for k in row)) if rows else ['status']
  with (OUT/name).open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def record(stage,ok,msg=''):
  STATUS.append({'stage':stage,'ok':bool(ok),'message':str(msg)[:500]});(OUT/'stages.json').write_text(json.dumps(STATUS,indent=2));print(stage,'PASS' if ok else 'FAIL',str(msg)[:100])
def sha(b):return hashlib.sha256(b).hexdigest()
def fetch(url,maxbytes=36_000_000):
  req=urllib.request.Request(url,headers={'User-Agent':'SGX-Research-CPU/1.0'})
  with urllib.request.urlopen(req,timeout=35) as f:
    chunks=[];size=0
    while True:
      b=f.read(1<<20)
      if not b:break
      size+=len(b)
      if size>maxbytes:raise ValueError('Download over size cap')
      chunks.append(b)
  return b''.join(chunks)
def load_real(maxvals=16384):
  res=[];prov=[]
  sources=[('air','https://archive.ics.uci.edu/static/public/360/air+quality.zip',['T','RH','CO(GT)']),('house','https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip',['Global_active_power','Voltage','Global_intensity'])]
  for ds,url,cols in sources:
    try:
      b=fetch(url);zf=zipfile.ZipFile(io.BytesIO(b));filenames=zf.namelist(); ext='.csv' if ds=='air' else '.txt'
      f=next((v for v in filenames if v.lower().endswith(ext)),None)
      if not f:raise RuntimeError('Expected input absent')
      reader=csv.DictReader(io.TextIOWrapper(zf.open(f),encoding='utf-8-sig',errors='replace'),delimiter=';')
      vals={c:[] for c in cols}
      for row in reader:
        for col in cols:
          if len(vals[col])>=maxvals:continue
          try:
            st=(row.get(col) or '').strip().replace(',','.')
            v=float(st)
            if math.isfinite(v) and (ds!='air' or v!=-200):vals[col].append(v)
          except (ValueError,TypeError):pass
        if all(len(v)>=maxvals for v in vals.values()):break
      for col in cols:
        arr=np.asarray(vals[col],dtype='<f8')
        if arr.size<256:continue
        name=ds+'_'+''.join(x if x.isalnum() else '_' for x in col)
        res.append((name,arr,'REAL_UCI'))
      prov.append({'dataset':ds,'url':url,'archive_sha256':sha(b),'status':'READY','columns_found':sum(len(vals[c])>=256 for c in cols)})
      record('download_'+ds,True,f'{len(vals)} parsed')
    except Exception as e:
      prov.append({'dataset':ds,'url':url,'status':'FAILED','reason':repr(e)});record('download_'+ds,False,repr(e))
  save_csv('dataset_provenance.csv',prov)
  return res

def synthetic_controls(n=4096):
  rng=np.random.default_rng(47)
  return [('synthetic_walk',np.cumsum(rng.normal(0,.05,n)).astype('<f8'),'SYNTHETIC'),('synthetic_bits',rng.integers(0,2**64,size=n,dtype=np.uint64).view('<f8'),'SYNTHETIC')]
def export_zip(label):
  (OUT/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'numpy':np.__version__},indent=2))
  manifest={x.name:sha(x.read_bytes()) for x in OUT.iterdir() if x.is_file() and x.name!='sha256_manifest.json'}
  (OUT/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2))
  path=ROOT/(label+'_results.zip')
  with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for x in OUT.iterdir():
      if x.is_file():z.write(x,x.name)
  print('RESULTS ZIP:',path)
  try:
    from google.colab import files
    files.download(str(path))
  except ImportError:pass
  return path


# Reversible transforms on raw 64-bit float representations; no arithmetic rounding.
def transform(b,mode):
  w=np.frombuffer(b,dtype='<u8').copy()
  if mode=='identity':return b
  if mode=='xor':
    o=w.copy();o[1:]=w[1:]^w[:-1];return o.tobytes()
  if mode=='delta':
    o=w.copy();o[1:]=w[1:]-w[:-1];return o.tobytes()
  if mode=='xor_lanes':
    o=w.copy();o[1:]=w[1:]^w[:-1]
    return o.view(np.uint8).reshape(-1,8).T.copy().tobytes()
  if mode=='delta_lanes':
    o=w.copy();o[1:]=w[1:]-w[:-1]
    return o.view(np.uint8).reshape(-1,8).T.copy().tobytes()
  if mode=='lanes':return w.view(np.uint8).reshape(-1,8).T.copy().tobytes()
  raise ValueError(mode)
def untransform(b,mode):
  if mode=='identity':return b
  if mode in ('lanes','xor_lanes','delta_lanes'):
    n=len(b)//8;b=np.frombuffer(b,dtype=np.uint8).reshape(8,n).T.copy().tobytes()
    if mode=='lanes':return b
    mode='xor' if mode=='xor_lanes' else 'delta'
  w=np.frombuffer(b,dtype='<u8').copy()
  for i in range(1,w.size):
    w[i]=np.uint64(int(w[i])^int(w[i-1])) if mode=='xor' else np.uint64((int(w[i])+int(w[i-1]))&((1<<64)-1))
  return w.tobytes()
def compress(b,backend):
  if backend=='zlib6':return zlib.compress(b,6)
  if backend=='lzma1':return lzma.compress(b,preset=1)
  if backend=='raw':return b
  raise ValueError(backend)
def decompress(b,backend):
  if backend=='zlib6':return zlib.decompress(b)
  if backend=='lzma1':return lzma.decompress(b)
  if backend=='raw':return b
  raise ValueError(backend)


series=load_real(16384)+synthetic_controls()
rows=[]
for name,a,origin in series:
  try:
    b=a.tobytes();bits=a.view('<u8');xor=np.zeros(len(bits),dtype=np.uint64);xor[1:]=bits[1:]^bits[:-1]
    runs=float(np.mean(xor[1:]==0)) if len(bits)>1 else 0
    low=np.bitwise_and(xor,np.uint64(255)).astype(np.uint8)
    probs=np.bincount(low,minlength=256).astype(float);probs/=max(1,sum(probs))
    p=probs[probs>0];entropy=float(-(p*np.log2(p)).sum())
    for mode in ['identity','xor','delta','lanes','xor_lanes','delta_lanes']:
      t0=time.perf_counter();tb=transform(b,mode);payload=zlib.compress(tb,6);sec=time.perf_counter()-t0
      ok=untransform(zlib.decompress(payload),mode)==b
      rows.append({'series':name,'origin':origin,'n':len(a),'mode':mode,'encoded_bytes':len(payload),'bpv':8*len(payload)/len(a),'transform_encode_s':sec,'zero_xor_fraction':runs,'low_xor_byte_entropy':entropy,'exact_ok':ok})
    record(name,True,'all 6 transforms')
  except Exception as e:record(name,False,repr(e))
save_csv('gap_forensics.csv',rows)
assert all(r['exact_ok'] for r in rows)


# Optional: inspect previously produced official ALP figures without confusing measurement protocols.
import io
ALP_ZIP=None
if 'google.colab' in sys.modules:
  print('Optional: upload SGX_NB20_Official_ALP_Multidata_results.zip, or skip this cell.')
# Set ALP_ZIP='/content/SGX_NB20_Official_ALP_Multidata_results.zip' when available.
if ALP_ZIP and pathlib.Path(ALP_ZIP).is_file():
  with zipfile.ZipFile(ALP_ZIP) as z:
    (OUT/'prior_alp_summary.csv').write_bytes(z.read('official_alp_parsed.csv'))
    record('prior_alp_import',True,'Historical official BPV attached; not comparable timing')
else:record('prior_alp_import',True,'Optional, not provided')


export_zip('SGX_NB21')

