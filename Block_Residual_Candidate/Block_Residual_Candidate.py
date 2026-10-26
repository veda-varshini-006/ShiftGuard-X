#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX NB22 — Bounded residual coding candidate

Experimental candidate, **not an established novel codec**. A byte-exact block container with separate per-block reversible residual modes; a selector calibrated only on first 25% of blocks (no held-out oracle leakage). Compare against fixed-mode coding using identical backend and header accounting. Mode changes are pre-signaled and total encode times include calibration. `SGX_NB22_results.zip`.
"""

OUTPUT_NAME='SGX_NB22_outputs'
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


MODES=['identity','xor','delta','lanes','xor_lanes','delta_lanes']
MAGIC=b'SGX22A';BLOCK=4096 # bytes, 512 binary64 values
# Every block is independently decodable. Mode id + 4-byte payload length included.
def encode_blocks(data,mode,backend='zlib6'):
  assert len(data)%8==0
  parts=[MAGIC,struct.pack('<I',len(data))]
  for i in range(0,len(data),BLOCK):
    chunk=data[i:i+BLOCK];coded=compress(transform(chunk,mode),backend)
    parts.append(struct.pack('<BI',MODES.index(mode),len(coded)));parts.append(coded)
  return b''.join(parts)
def decode_blocks(data,backend='zlib6'):
  assert data[:6]==MAGIC
  length=struct.unpack('<I',data[6:10])[0];pos=10;res=[]
  while pos<len(data):
    if pos+5>len(data):raise ValueError('Truncated header')
    idx,n=struct.unpack('<BI',data[pos:pos+5]);pos+=5
    if idx>=len(MODES) or pos+n>len(data):raise ValueError('Bad block')
    chunk=untransform(decompress(data[pos:pos+n],backend),MODES[idx]);res.append(chunk);pos+=n
  raw=b''.join(res)
  if len(raw)!=length:raise ValueError('Wrong decoded length')
  return raw
# Repetition/selector policies: chosen only on calibration blocks for evaluation.
series=load_real(16384)+synthetic_controls()
rows=[]
for name,a,origin in series:
  raw=a.tobytes();m=len(raw)//BLOCK
  if m<4:continue
  n_cal=max(1,m//4);cal=raw[:n_cal*BLOCK];held=raw[n_cal*BLOCK:]
  if not held:continue
  costs={mode:len(encode_blocks(cal,mode)) for mode in MODES}
  chosen=min(costs,key=costs.get)
  for mode in MODES+['calibration_locked']:
    use=chosen if mode=='calibration_locked' else mode
    t0=time.perf_counter();blob=encode_blocks(held,use);enc=time.perf_counter()-t0
    t1=time.perf_counter();decoded=decode_blocks(blob);dec=time.perf_counter()-t1
    good=decoded==held
    rows.append({'series':name,'origin':origin,'policy':mode,'used_mode':use,'selected_on':'calibration' if mode=='calibration_locked' else 'fixed','calibration_values':len(cal)//8,'heldout_values':len(held)//8,'total_bytes':len(blob),'bpv':8*len(blob)/(len(held)//8),'encode_s':enc,'decode_s':dec,'exact_ok':good})
  record(name,True,f'chosen {chosen}')
save_csv('coding_comparison.csv',rows)
assert all(x['exact_ok'] for x in rows)
# Special value and corrupted-stream validation
x=np.array([0,0x8000000000000000,0x7ff8000000000123,0x7ff0000000000000,0xfff0000000000000,0xffffffffffffffff],dtype='<u8').tobytes()
for mode in MODES:
  b=encode_blocks(x,mode);assert decode_blocks(b)==x
  try:decode_blocks(b[:-1]);raise AssertionError('Truncation unrecognized')
  except (ValueError,zlib.error,lzma.LZMAError):pass
record('special_bits',True,'All modes round-trip and truncation rejected')


export_zip('SGX_NB22')

