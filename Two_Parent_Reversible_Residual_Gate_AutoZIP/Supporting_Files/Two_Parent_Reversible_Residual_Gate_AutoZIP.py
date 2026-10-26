#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB115 — Two-parent reversible residual coding

**Question:** Does a decoder-exact two-parent modular predictor beat the strongest *matched* one-parent transform codec on an unseen synthetic split, counting descriptors and all compressed payloads?

**Scientific status:** No novelty is claimed. Two-parent linear predictors, integer modular residuals, and blockwise minimum-byte selection are known concepts. Synthetic gain would justify only real-data and prior-art tests.

No previous ZIPs or uploads are necessary. Runs in Google Colab, then automatically downloads the evidence ZIP.
"""

"""
### Install/import compressor and generate deterministic synthetic data
"""

import io,sys,subprocess,struct,csv,json,hashlib,zipfile,time
from pathlib import Path
import numpy as np
try:
 import zstandard as zstd
except ImportError:
 subprocess.check_call([sys.executable,'-m','pip','-q','install','zstandard'])
 import zstandard as zstd
from google.colab import files
np.seterr(over='ignore')
N=90000; rng=np.random.default_rng(1142026)
a=rng.integers(0,100000,N,dtype=np.int64); bb=rng.integers(0,100000,N,dtype=np.int64);c=rng.integers(1,100,N,dtype=np.int64)
v=np.vstack([a.astype(float),bb.astype(float),(a+bb).astype(float),(a-bb).astype(float),(a*2+bb).astype(float),np.round(a/100,2),np.round(bb/100,2),np.round((a+bb)/100,2),np.round((a+bb+c)/100,2)])
WORDS=v.view(np.uint64).copy()
special=np.array([0,0x8000000000000000,1,0x7ff0000000000000,0x7ff8000000000042],dtype=np.uint64)
for j,x in enumerate(special):WORDS[j,1000+j]=x
print('Dataset',WORDS.shape,'SHA256',hashlib.sha256(WORDS.tobytes()).hexdigest())


"""
### Build self-contained reversible codec (strict decoder; no originals at decode time)
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
### Adversarial bitwise inversion checks, including IEEE specials
"""

rr=np.random.default_rng(115);test=rr.integers(0,2**64,(9,127),dtype=np.uint64)
test[:,:9]=np.array([0,0x8000000000000000,1,0x8000000000000001,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000001234,0x7ff0000000001234,0xfff8000000007777],dtype=np.uint64)
for enabled in [False,True]:
 enc=encode(test,enabled,32);assert np.array_equal(decode(enc),test)
 try:decode(enc[:-1]);raise AssertionError('Truncated blob should fail')
 except (ValueError,Exception) as ex:
  if isinstance(ex,AssertionError):raise
 print('arbitrary IEEE bits pass',enabled,'bytes',len(enc))


"""
### Fair same-codec comparison: one-parent vs two-parent with complete byte accounting
"""

OUT=Path('/content/SGX_NB115_outputs');OUT.mkdir(exist_ok=True)
records=[]; selections={}
for name,m in [('calibration',WORDS[:,:27000]),('heldout',WORDS[:,27000:])]:
 basic=encode(m,False);two=encode(m,True);ref=C.compress(m.astype('<u8',copy=False).tobytes())
 assert np.array_equal(decode(basic),m) and np.array_equal(decode(two),m)
 (OUT/f'{name}_single.sgx115').write_bytes(basic)
 (OUT/f'{name}_two.sgx115').write_bytes(two)
 (OUT/f'{name}_whole_zstd.bin').write_bytes(ref)
 record={'partition':name,'rows':int(m.shape[1]),'one_parent_bytes':len(basic),'two_parent_bytes':len(two),'whole_matrix_zstd_bytes':len(ref),'two_vs_one_gain_percent':round(100*(1-len(two)/len(basic)),4),'two_vs_zstd_gain_percent':round(100*(1-len(two)/len(ref)),4),'all_bits_exact':True}
 records.append(record);print(record)
with (OUT/'comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,records[0].keys());w.writeheader();w.writerows(records)


"""
### Make conservative novelty decision and automatically export a ZIP
"""

gain=records[-1]['two_parent_bytes']<records[-1]['one_parent_bytes']
decision={'candidate':'two-parent modular sum predictor within reversible word-codec','data':'generated synthetic; same RNG as NB114, seed 1142026','strict_decoder_all_bits_exact':True,'heldout_two_parent_beats_same_codec_one_parent':gain,'novelty_proven':False,'full_prior_art_comparison':'NOT DONE','independent_real_data':'NOT DONE','interpretation':'This tests compression feasibility only. Multi-parent residual coding and exact modular inversions have prior art. Do not assert novelty.'}
(OUT/'decision.json').write_text(json.dumps(decision,indent=2))
(OUT/'limitations.txt').write_text('Synthetic numerical dependencies are deliberately constructed and favor transform coding.\nThe one-parent and two-parent encoders have identical framing.\nWhole-matrix Zstd uses the same compressor level but a different storage layout.\nTwo-source modular residuals are not intrinsically a novel algorithm.\n')
zip_path=Path('/content/SGX_NB115_results.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(OUT.iterdir()):z.write(f,f.name)
with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None
print('Verified',zip_path,'size',zip_path.stat().st_size);files.download(str(zip_path))

