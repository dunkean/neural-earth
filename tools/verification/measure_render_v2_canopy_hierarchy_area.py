# Geometric diagnostic only: unit crowns, not generated terrain acceptance.
import json,math
import numpy as np
from pathlib import Path
MASK=0xffffffff
def hash3(x,y,z,seed):
 h=((x*0x9e3779b9)^(y*0x85ebca6b)^(z*0xc2b2ae35)^seed)&MASK
 h=((h^(h>>16))*0x7feb352d)&MASK;h=((h^(h>>15))*0x846ca68b)&MASK
 return (h^(h>>16))&MASK
def u(h,s):return (((h>>s)&255)+.5)/256
def direction(h):
 d=np.array([u(h,0)*2-1+.013,u(h,8)*2-1-.019]);return d/np.linalg.norm(d)
def lobes(id,needle):
 result=[]
 for i in range(9):
  parent=i//3;child=i%3;hp=hash3(parent,71,13,id^0xa511e9b3);hc=hash3(i,43,97,id^0x63d83595)
  d=direction(hp);side=np.array([-d[1],d[0]])
  if needle:
   tier=parent;branch=direction(hc);size=(.60-.17*tier)*(.92+.16*u(hc,16));centre=(.30-.075*tier)*branch;axes=size*np.array([1.05+.10*u(hp,16),.89+.10*u(hp,24)]);rotation=branch
  else:
   along=.27+.12*u(hp,16);lateral=(child-1)*.23+.09*(u(hc,0)-.5);centre=d*(along+.12*(u(hc,8)-.5))+side*lateral;size=.51+.12*u(hc,16);axes=size*np.array([1.03+.12*u(hp,24),.88+.12*u(hc,24)]);rotation=d
  horizontalScale=1.261 if needle else 1.096
  result.append((centre*horizontalScale,axes*horizontalScale,rotation))
 return result
N=256;span=3.2;grid=(np.arange(N)+.5)*span/N-span/2;xx,yy=np.meshgrid(grid,grid);r=np.hypot(xx,yy);dx=xx/np.maximum(r,1e-12);dy=yy/np.maximum(r,1e-12)
results=[]
for needle in [False,True]:
 samples={fp:[] for fp in [0,.03,.15,.6]}
 for k in range(64):
  id=hash3(k,13,79,0x236823);h=id*0x9e3779b9&MASK;a=u(h,0)*2*math.pi;b=u(h,8)*2*math.pi;amp=.07 if needle else .13
  oldradius=1+amp*((dx*dx-dy*dy)*math.cos(a)+2*dx*dy*math.sin(a)+.6*dx*(4*dx*dx-3)*math.cos(b)+.6*dy*(3-4*dy*dy)*math.sin(b))
  distances=[]
  for centre,axes,rot in lobes(id,needle):
   deltaX=xx-centre[0];deltaY=yy-centre[1];qx=(rot[0]*deltaX+rot[1]*deltaY)/axes[0];qy=(-rot[1]*deltaX+rot[0]*deltaY)/axes[1]
   distances.append((1-np.hypot(qx,qy))*min(axes))
  sd=np.maximum.reduce(distances)
  for fp in samples:
   aa=max(fp*.75,1e-8);old=np.clip((oldradius-r)/aa+.5,0,1).sum()*(span/N)**2;new=np.clip(sd/aa+.5,0,1).sum()*(span/N)**2;samples[fp].append([old,new,new/old])
 for fp,values in samples.items():
  v=np.array(values);results.append(dict(genus='needle' if needle else 'broadleaf',footprintOverCrownRadius=fp,crowns=64,meanOldArea=float(v[:,0].mean()),meanNewArea=float(v[:,1].mean()),meanRatio=float(v[:,2].mean()),ratioP10=float(np.percentile(v[:,2],10)),ratioP90=float(np.percentile(v[:,2],90))))
report=dict(kind='CPU geometric diagnostic of64isolated unit crowns per genus; not real-terrain acceptance or population/Tau calibration',results=results)
p=Path('output/render-v2-floor/canopy-hierarchy-causal');p.mkdir(parents=True,exist_ok=True);(p/'unit-crown-area.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
