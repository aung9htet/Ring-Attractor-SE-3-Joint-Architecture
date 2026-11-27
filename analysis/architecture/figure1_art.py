"""Shared geometry and population glyphs for the architecture illustrations.

All geometric and neural glyphs are schematic. Geometry and corrective shares
come from the same declared TIAGo pose pair; no new experimental result is made.
"""
from pathlib import Path
from html import escape
import math,json,sys,base64
import numpy as np

from .snapshot import Q_START,Q_GOAL,snapshot
from ..common import DATA,FIGURES,ensure_dirs
from reaching_benchmark.geometry.chains import get_chain,workspace
from reaching_benchmark.geometry.kinematics import joint_origins,position,joint_axes,moves
from reaching_benchmark.geometry.rotations import turn_between
from reaching_benchmark.limits import limit_room
from reaching_benchmark.task import IkDesign
from ring_attractor_kinematics.rules.cascade_rule import pose_cascade,levers,share_caps

STATE='#087F8C';GOAL='#BB7D12';CORR='#7748B4';INK='#193747';GREY='#8295A2';PALE='#E4EBEF';SOFT='#F4F7F9'
W=1000;H=702

def mix(c,t=0.5,to='#FFFFFF'):
 a=[int(c[i:i+2],16) for i in (1,3,5)];b=[int(to[i:i+2],16) for i in (1,3,5)]
 return '#'+''.join('%02x'%round((1-t)*x+t*y) for x,y in zip(a,b))
def f(x):return ('%.3f'%float(x)).rstrip('0').rstrip('.')

class Art:
 def __init__(self):
  self.parts=['<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="178mm" height="125mm" viewBox="0 0 1000 702" font-family="DejaVu Sans, Arial, sans-serif">',self.definitions()]
  self.rect(0,0,W,H,'#FFFFFF')
 def definitions(self):
  out=['<defs>']
  for k,c in [('state',STATE),('goal',GOAL),('corr',CORR),('grey',GREY)]:
   out.append('<marker id="head-%s" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10L2.5 5Z" fill="%s"/></marker>'%(k,c))
   out.append('<radialGradient id="glow-%s"><stop offset="0" stop-color="%s" stop-opacity=".55"/><stop offset="1" stop-color="%s" stop-opacity="0"/></radialGradient>'%(k,c,c))
  out.append('<linearGradient id="metal"><stop stop-color="#B8C6D0"/><stop offset=".48" stop-color="#F9FBFC"/><stop offset="1" stop-color="#D1DCE4"/></linearGradient>')
  out.append('</defs>');return ''.join(out)
 def add(self,s):self.parts.append(s)
 def group(self,transform=None,opacity=None):
  a=' transform="%s"'%transform if transform else ''
  if opacity is not None:a+=' opacity="%s"'%f(opacity)
  self.add('<g'+a+'>')
 def end(self):self.add('</g>')
 def rect(self,x,y,w,h,fill='none',stroke=None,lw=1,rx=0):
  self.add('<rect x="%s" y="%s" width="%s" height="%s" rx="%s" fill="%s"%s/>'%(f(x),f(y),f(w),f(h),f(rx),fill,'' if not stroke else ' stroke="%s" stroke-width="%s"'%(stroke,f(lw))))
 def circle(self,x,y,r,fill='none',stroke=None,lw=1):
  self.add('<circle cx="%s" cy="%s" r="%s" fill="%s"%s/>'%(f(x),f(y),f(r),fill,'' if not stroke else ' stroke="%s" stroke-width="%s"'%(stroke,f(lw))))
 def ellipse(self,x,y,rx,ry,fill='none',stroke=None,lw=1):
  self.add('<ellipse cx="%s" cy="%s" rx="%s" ry="%s" fill="%s"%s/>'%(f(x),f(y),f(rx),f(ry),fill,'' if not stroke else ' stroke="%s" stroke-width="%s"'%(stroke,f(lw))))
 def path(self,d,color=GREY,lw=1,fill='none',dash=None,head=None):
  a=' stroke-dasharray="%s"'%dash if dash else ''
  if head:a+=' marker-end="url(#head-%s)"'%head
  self.add('<path d="%s" fill="%s" stroke="%s" stroke-width="%s" stroke-linecap="round" stroke-linejoin="round"%s/>'%(d,fill,color,f(lw),a))
 def line(self,x,y,xx,yy,color=GREY,lw=1,dash=None,head=None):self.path('M%s %sL%s %s'%(f(x),f(y),f(xx),f(yy)),color,lw,dash=dash,head=head)
 def poly(self,points,fill='none',stroke=GREY,lw=1):
  self.add('<polygon points="%s" fill="%s" stroke="%s" stroke-width="%s"/>'%(' '.join(f(x)+','+f(y) for x,y in points),fill,stroke,f(lw)))
 def text(self,x,y,s,size=13,color=INK,anchor='start',weight='normal',italic=False):
  self.add('<text x="%s" y="%s" font-size="%s" fill="%s" text-anchor="%s" font-weight="%s"%s>%s</text>'%(f(x),f(y),f(size),color,anchor,weight,' font-style="italic"' if italic else '',escape(str(s))))
 def heading(self,x,y,letter,title):
  self.text(x,y,letter,size=17,weight='bold');self.text(x+25,y,title,size=15.3,weight='bold')
 def save(self,path):Path(path).write_text('\n'.join(self.parts+['</svg>']))

def projected(v):
 x,y,z=v;az=math.radians(-40);el=math.radians(20)
 depth=-x*math.sin(az)+y*math.cos(az)
 return np.array([x*math.cos(az)+y*math.sin(az),-(z*math.cos(el)+depth*math.sin(el))])

def get_data():
 chain=get_chain('tiago7');q=np.array(Q_START);goal=np.array(Q_GOAL);past=chain.clip(q-1.5*(goal-q))
 s=snapshot();design=IkDesign(task='pose');space=workspace(chain)
 e=turn_between(chain.rotation(q),chain.rotation(goal));d=(position(chain,goal)-position(chain,q))/space.radius
 a=joint_axes(chain,q);m=moves(chain,q);caps=share_caps(chain,design)
 shares,rests=pose_cascade(a,m,e,d,design,levers(chain),caps,limit_room(chain,q,design))
 s.update(q=q,q_goal=goal,q_past=past,chain=chain,shares_rad=shares,rests=rests,caps=caps,motor_joint=int(np.argmax(np.abs(s['leads']))))
 return s

def fit_robot(s,rect):
 x,y,w,h=rect;chain=s['chain']
 points=[]
 for q in [s['q'],s['q_goal'],s['q_past']]:points.extend(joint_origins(chain,q));points.append(position(chain,q))
 points.extend([[0,0,-.51],[0,0,.03]])
 v=np.array([projected(p) for p in points]);lo=v.min(axis=0);hi=v.max(axis=0)
 scale=min((w-36)/(hi[0]-lo[0]),(h-40)/(hi[1]-lo[1]))
 offset=np.array([x+w/2,y+h/2])-.5*(lo+hi)*scale
 return lambda p:projected(p)*scale+offset,scale

def arc(cv,cx,cy,r,a0,a1,color=GREY,lw=1.3,dash=None,head=None):
 # angles are degrees; positive angles follow the page's clockwise direction
 x=cx+r*math.cos(math.radians(a0));y=cy+r*math.sin(math.radians(a0));xx=cx+r*math.cos(math.radians(a1));yy=cy+r*math.sin(math.radians(a1))
 large=1 if abs(a1-a0)>180 else 0;sweep=1 if a1>a0 else 0
 cv.path('M%s %sA%s %s 0 %d %d %s %s'%(f(x),f(y),f(r),f(r),large,sweep,f(xx),f(yy)),color,lw,dash=dash,head=head)

def ring(cv,cx,cy,r,index,color=STATE,bounded=True,recurrent=True,label=None,reference=None):
 cv.circle(cx,cy,r,stroke=mix(color,.76),lw=.8)
 if bounded:
  arc(cv,cx,cy,r+5,-36,216,mix(color,.48),1.8)
  for a in [-36,216]:
   a=math.radians(a);cv.line(cx+(r+2)*math.cos(a),cy+(r+2)*math.sin(a),cx+(r+9)*math.cos(a),cy+(r+9)*math.sin(a),GREY,1)
  arc(cv,cx,cy,r+5,217,323,PALE,1,dash='2 3')
 for i in range(64):
  n=200*(i+.5)/64;dist=(n-index+100)%200-100;amp=math.exp(-.5*(dist/7)**2)
  a=math.radians(-90+360*n/200);xx=cx+r*math.cos(a);yy=cy+r*math.sin(a)
  cv.circle(xx,yy,1.5+1.5*amp,fill=mix(color,.78*(1-amp)),stroke=mix(color,.66) if amp<.02 else None,lw=.35)
 if recurrent:arc(cv,cx,cy,r*.68,25,300,color,.9,head='state' if color==STATE else 'corr')
 else:arc(cv,cx,cy,r*.7,0,350,mix(color,.7),.6,dash='1 3')
 if reference is not None:
  a=math.radians(-90+360*reference/200)
  cv.line(cx+(r-7)*math.cos(a),cy+(r-7)*math.sin(a),cx+(r+1)*math.cos(a),cy+(r+1)*math.sin(a),STATE,1.5)
 if label:cv.text(cx,cy+4,label,size=13,color=color,anchor='middle',italic=True)

def triad(cv,cx,cy,R,length,color,labels=False):
 for i,l in enumerate('xyz'):
  v=projected(np.asarray(R)[:,i])*length
  cv.line(cx,cy,cx+v[0],cy+v[1],color,1.6)
  cv.circle(cx+v[0],cy+v[1],1.8,color)
  if labels:cv.text(cx+v[0]+4,cy+v[1]+2,l,size=12,color=color)
 cv.circle(cx,cy,2.2,fill='#FFFFFF',stroke=color,lw=1.3)

def robot(cv,s,rect,numbers=True):
 P,scale=fit_robot(s,rect);chain=s['chain']
 # Cast shadow and a restrained cylindrical support carry the arm in space.
 bx,by=P([0,0,-.49]);tx,ty=P([0,0,.02]);rw=27
 cv.ellipse(bx,by+14,59,12,'#EDF1F4')
 cv.path('M%s %sL%s %sL%s %sL%s %sZ'%(f(tx-rw),f(ty),f(bx-rw),f(by),f(bx+rw),f(by),f(tx+rw),f(ty)),GREY,1.1,'url(#metal)')
 cv.ellipse(tx,ty,rw,8,'#F7FAFB',GREY,1)
 cv.ellipse(bx,by,rw,8,'#D9E3EA',GREY,1)
 cv.line(tx-12,ty+9,bx-12,by-7,'#FFFFFF',2)
 def chain_points(q):return [np.array([0.,0.,-.151])]+list(joint_origins(chain,q))+[position(chain,q)]
 # A geometric earlier posture and desired posture are deliberately schematic.
 for q,col,op,dash in [(s['q_past'],GREY,.20,None),(s['q_goal'],GOAL,.28,'4 5')]:
  cv.group(opacity=op);p=[P(v) for v in chain_points(q)]
  for k,(a,b) in enumerate(zip(p[:-1],p[1:])):
   cv.line(*a,*b,color=col,lw=max(8,24-2*k),dash=dash)
  cv.end()
 p=[P(v) for v in chain_points(s['q'])]
 for k,(a,b) in enumerate(zip(p[:-1],p[1:])):
  width=max(11,29-2.3*k)
  cv.line(*a,*b,color='#92A5B2',lw=width+2.2)
  cv.line(*a,*b,color='#E0E8EE',lw=width)
  cv.line(a[0],a[1]-2,b[0],b[1]-2,color='#FFFFFF',lw=max(2,width*.22))
 seen=[]
 for k,v in enumerate(s['origins']):
  x,y=P(v)
  if any(np.linalg.norm(np.array([x,y])-a)<3 for a in seen):continue
  seen.append(np.array([x,y]));cv.circle(x,y,9.5,'#F9FBFC','#78909F',1.2);cv.circle(x,y,4.7,'#CBD8E1')
  if numbers and k in [0,1,3]:cv.text(x,y+4,str(k+1),size=11.8,color=INK,anchor='middle',weight='bold')
 hand=P(s['p']);goal=P(s['p_goal'])
 side=projected(np.array(s['r_tool'])[:,1]);ahead=projected(np.array(s['r_tool'])[:,2])
 for sign in [-1,1]:
  origin=hand+side*8*sign;tip=origin+ahead*18
  cv.line(*origin,*tip,color=INK,lw=3)
 cv.line(*(hand-side*8),*(hand+side*8),color=INK,lw=3)
 # A geometric path is generated from the illustrative configurations, not a trial.
 path=[]
 for v in np.linspace(0,1,35):path.append(P(position(chain,s['q_past']+v*(s['q']-s['q_past']))))
 cv.path('M'+'L'.join(f(a)+' '+f(b) for a,b in path),STATE,2,dash='2 4')
 cv.line(*hand,*goal,color=GOAL,lw=1.6,dash='3 4',head='goal')
 triad(cv,*hand,s['r_tool'],36,STATE)
 triad(cv,*goal,s['g_tool'],36,GOAL)
 cv.circle(*goal,12,'none',GOAL,1.2)
 # Concurrent motion is signalled at several joints, never by serial arm states.
 for k in [0,3,5]:
  x,y=P(s['origins'][k]);arc(cv,x,y,17,-140,-25,CORR,1.25,head='corr')
 return P

def matrix(cv,x,y,R,color=STATE,cell=12,label=None):
 for i in range(3):
  for j in range(3):
   amp=abs(float(R[i][j]));cv.rect(x+j*cell,y+i*cell,cell-1.5,cell-1.5,mix(color,.86-.68*amp),rx=1)
 if label:cv.text(x+1.5*cell,y-8,label,size=14,color=color,anchor='middle',italic=True)

def plane(cv,cx,cy,size,uv,color,label=None,back=False):
 # An individual planar field, explicitly separate from the other coordinate planes.
 u=np.array([size*.5,size*.13]);v=np.array([-size*.28,size*.27]);origin=np.array([cx,cy])
 def P(a,b):return origin+a*u+b*v
 poly=[P(-1,-1),P(1,-1),P(1,1),P(-1,1)]
 cv.poly(poly,mix(color,.955),mix(color,.58),.8)
 for i in np.linspace(-.85,.85,9):
  for j in np.linspace(-.85,.85,7):
   x,y=P(i,j);cv.circle(x,y,.65,mix(color,.66))
 pu,pv=np.clip(uv,-1,1)
 for i in np.linspace(-.22,.22,7):
  for j in np.linspace(-.22,.22,7):
   a=math.exp(-((i/.12)**2+(j/.12)**2)/2)
   if a<.08:continue
   x,y=P(pu+i,pv+j);cv.circle(x,y,.8+1.6*a,mix(color,.65*(1-a)))
 if label:cv.text(cx,cy+size*.47+10,label,size=13,color=INK,anchor='middle',italic=True)

def planes(cv,centres,s,size=63):
 for (cx,cy),name,axes in zip(centres,['xy','xz','yz'],[(0,1),(0,2),(1,2)]):
  plane(cv,cx+5,cy-35,size,[s['s_goal'][i]/1.2 for i in axes],GOAL,back=True)
  plane(cv,cx-5,cy+4,size,[s['s'][i]/1.2 for i in axes],STATE,label=name)

def population(cv,x,y,color=CORR,strength=1,rad=2.4):
 for j in range(3):
  for k in range(4):
   amp=max(.12,min(1,strength))*(.75+.25*math.cos(k+j))
   cv.circle(x+(k-1.5)*6.5,y+(j-1)*6.5,rad,fill=mix(color,.72*(1-amp)))

def gear(cv,cx,cy,r=17):
 pts=[]
 for i in range(40):
  a=2*math.pi*i/40;rr=r if i%4 in [0,3] else r*.8
  pts.append((cx+rr*math.cos(a),cy+rr*math.sin(a)))
 cv.poly(pts,'#F0F4F6',GREY,1.1);cv.circle(cx,cy,r*.48,'#FFFFFF',GREY,1);cv.circle(cx,cy,r*.17,GREY)

def motor_horizontal(cv,s,x,y,w=325):
 cv.group('translate(%s %s) scale(%s)'%(f(x),f(y),f(w/325)))
 k=s['motor_joint'];idx=s['positions'][k];lead=s['leads'][k];share=s['shares'][k]
 ring(cv,48,59,34,idx,STATE,label='q₂')
 ring(cv,275,59,34,idx+lead,CORR,recurrent=False,label='q₂*',reference=idx)
 cv.line(84,50,139,40,STATE,1.2,head='state');cv.line(84,70,139,85,STATE,1.2,head='state')
 population(cv,161,39,strength=.7+.3*share);population(cv,161,84,strength=.7-.3*share)
 cv.text(161,18,'+',size=16,color=CORR,anchor='middle');cv.text(161,114,'−',size=18,color=CORR,anchor='middle')
 cv.line(180,40,237,50,CORR,1.2,head='corr');cv.line(180,84,237,70,CORR,1.2,head='corr')
 # A shared correction modulates both opponent populations. A white bridge
 # marks the crossing with the upper population's independent output line.
 cv.path('M161 -31L161 -24L215 -24L215 61L196 61',CORR,1.3)
 cv.line(215,41,215,51,'#FFFFFF',4)
 cv.line(215,41,215,51,CORR,1.3)
 cv.line(196,61,180,51,CORR,1.1,head='corr')
 cv.line(196,61,180,73,CORR,1.1,head='corr')
 cv.circle(196,61,2,CORR)
 cv.text(178,-6,'s₂',size=14,color=CORR,italic=True)
 cv.text(48,114,'State',size=13,anchor='middle');cv.text(275,114,'Target',size=13,anchor='middle')
 cv.end()

def residual_horizontal(cv,s,x0,y0,length=520):
 xs=np.linspace(x0,x0+length,7)
 cv.line(x0-24,y0,x0+length+25,y0,mix(CORR,.82),13)
 cv.line(x0-24,y0,x0+length+25,y0,CORR,1.05,head='corr')
 for k,x in enumerate(xs):
  cv.circle(x,y0,11,'#FFFFFF',mix(CORR,.35),1.2);cv.text(x,y0+4,str(k+1),size=12.5,color=CORR,anchor='middle',weight='bold')
  # Signed shares branch from the fixed-width rail; height encodes |share|.
  cv.line(x,y0+13,x,y0+34,CORR,1,head='corr');amp=float(s['shares'][k]);cv.line(x-13,y0+47,x+13,y0+47,PALE,3)
  cv.line(x,y0+47,x+amp*13,y0+47,CORR,3)
  # Small residual-vector glyphs use computed directions, not an imposed funnel.
  e=s['rests'][k,:3];v=projected(e);norm=np.linalg.norm(v)
  if norm>0:
   end=v/max(.12,norm)*12;cv.line(x,y0-24,x+end[0],y0-24+end[1],CORR,1.1,head='corr')
 return xs

def state_leader(cv,origin,centre,r,index,label):
 ox,oy=origin;cx,cy=centre;d=np.array([ox-cx,oy-cy]);d/=max(np.linalg.norm(d),1e-8)
 cv.line(cx+d[0]*(r+10),cy+d[1]*(r+10),ox,oy,mix(STATE,.58),.85)
 ring(cv,cx,cy,r,index,STATE,label=label)
 cv.circle(ox,oy,2.5,STATE)

def export_metadata(s):
 d={'purpose':'schematic_architecture_geometry_not_neural_recording','source':'analysis/architecture/snapshot.py Q_START/Q_GOAL',
 'q_current_rad':s['q'].tolist(),'q_goal_rad':s['q_goal'].tolist(),'q_earlier_illustrative_rad':s['q_past'].tolist(),
 'hand_current_m':s['p'],'hand_goal_m':s['p_goal'],'orientation_error_deg':s['turn_deg'],
 'joint_shares_rad':s['shares_rad'].tolist(),'joint_shares_fraction_of_cap':s['shares'],'target_leads_ring_indices':s['leads'],
 'residuals_before_each_joint_and_after_last':s['rests'].tolist(),'motor_motif_joint_one_based':s['motor_joint']+1,
 'motor_motif_selection':'largest absolute computed target-ring lead at the declared pose pair',
 'path':'kinematic interpolation from illustrative earlier posture to current posture, not a recorded reach',
 'state_interval':'central70% of ring, endpoints indices30 and170; physical wall margin15indices',
 'glyphs':'computed/schematic; fresh recorded spikes remain in Figure2','residual_rail':'fixed width; no monotonic descent claim',
 'options':'A andB share geometry, computation and colour roles'}
 (DATA/'figure1_snapshot.json').write_text(json.dumps(d,indent=2)+'\n')
