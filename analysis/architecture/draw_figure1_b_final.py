"""Final three-panel architecture: typed projections, separate joint channels.

All activity and the arm are schematic at the declared, pinned TIAGo pose pair.
No neural simulation is run. Full motif wiring is exported separately for SI.
"""
import re,json
from pathlib import Path
from .figure1_art import *
from .figure1_art import DATA,FIGURES,ensure_dirs
from .figure1_population_strip import draw_strip

class FigureArt(Art):
 def __init__(self,height=135,minfont=12.0):
  self.minfont=minfont;self.scales=[1.];self.fonts=[];h=1000*height/178
  self.parts=['<svg xmlns="http://www.w3.org/2000/svg" width="178mm" height="%smm" viewBox="0 0 1000 %s" font-family="DejaVu Sans, Arial, sans-serif">'%(f(height),f(h)),self.definitions()]
  self.rect(0,0,1000,h,'#FFFFFF')
 def group(self,transform=None,opacity=None):
  scale=1.
  for token in re.findall(r'scale\(([^)]+)\)',transform or ''):
   v=[float(x) for x in re.split(r'[,\s]+',token.strip())];assert len(v)==1 or v[0]==v[1];scale*=v[0]
  self.scales.append(self.scales[-1]*scale);super().group(transform,opacity)
 def end(self):self.scales.pop();super().end()
 def heading(self,x,y,letter,title):
  self.text(x,y,letter,size=17,weight='bold',anchor='start');self.text(x+25,y,title,size=15.3,weight='bold',anchor='start')
 def text(self,x,y,s,size=12.4,anchor='middle',**kwargs):
  size=max(size,self.minfont/self.scales[-1]);self.fonts.append(size*self.scales[-1]*178/1000*72/25.4)
  super().text(x,y,s,size=size,anchor=anchor,**kwargs)

def label(c,x,y,lines,color=INK,anchor='middle',size=12.4,weight='normal'):
 for j,t in enumerate(lines):c.text(x,y+j*15,t,size=size,color=color,anchor=anchor,weight=weight)
def pop(c,x,y,col=STATE,strength=.85):
 for i in range(3):
  for j in range(3):c.circle(x+(i-1)*5,y+(j-1)*5,1.9,mix(col,.45*(1-strength*(.8+.2*math.cos(i+j)))))
def arrow(c,a,b,col=STATE,lw=1.3):
 c.line(*a,*b,col,lw,head='state' if col==STATE else 'goal' if col==GOAL else 'corr' if col==CORR else 'pred' if col==PRED else 'grey')
def mini_ring(c,x,y,r,index,col=STATE,bounded=True,recurrent=True,text=None):
 for i in range(40):
  idx=200*(i+.5)/40;dist=(idx-index+100)%200-100;a=math.radians(-90+360*idx/200);v=math.exp(-.5*(dist/8.)**2)
  c.circle(x+r*math.cos(a),y+r*math.sin(a),1+1.5*v,mix(col,.5*(1-v)))
 if bounded:
  arc(c,x,y,r+5,-36,216,mix(col,.5),1.2)
  for deg in [-36,216]:
   a=math.radians(deg);c.line(x+(r+3)*math.cos(a),y+(r+3)*math.sin(a),x+(r+9)*math.cos(a),y+(r+9)*math.sin(a),GREY,.8)
 if recurrent:arc(c,x,y,r*.68,25,300,col,.9,head='state' if col==STATE else 'goal')
 if text:c.text(x,y+4,text,size=13,color=col,anchor='middle',italic=True)

def panel_a(c,s):
 P=robot(c,s,(39,88,259,170),numbers=False)
 # Illustration of embodiment, not an independently measured neural pose.
 c.text(40,268,'TIAGo geometry',color=GREY,size=12.2,anchor='start')
 c.text(287,117,'p*, R*',color=GOAL,size=12.3,anchor='end',italic=True)
 c.text(165,288,'Measured joint angles q',color=STATE)
 c.line(36,299,286,299,STATE,1.1)
 for k,x,lab in [(0,65,'q₁'),(1,165,'qₖ'),(6,265,'qₙ')]:
  arrow(c,(x,299),(x,307),STATE,1)
  mini_ring(c,x,334,23,s['positions'][k],STATE,text=lab)
 c.text(165,374,'Recurrent joint-state rings',color=STATE)
 arrow(c,(165,359),(165,397));pop(c,165,410)
 c.text(198,414,'×n',color=STATE)
 c.text(165,439,'sin/cos populations',color=STATE)
 c.path('M165 447L165 458L111 458L111 473',STATE,1.3,head='state')
 pop(c,111,486);pop(c,220,486)
 arrow(c,(125,486),(204,486))
 c.text(111,510,'Products',color=STATE);c.text(220,510,'Sums',color=STATE)
 # Grouped forward projection; matrix/position/geometry outputs are a tuple.
 c.rect(29,529,279,105,'#FFFFFF',PALE,.8,rx=5)
 arrow(c,(220,512),(220,528))
 matrix(c,58,543,s['r_tool'],STATE,cell=9.5,label='R')
 triad(c,125,559,s['r_tool'],22,STATE)
 pop(c,247,552);c.text(247,576,'aₖ, mₖ',color=STATE)
 for x,name,axes in [(72,'xy',(0,1)),(164,'xz',(0,2)),(255,'yz',(1,2))]:
  plane(c,x,602,45,[s['s'][i]/1.2 for i in axes],STATE,label=name)
 c.text(177,647,'Pose and joint-geometry code',color=STATE)
 c.line(308,578,330,578,STATE,1.3)
 c.path('M330 578L330 210L487 210',STATE,1.5,head='state')
 # One small local detail; the full motif catalogue is in the supplement.
 c.rect(29,662,279,49,'#FFFFFF',PALE,.8,rx=4)
 pop(c,62,695,STATE);c.text(62,676,'Anchor',color=STATE)
 plane(c,143,695,29,[0,0],STATE)
 pop(c,244,695,GREY)
 arrow(c,(77,691),(120,691),STATE,1)
 arrow(c,(167,689),(228,689),GREY,1)
 c.path('M244 704L244 709L143 709L143 705',GREY,.9,head='grey')
 c.text(244,676,'Activity-level cells',color=GREY)

def allocation(c,s,number,cy,selected=False):
 col=CORR
 c.rect(410,cy-40,128,80,'#FFFFFF',mix(CORR,.65),1.1 if selected else .8,rx=8)
 c.text(474,cy-19,'Joint '+str(number),size=13,color=CORR,weight='bold')
 pop(c,445,cy+1,CORR,.65)
 c.circle(445,cy+25,8,'#FFFFFF',mix(CORR,.35),.8)
 c.text(445,cy+29,'−',size=14,color=CORR)
 c.line(445,cy+10,445,cy+17,CORR,1)
 c.line(445,cy+33,445,cy+39,CORR,1)
 c.path('M530 %sL530 %sL455 %s'%(f(cy+1),f(cy+25),f(cy+25)),CORR,.9,head='corr')
 c.text(505,cy+19,'Effect',color=CORR,size=12)
 arrow(c,(458,cy+1),(572,cy+1),CORR,1.1)
 suffix={'1':'₁','k':'ₖ','n':'ₙ'}[str(number)]
 c.text(385,cy-10,'a'+suffix+', m'+suffix,color=STATE)
 arrow(c,(355,cy),(408,cy),STATE,1)
 # Independent sign channels: bracketed pairs, not a shared scalar bus.
 pop(c,600,cy-10,CORR,.35);pop(c,600,cy+10,CORR,.9)
 c.text(600,cy-25,'u⁺',color=CORR);c.text(600,cy+31,'u⁻',color=CORR)
 if not selected:
  arrow(c,(612,cy-10),(626,cy-10),CORR,1)
  arrow(c,(612,cy+10),(626,cy+10),CORR,1)

def panel_b(c,s):
 label(c,502,104,['Task goal'],GOAL,weight='bold')
 # Goal input is encoded, not a physically executed target posture.
 mini_ring(c,465,140,16,90,GOAL,bounded=False,recurrent=True)
 pop(c,557,140,GOAL)
 c.text(451,173,'Orientation',color=GOAL,anchor='end');c.text(570,173,'Position',color=GOAL,anchor='start')
 c.path('M557 151L557 210L518 210',GOAL,1.4,head='goal')
 c.path('M465 158L465 184L557 184',GOAL,1.1)
 c.circle(557,184,2,GOAL)
 c.circle(502,210,13,'#FFFFFF',mix(CORR,.35),1)
 c.text(502,215,'−',size=19,color=CORR)
 c.text(390,199,'p, R',color=STATE,italic=True)
 c.text(573,205,'p*, R*',color=GOAL,anchor='start',italic=True)
 c.text(502,244,'Pose error e, d',color=CORR)
 c.path('M502 224L502 265L445 265L445 286',CORR,1.5,head='corr')
 # A distinct forward-feature branch supplies axes/lever sensitivities.
 c.path('M355 210L355 620',STATE,1.1)
 label(c,371,264,['Joint','geometry'],STATE,anchor='start',size=12)
 c.line(445,282,445,665,mix(CORR,.86),11)
 c.line(445,282,445,665,CORR,.9,head='corr')
 allocation(c,s,'1',330);allocation(c,s,'k',480,True);allocation(c,s,'n',630)
 c.text(445,409,'⋮',size=21,color=CORR);c.text(445,559,'⋮',size=21,color=CORR)
 c.text(472,396,'Residual',color=CORR,anchor='start')
 c.text(472,546,'Residual',color=CORR,anchor='start')
 c.text(474,693,'Base → wrist',color=GREY)
 arrow(c,(445,670),(445,682),CORR,1)
 # A grey brace denotes a collection; it is deliberately not a neural wire.
 c.path('M644 306Q653 306 653 320L653 464Q653 479 663 480Q653 481 653 496L653 640Q653 655 644 655',GREY,.9)
 label(c,609,269,['Per-joint','corrections'],CORR,size=12)
 # Selected k pair proceeds to the target-writing circuit through two lanes.
 c.path('M612 470L657 470L657 442L738 442',CORR,1.35,head='corr')
 c.path('M612 490L667 490L667 462L738 462',CORR,1.35,head='corr')

def panel_c(c,s):
 k=s['motor_joint'];idx=s['positions'][k];lead=s['leads'][k]
 c.text(836,107,'Joint k, enlarged from '+STATE_PANEL[0],color=GREY)
 mini_ring(c,800,204,39,idx,STATE,text='qₖ')
 c.text(800,263,'Same joint-state ring',color=STATE)
 arrow(c,(800,247),(800,366),STATE,1.4)
 label(c,812,300,['Joint-state','gating'],STATE,anchor='start')
 c.rect(740,367,115,155,'#FFFFFF',mix(CORR,.56),.9,rx=8)
 label(c,918,347,['Opponent gain','populations'],CORR)
 c.line(875,372,856,383,GREY,.8)
 pop(c,795,410,CORR,.45);pop(c,795,478,CORR,.95)
 c.text(795,391,'G⁺',size=14,color=CORR)
 c.text(795,508,'G⁻',size=14,color=CORR)
 c.text(719,427,'uₖ⁺, uₖ⁻',color=CORR,anchor='end')
 # Input ports denote grouped opposite-sign modulation, detailed in SI.
 c.text(797,448,'±',size=19,color=CORR)
 c.line(809,410,855,410,CORR,1.1);c.line(809,478,855,478,CORR,1.1)
 arrow(c,(855,410),(890,425),CORR,1.3)
 arrow(c,(855,478),(890,465),CORR,1.3)
 mini_ring(c,920,445,32,idx+lead,CORR,recurrent=False,text='qₖᵗᵃʳ')
 label(c,920,496,['Feed-forward','target ring'],CORR)
 c.text(839,552,'Target circuit repeated for all joints',color=GREY)
 # A vector command belongs to the set of repeated circuits, not only k.
 c.path('M955 445L978 445L978 591L838 591L838 627',CORR,1.4,head='corr')
 c.text(920,577,'Joint targets qᵗᵃʳ',color=CORR)
 c.rect(736,628,226,72,'#F7F9FA',PALE,.9,rx=6)
 gear(c,838,651,18)
 c.text(847,688,'External joint servo / plant',color=GREY)
 c.path('M817 651L714 651L714 729L12 729L12 299L35 299',STATE,1.5,head='state')
 c.text(414,751,'Measured joint angles q(t)',color=STATE)

#: which letter the joint-state panel has (panel d refers to it)
STATE_PANEL=['a']

def badge(c,x,y,word,col,size=11.2):
 """A solid label marking an ablation site (Fig. 5 colours): white text on the colour."""
 w=14+.62*size*len(word)
 c.rect(x,y-11,w,15.5,col,None,0,rx=3.5)
 c.text(x+w/2,y+1,word,size=size,color='#FFFFFF',weight='bold')
 return w

def overlays(c,s):
 """Predictive encoding (blue, Fig. 4) and the ablation sites (Fig. 5), in the circuit's own coordinates."""
 # predictive encoding: the forward joint rings are written ahead by the commanded velocity...
 label(c,214,399,['written ahead:','qₖ+τₖq̇ₖᶜᵐᵈ'],PRED,size=11.2,weight='bold',anchor='start')
 c.path('M218 388Q202 372 189 351',PRED,1.2,head='pred')
 # ...and a separate motor ring keeps the measured angle for the target circuit
 mini_ring(c,716,146,15,s['positions'][s['motor_joint']],PRED,recurrent=True)
 c.text(716,150,'qₖ',size=10.6,color=PRED,italic=True)
 label(c,716,181,['Motor ring','(measured)'],PRED,size=10.4,weight='bold')
 c.path('M716 200L716 330L796 330',PRED,1.2,dash='3 2',head='pred')
 # ablations of Fig. 5, where each acts
 badge(c,883,640,'✕ Lock',ABLATION['lock'])
 badge(c,702,357,'✕ Motor',ABLATION['motor'])
 badge(c,549,700,'✕ Allocation',ABLATION['allocation'])

def legend(c,x0=20,y=16):
 """Colour meanings: roles, predictive encoding, ablations."""
 x=x0
 for col,word in [(STATE,'State'),(GOAL,'Goal'),(CORR,'Correction'),(PRED,'Predictive encoding (Fig. 4)')]:
  c.circle(x,y-4,3.6,col);c.text(x+9,y,word,size=11.6,color=col,anchor='start',weight='bold');x+=22+7.0*len(word)
 x+=8;c.text(x,y,'Ablated (Fig. 5):',size=11.2,color=INK,anchor='start');x+=104
 for key,word in [('lock','✕ Lock'),('motor','✕ Motor'),('allocation','✕ Allocation'),('diffuse','✕ 10% of all')]:
  x+=badge(c,x,y-1,word,ABLATION[key],10.4)+6

def circuit(c,s,letters='abc'):
 """The circuit (three panels) in its own 1000 x 758 coordinates."""
 for x in [340,677]:
  c.line(x,82,x,188,PALE,.9);c.line(x,233 if x==340 else 513,x,707,PALE,.9)
 for x,letter,lines in [(20,letters[0],['Joint-state and','pose representation']),(355,letters[1],['Pose-error and','correction allocation']),(694,letters[2],['Joint-target generation','and feedback'])]:
  c.text(x,55,letter,size=18,weight='bold',anchor='start')
  label(c,x+25,50,lines,anchor='start',size=14.2,weight='bold')
 STATE_PANEL[0]=letters[0]
 panel_a(c,s);panel_b(c,s);panel_c(c,s);overlays(c,s)

def inline_svg(path,x,y,w,h):
 """Another SVG (Fig. 1a, drawn by Matplotlib) placed as a nested, vector SVG."""
 text=Path(path).read_text()
 text=re.sub(r'<\?xml[^>]*\?>','',text);text=re.sub(r'<!DOCTYPE[^>]*>','',text,flags=re.S)
 head=re.search(r'<svg[^>]*>',text).group(0)
 box=re.search(r'viewBox="([^"]+)"',head).group(1)
 new='<svg x="%s" y="%s" width="%s" height="%s" viewBox="%s" preserveAspectRatio="xMidYMin meet">'%(f(x),f(y),f(w),f(h),box)
 return text.replace(head,new,1)

#: Fig. 1 on a 178 mm wide page: a (hand paths, 44 mm) | b-d (circuit, 134 mm); e (motifs) underneath
LEFT=247;SCALE=(1000-LEFT-3)/1000;STRIP_Y=598;HEIGHT_MM=167

def make():
 ensure_dirs()
 s=get_data()
 c=FigureArt(HEIGHT_MM,minfont=9.9)
 legend(c)
 # a: every controller's hand path (python -m analysis figure fig1 draws it)
 c.text(8,62,'a',size=18,weight='bold',anchor='start')
 label(c,30,57,['Target pair 0,','every controller'],anchor='start',size=14.2,weight='bold')
 hand=FIGURES/'fig1a_hand_paths.svg'
 if not hand.exists():raise SystemExit('draw Fig. 1a first: python3 -m analysis figure fig1')
 c.add(inline_svg(hand,0,84,LEFT,STRIP_Y-84-6))
 c.group('translate(%s 26) scale(%s)'%(f(LEFT+3),f(SCALE)));circuit(c,s,'bcd');c.end()
 c.group('translate(0 %s)'%f(STRIP_Y-612));draw_strip(c,s);c.end()
 c.parts=[v.replace('>f</text>','>e</text>').replace('>hₖsₖ</text>','>Joint effect</text>') for v in c.parts]
 c.save(FIGURES/'fig1_model.svg')
 out={'figure':'Fig. 1: a, hand paths of every controller (Gazebo target pair 0); b-d, the circuit; e, population motifs',
 'dimensions_mm':[178,HEIGHT_MM],'minimum_label_pt':min(c.fonts),
 'panels':{'a':'Target pair 0, every controller (fig1a_hand_paths.svg)','b':'Joint-state and pose representation','c':'Pose-error and correction allocation','d':'Joint-target generation and feedback','e':'Population motifs'},
 'overlays':{'predictive_encoding':'blue (Fig. 4): forward rings hold q_k + tau_k qdot_k^cmd; a separate motor ring keeps the measured angle for the target circuit',
  'ablations':'solid badges in the Fig. 5 colours: lock at the actuator, motor at the gain pair and target ring, allocation at the u+/u- correction pairs; diffuse 10% silencing applies to every population (key only)'},
 'geometry':'pinned TIAGo pose pair: analysis/source_data/figure1_snapshot.json',
 'diagram_convention':'arrows carry named neural signals; a grey brace groups independent joint channels; gain input ports represent grouped projections',
 'activity':'all circuit activity glyphs are schematic; measured activity is in Figure 2'}
 (DATA/'figure1_b_manifest.json').write_text(json.dumps(out,indent=2)+'\n')
 print('Fig. 1 SVG written; minimum label %.2f pt'%min(c.fonts))

if __name__=='__main__':make()
