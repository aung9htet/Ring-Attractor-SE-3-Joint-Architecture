"""Twelve editable population motifs for the selected Figure 1.

These are schematic reusable circuits, not measured activity or serial stages.
The strip uses absolute SVG coordinates, with no nested scale transformations.
"""
import math
from .figure1_art import STATE, GOAL, CORR, INK, GREY, PALE, SOFT, mix, arc


# (canonical name, overview region letters, two-line display title)
MOTIFS = [
    ('Joint-state ring attractor', 'a', ['Joint-state ring', 'attractor']),
    ('Task-goal populations', 'b', ['Task-goal', 'populations']),
    ('Pose-computation populations', 'ab', ['Pose-computation', 'populations']),
    ('Position-sheet populations', 'b', ['Position-sheet', 'populations']),
    ('Sheet anchoring', 'b', ['Sheet', 'anchoring']),
    ('Sheet activity regulation', 'b', ['Sheet activity', 'regulation']),
    ('Pose-error populations', 'c', ['Pose-error', 'populations']),
    ('Joint-geometry signals', 'd', ['Joint-geometry', 'signals']),
    ('Positive/negative correction populations', 'd', ['Positive/negative', 'correction populations']),
    ('Residual-error populations', 'd', ['Residual-error', 'populations']),
    ('Positive/negative gain populations', 'e', ['Positive/negative', 'gain populations']),
    ('Output and readout populations', 'be', ['Output and readout', 'populations']),
]


def txt(c, x, y, value, color=INK, anchor='middle', size=11.1, **kwargs):
    c.text(x, y, value, size=size, color=color, anchor=anchor, **kwargs)


def cells(c, x, y, color=CORR, strength=1.0):
    """A discrete population glyph, not a plot of spike data."""
    for i in range(3):
        for j in range(3):
            alpha = max(.15, min(1., strength)) * (1. - .13 * abs(i-j))
            c.circle(x+(i-1)*4.3, y+(j-1)*4.3, 1.55,
                     mix(color, .67*(1.-alpha)))


def mini_ring(c, x, y, r=17, color=STATE, bounded=False, recurrent=True, phase=100):
    c.circle(x, y, r, 'none', mix(color, .7), .7)
    for i in range(24):
        idx=(i+.5)*200/24
        delta=(idx-phase+100)%200-100
        amp=math.exp(-.5*(delta/11.)**2)
        angle=math.radians(-90+360*idx/200)
        c.circle(x+r*math.cos(angle), y+r*math.sin(angle), 1.1+amp,
                 mix(color, .75*(1.-amp)))
    if bounded:
        arc(c,x,y,r+3.5,-36,216,mix(color,.4),1.)
        for a in [-36,216]:
            a=math.radians(a)
            c.line(x+(r+1)*math.cos(a),y+(r+1)*math.sin(a),
                   x+(r+6)*math.cos(a),y+(r+6)*math.sin(a),GREY,.8)
    if recurrent:
        head='goal' if color==GOAL else 'state' if color==STATE else 'grey'
        arc(c,x,y,r*.62,30,298,color,.8,head=head)


def sheet(c, x, y, color=STATE, w=31, recurrent=False):
    pts=[(x-w*.55,y-4),(x+w*.2,y-12),(x+w*.55,y+4),(x-w*.2,y+12)]
    c.poly(pts,mix(color,.955),mix(color,.48),.8)
    for i in range(4):
        for j in range(3):
            xx=x+(i-1.5)*4+(j-1)*2
            yy=y+(j-1)*4-(i-1.5)*1.2
            amp=math.exp(-((i-2.)**2+(j-1.)**2)/1.5)
            c.circle(xx,yy,.8+amp*.7,mix(color,.68*(1.-amp)))
    if recurrent:
        head='goal' if color==GOAL else 'state'
        arc(c,x+1,y-1,7,15,290,color,.65,head=head)


def arrow(c,x,y,xx,yy,color=CORR,dash=None):
    tag='state' if color==STATE else 'goal' if color==GOAL else 'corr' if color==CORR else 'grey'
    c.line(x,y,xx,yy,color,.9,dash=dash,head=tag)


def tile1(c, s):
    phase=float(s.get('positions',[100])[0])
    mini_ring(c,84,73,20,STATE,bounded=True,phase=phase)
    txt(c,17,77,'qₖ',STATE,size=12.5)
    arrow(c,29,73,56,73,STATE)
    txt(c,79,123,'Sensory angle input',STATE)


def tile2(c, s):
    mini_ring(c,38,69,16,GOAL,recurrent=True)
    cells(c,115,69,GOAL)
    txt(c,38,46,'θ*',GOAL);txt(c,115,46,'p*',GOAL)
    arrow(c,38,49,38,49.8,GOAL)
    arrow(c,115,49,115,58,GOAL)
    txt(c,38,101,'Orientation',GOAL)
    txt(c,115,101,'Position',GOAL)
    txt(c,79,123,'Task goal',GOAL)


def tile3(c, s):
    for x in [26,78,130]: cells(c,x,65,STATE)
    arrow(c,37,65,65,65,STATE);arrow(c,89,65,117,65,STATE)
    txt(c,26,91,'sin/cos',STATE)
    txt(c,78,91,'Product',STATE)
    txt(c,130,91,'Sum',STATE)
    txt(c,79,123,'Reused along the chain',GREY)


def tile4(c, s):
    for x,col,lab in [(38,STATE,'Current'),(118,GOAL,'Goal')]:
        txt(c,x,45,lab,col)
        sheet(c,x,64,col,w=30,recurrent=True)
        arrow(c,x,78,x,88,col)
        cells(c,x,97,col)
    txt(c,79,114,'Coordinate readouts',GREY)
    txt(c,79,127,'xy, xz, yz',GREY)


def tile5(c, s):
    txt(c,55,46,'Coordinate anchor',STATE)
    txt(c,129,46,'Bias',GREY)
    cells(c,37,65,STATE);cells(c,124,65,GREY)
    arrow(c,45,73,66,88,STATE);arrow(c,116,73,93,88,GREY)
    sheet(c,79,101,STATE,w=32)
    txt(c,79,125,'Packet location',STATE)


def tile6(c, s):
    txt(c,34,46,'Activity',STATE)
    txt(c,116,46,'Set point',GREY)
    sheet(c,34,77,STATE,w=29)
    cells(c,116,77,GREY)
    arrow(c,53,71,102,71,GREY)
    txt(c,76,64,'Σr',GREY)
    arrow(c,116,50,116,64,GREY)
    txt(c,131,61,'−',GREY,size=13)
    # The return is common signed feedback, not a position cue or a divider.
    c.path('M126 82L143 82L143 109L34 109L34 93',GREY,.9,head='grey')
    txt(c,116,96,'Level cells',GREY)
    txt(c,79,126,'Homeostatic feedback',GREY)


def tile7(c, s):
    for x in [39,118]:
        cells(c,x,68,CORR)
        arrow(c,x-22,47,x-8,58,STATE)
        arrow(c,x+22,47,x+8,58,GOAL)
    txt(c,39,91,'e',CORR,size=13)
    txt(c,118,91,'d',CORR,size=13)
    txt(c,39,109,'Orientation',CORR)
    txt(c,39,123,'discrepancy',CORR)
    txt(c,118,109,'Position error',CORR)
    txt(c,118,123,'(offset)',CORR)


def tile8(c, s):
    # The axis is a reused geometric signal, not a new dedicated population.
    c.line(35,69,48,60,STATE,1.);c.line(35,69,27,58,STATE,1.)
    c.line(35,69,35,51,STATE,1.);c.circle(35,69,2,'#FFFFFF',STATE,.8)
    cells(c,85,67,STATE);cells(c,132,67,STATE)
    c.path('M43 53C68 43 107 44 124 57',STATE,.9,head='state')
    arrow(c,97,67,120,67,STATE)
    txt(c,35,89,'aₖ',STATE);txt(c,85,89,'ℓₖ',STATE);txt(c,132,89,'mₖ',STATE)
    txt(c,35,106,'Axis',GREY)
    txt(c,35,120,'(reused)',GREY)
    txt(c,85,106,'Lever',STATE)
    txt(c,125,124,'Sensitivity',STATE)


def tile9(c, s):
    txt(c,88,46,'Fixed weights',GREY)
    c.path('M80 51L80 59L43 59L43 64',CORR,.9,head='corr')
    c.path('M80 59L118 59L118 64',CORR,.9,head='corr')
    cells(c,43,75,CORR);cells(c,118,75,CORR)
    txt(c,43,97,'u⁺',CORR,size=13);txt(c,118,97,'u⁻',CORR,size=13)
    txt(c,16,46,'Limit',GREY)
    c.path('M16 51L16 74L30 74',GREY,.85,dash='2 2')
    c.line(30,68,30,80,GREY,1.)
    txt(c,79,112,'u⁺, u⁻ ∈ [0, 1]',CORR)
    txt(c,79,127,'Outward inhibition',GREY)


def tile10(c, s):
    cells(c,25,71,CORR);cells(c,132,71,CORR)
    c.circle(79,71,10,'#FFFFFF',mix(CORR,.25),.9)
    txt(c,79,75,'−',CORR,size=15)
    arrow(c,37,71,66,71,CORR);arrow(c,92,71,119,71,CORR)
    txt(c,79,48,'hₖsₖ',CORR)
    arrow(c,79,52,79,58,CORR)
    txt(c,25,97,'eₖ, dₖ',CORR)
    txt(c,125,97,'eₖ₊₁, dₖ₊₁',CORR)
    txt(c,79,124,'Remaining error',CORR)


def tile11(c, s):
    mini_ring(c,79,49,10,STATE,recurrent=True)
    txt(c,79,53,'q',STATE)
    cells(c,45,77,CORR);cells(c,113,77,CORR)
    arrow(c,69,58,50,67,STATE);arrow(c,89,58,108,67,STATE)
    txt(c,45,99,'G⁺',CORR,size=12)
    txt(c,113,99,'G⁻',CORR,size=12)
    c.circle(79,108,1.8,CORR)
    c.line(79,113,79,108,CORR,.9)
    arrow(c,77,106,54,86,CORR);arrow(c,81,106,104,86,CORR)
    txt(c,64,103,'+',CORR,size=12);txt(c,95,103,'−',CORR,size=13)
    txt(c,79,128,'u⁺ − u⁻',CORR)


def tile12(c, s):
    # Three separate branches: the centroid is an external spike-count readout;
    # the non-spiking value integrator is not between joint target and servo.
    mini_ring(c,24,51,9,CORR,recurrent=False)
    arrow(c,37,51,57,51,CORR)
    c.rect(60,42,63,18,'#FFFFFF',mix(CORR,.55),.7,rx=2)
    txt(c,91,55,'Centroid',CORR)
    arrow(c,125,51,137,51,CORR)
    txt(c,146,55,'q*',CORR)
    txt(c,24,70,'Target',CORR)
    txt(c,92,70,'3 windows',GREY)
    cells(c,25,88,GREY)
    arrow(c,36,88,68,88,GREY)
    mini_ring(c,82,88,9,GREY,recurrent=False)
    txt(c,103,92,'Display',GREY,anchor='start')
    txt(c,25,106,'Norm',GREY)
    c.rect(17,115,16,12,'#FFFFFF',GREY,.8,rx=2)
    txt(c,25,125,'∫',GREY)
    arrow(c,4,121,14,121,GREY)
    txt(c,42,125,'Non-spiking readout',GREY,anchor='start')


DRAWERS=[tile1,tile2,tile3,tile4,tile5,tile6,
         tile7,tile8,tile9,tile10,tile11,tile12]


def draw_strip(c, s):
    c.line(9,612,991,612,PALE,.9)
    c.heading(13,631,'f','Population motifs')
    txt(c,986,631,'Representative circuits, reused across the controller',GREY,anchor='end')
    for index, (name, regions, titles) in enumerate(MOTIFS):
        col=index%6; row=index//6
        x=9+165*col; y=643+141*row
        c.group('translate(%s %s)'%(x,y))
        c.rect(0,0,157,132,'#FFFFFF',PALE,.7,rx=3)
        c.rect(5,5,18,14,SOFT,GREY,.55,rx=2)
        txt(c,14,16,str(index+1),INK,size=11.1,weight='bold')
        for k,line in enumerate(titles):
            # The subtitle has the full tile width; the badge occupies only
            # the first line, leaving the long correction-population name legible.
            txt(c,28 if k==0 else 8,14+14*k,line,INK,anchor='start',size=11.1,weight='bold')
        DRAWERS[index](c,s)
        c.end()
