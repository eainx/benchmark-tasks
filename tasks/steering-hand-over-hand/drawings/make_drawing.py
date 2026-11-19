"""Writes drawings/column.svg: dimensioned build drawing of the steering test column (mm)."""
import math, sys
out = sys.argv[1]
E = []
def ln(x1,y1,x2,y2,w=1.2,c="#111",dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    E.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{w}"{d}/>')
def poly(pts,fill="none",w=1.4,c="#111"):
    E.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)}" fill="{fill}" stroke="{c}" stroke-width="{w}"/>')
def circ(x,y,r,fill="none",w=1.3,c="#111",dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    E.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{c}" stroke-width="{w}"{d}/>')
def tx(x,y,s,size=11,anchor="start",c="#111",bold=False):
    b=' font-weight="bold"' if bold else ''
    E.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica,Arial,sans-serif" font-size="{size}" text-anchor="{anchor}" fill="{c}"{b}>{s}</text>')
def dim(x1,y1,x2,y2,label,off=14,size=10):
    # dimension line offset perpendicular
    dx,dy=x2-x1,y2-y1; L=math.hypot(dx,dy); nx,ny=-dy/L*off,dx/L*off
    a=(x1+nx,y1+ny); b=(x2+nx,y2+ny)
    ln(x1,y1,a[0],a[1],0.6,"#555"); ln(x2,y2,b[0],b[1],0.6,"#555")
    ln(a[0],a[1],b[0],b[1],0.8,"#c22")
    for p,q in ((a,b),(b,a)):
        ux,uy=(q[0]-p[0])/L,(q[1]-p[1])/L
        poly([p,(p[0]+7*ux-3*uy,p[1]+7*uy+3*ux),(p[0]+7*ux+3*uy,p[1]+7*uy-3*ux)],fill="#c22",w=0.5,c="#c22")
    tx((a[0]+b[0])/2+nx*0.6,(a[1]+b[1])/2+ny*0.6+4,label,size,"middle","#c22")

W,H=1450,960
# ---------------- View A: side elevation, scale 1:2 (1 mm = 0.5 px) ... use s=1.0 px/mm
s=0.9; ox,oy=80,560   # origin = bench top at the stand's rear edge
def P(x,z): return (ox+x*s, oy-z*s)
tx(40,40,"STEERING TEST COLUMN — build drawing (all dimensions mm; steel unless stated)",16,bold=True)
tx(40,62,"Task: steering-hand-over-hand. Wheel centre 200 above bench, column raked 25° (wheel plane 65° from horizontal). Full lock ±450° (1.25 turns).",11)
tx(60,100,"VIEW A — side elevation (driver/robot on the right)",13,bold=True)
# bench
ln(*P(-40,0),*P(560,0),2.0); E.append(f'<text x="{P(470,-14)[0]:.0f}" y="{P(470,-14)[1]:.0f}" font-size="10" font-family="Helvetica">bench top</text>')
# base plate 360 x 10
poly([P(0,0),P(360,0),P(360,10),P(0,10)],"#ddd")
# column axis: passes through wheel centre at (x_c, 200), direction toward driver = (cos25, sin25)
a=math.radians(25); ux,uz=math.cos(a),math.sin(a)
xc,zc=330,200
def A(t,o=0):  # point along axis t (mm from wheel centre, + toward driver), offset o perpendicular (up)
    return P(xc+t*ux-o*uz, zc+t*uz+o*ux)
# column tube OD 60 from t=-60 (hub flange) back to t=-360
t0,t1=-60,-360
poly([A(t0,30),A(t1,30),A(t1,-30),A(t0,-30)],"#eee")
ln(*A(40,0),*A(-420,0),0.7,"#06c","8 4 2 4")
# shaft 20 dia inside (dashed)
for o in (10,-10): ln(*A(t0+20,o),*A(t1-30,o),0.8,"#333","4 3")
# bearings at both tube ends
for tb in (t0-14,t1+14):
    poly([A(tb-7,23.5),A(tb+7,23.5),A(tb+7,-23.5),A(tb-7,-23.5)],"#bbb",0.8)
# hub adapter + wheel (edge on)
poly([A(t0+2,35),A(t0+20,35),A(t0+20,-35),A(t0+2,-35)],"#ccc")
# wheel plane is perpendicular to axis at t=+35 (dish)
ln(*A(35,190),*A(35,-190),3.0,"#222")
ln(*A(t0+20,0),*A(35,120),1.0); ln(*A(t0+20,0),*A(35,-120),1.0)
# protractor ring on shroud face at t0
ln(*A(t0,45),*A(t0,-45),2.2,"#e0a000")
# post: vertical 50x50 SHS from base to the clamp under the tube at t=-210
tp=-210; px,pz=xc+tp*ux, zc+tp*uz
poly([P(px-25,10),P(px+25,10),P(px+25,pz-30/ux),P(px-25,pz-30/ux)],"#ddd")
# gusset strip
# gearbox plate at lower end (t=-360 .. -380)
gb=-372
poly([A(gb-6,95),A(gb+6,95),A(gb+6,-45),A(gb-6,-45)],"#cfe3ff")
circ(*A(gb-14,75-0),4,"#888")  # output stub shaft marker
tx(*P(-45,250),"gear / stop / spring plate",10); tx(*P(-45,237),"(View B), at the lower end",10)
# dims
dim(*P(0,0),*P(360,0),"360 base plate (×220 wide ×10)",22)
dim(*P(xc,0),*P(xc,zc),"200",-30 if False else 40)
dim(*A(t0,30),*A(t1,30),"300 column tube Ø60×3",18)
tx(*A(60,-60),"25°",12,c="#c22")
ln(*A(0,0),*P(xc+120,zc),0.6,"#c22","3 3")
tx(*P(xc+40,zc+215),"Ø380 wheel (edge-on)",10)
tx(*P(px+32,72),"50×50×3 SHS post,",10); tx(*P(px+32,59),"welded to base,",10); tx(*P(px+32,46),"saddle clamp to tube",10)
tx(*P(0,-58),"Base: 4× M8 bolts or G-clamps to the bench.",10)
tx(*P(0,-73),"Column: 2× 6204-2RS bearings (20×47×14) pressed in end bushes; Ø20 shaft;",10); tx(*P(0,-88),"6-bolt hub adapter on the top end; 20T pinion keyed on the lower end",10)
tx(*A(-30,70),"protractor ring (360°, Ø90) on shroud face",10)

# ---------------- View B: gear / stop / spring plate, end view, scale 2 px/mm
s2=1.7; cx,cy=830,330
tx(700,100,"VIEW B — gear, stop and centring-spring plate (seen along the column, wheel straight ahead)",13,bold=True)
tx(700,118,"Wheel shaft drives the stop gear 4:1 down, so ±450° at the wheel is ±112.5° at the stop gear.",11)
def Q(x,y): return (cx+x*s2, cy-y*s2)
# plate outline 170 x 130
poly([Q(-45,-75),Q(150,-75),Q(150,90),Q(-45,90)],"#eef4ff")
# pinion on wheel shaft at origin: 20T m1.5 -> pitch d30
circ(*Q(0,0),15*s2,"#ccc"); circ(*Q(0,0),10*s2,"#fff"); tx(*Q(-18,30),"20T m1.5 pinion",10,"end")
tx(*Q(-18,23),"on Ø20 wheel shaft",10,"end")
# gear 80T m1.5 pitch d120 at centre distance 75 along +x
gx=75
circ(*Q(gx,0),60*s2,"#e8e8e8"); circ(*Q(gx,0),6*s2,"#fff")
tx(*Q(gx,-84),"80T m1.5 stop gear (Ø120 pitch, 12 face)",10,"middle")
# stop pin at r=50, at top (straight ahead = 90° up)
spx,spy=gx,50
circ(*Q(spx,spy),4*s2,"#c22")
tx(*Q(spx+7,spy+2),"M8 stop pin, r50",10)
# stop screws at ±112.5° from top
for sgn in (1,-1):
    ang=math.radians(90+sgn*112.5)
    x,y=gx+50*math.cos(ang),50*math.sin(ang)
    # screw tip slightly outside pin path, block outside gear
    bx,by=gx+72*math.cos(ang),72*math.sin(ang)
    ln(*Q(x,y),*Q(bx,by),3.0,"#555")
    circ(*Q(x,y),4*s2,"none",1,"#c22","2 2")
    tx(*Q(bx,by-9),("stop for right lock −450°" if sgn>0 else "stop for left lock +450°"),10,"end" if sgn>0 else "start")
tx(700,600,"Stop pin stands 15 proud of the gear face; the stop blocks sit on standoffs in front of the gear face.",11); tx(700,617,"The gears reverse the turn: left at the wheel is clockwise here.",11); tx(700,560,"Stops: 2× M8×40 socket set screws + jam nuts in 20×20×30 steel blocks bolted to the 6 mm plate.",11)
tx(700,577,"1 mm of screw travel = 1.15° at the gear = 4.6° at the wheel. Set each to 450° ±5° at the wheel. Plate 195×165×6.",11)
# torsion spring around stub shaft (shown as circle) with legs to moving pin r40 and fixed pin
circ(*Q(gx,0),18*s2,"none",2.2,"#2a7")
tx(*Q(gx+2,-30),"centring spring (Ø36 coil)",10,"middle",c="#2a7")
for x0 in (-6,6):
    ln(*Q(gx+x0*0.5,18),*Q(gx+x0,42),2.0,"#2a7")
circ(*Q(gx,40),3*s2,"#2a7"); tx(*Q(gx+8,38),"spring pin r40",10,c="#2a7")
tx(*Q(gx+8,31),"(+ fixed pin on the plate",10,c="#2a7"); tx(*Q(gx+8,24),"behind it, same place)",10,c="#2a7")
# shaft labels
tx(*Q(gx+9,-3),"Ø12 stub shaft, 2× 6001-2RS",9)
dim(*Q(0,-68),*Q(gx,-68),"75 centres",0.001)

# ---------------- spring table / notes
y0=670
tx(40,y0,"CENTRING SPRING (double-acting 'scissor' return, as on throttle and joystick returns)",13,bold=True)
notes=[
"Double-leg torsion spring on the stop-gear stub shaft (over a Ø30 bush): music wire Ø4.5, mean coil Ø36, 11 active coils, straight legs 60, legs crossed.",
"Both legs straddle the spring pin on the gear (r40) and a fixed pin on the plate at the same place; turning either way, one leg is carried",
"and the other held, so the spring always winds up and pulls the wheel back to straight ahead.",
"Rate at the stop gear 3.3 N·m/rad (6.4 N·m at 112.5°)  →  at the wheel 3.3/16 = 0.20 N·m/rad: 1.6 N·m at 450° = 8.4 N on the 190 mm rim radius.",
"Bending stress at 6.4 N·m: 32M/πd³ × 1.1 = 0.79 GPa (music wire allowable ≈ 1.4 GPa).",
"Tooth load at the stop: a 30 N push on the rim (5.7 N·m) gives 23 N·m at the gear = 380 N tooth force: well within a steel m1.5 × 12 gear.",
"",
"COMMISSIONING CHECKS (record in the trial sheet)",
"1. Spring scale hooked on the rim at 9 o'clock, pulled tangentially at full lock: 8.4 N ± 1 N (adjust the spring-pin radius 36–44 if needed).",
"2. Turn the wheel 90° left and let go: it returns within 10° of the tape-mark straight-ahead position (otherwise reduce gear/bearing drag).",
"3. Turn to each lock: the yellow tape reads 450° ± 5° on the protractor ring (adjust the stop screw, lock the jam nut).",
"4. Free play at the rim with both pins seated: under 5 mm.",
]
for i,n in enumerate(notes):
    tx(40,y0+22+i*17,n,11,bold=n.startswith("COMMISSIONING"))
svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="{W}" height="{H}" fill="#fff"/>'+"".join(E)+'</svg>'
open(out,"w").write(svg)
