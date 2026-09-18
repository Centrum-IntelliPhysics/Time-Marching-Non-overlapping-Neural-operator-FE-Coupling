// =====================================================================
//  90-deg quarter tube, r in [1, 4], height h.
//  Split at r=2 into TWO solids that SHARE a conformal interface:
//     inner_solid : r = 1 .. 2
//     outer_solid : r = 2 .. 4
//  The interface at r=2 (Surface 104) is used by both volumes, so the
//  mesh matches node-for-node across it (ideal for domain decomposition).
//  Transfinite (structured) on the inner cyl (r=1) and middle cyl (r=2).
//  Built-in kernel (no SetFactory) -- keep it that way.
// =====================================================================
r1=1.0;  r2=2.0;  r3=4.0;  h=4.0;

dx     = 0.50;   // mesh size in the FREE (radial) direction & non-structured faces
dx1   = 0.05;
Ntheta = 50;     // nodes along each 90-deg arc  (circumferential)
Nz     = 80;     // nodes along the height

// ---------------- points ----------------
// r=1(细)
Point(1)={r1,0,0,dx1}; Point(4)={0,r1,0,dx1};
Point(11)={r1,0,h,dx1};Point(14)={0,r1,h,dx1};
// r=2(细)
Point(2)={r2,0,0,dx1}; Point(5)={0,r2,0,dx1};
Point(12)={r2,0,h,dx1};Point(15)={0,r2,h,dx1};
// r=4(粗)
Point(3)={r3,0,0,dx};  Point(6)={0,r3,0,dx};
Point(13)={r3,0,h,dx}; Point(16)={0,r3,h,dx};
Point(9)={0,0,0,dx1};  Point(10)={0,0,h,dx1};   // 弧心,不是材料点,随意

// ---------------- arcs (90 deg, centre = axis point) ----------------
Circle(1)={1,9,4};    Circle(2)={2,9,5};    Circle(3)={3,9,6};      // bottom inner/middle/outer
Circle(4)={11,10,14}; Circle(5)={12,10,15}; Circle(6)={13,10,16};   // top

// ---------------- radial spokes ----------------
Line(11)={1,2};  Line(12)={2,3};    // bottom, x-axis
Line(13)={4,5};  Line(14)={5,6};    // bottom, y-axis
Line(15)={11,12};Line(16)={12,13};  // top, x-axis
Line(17)={14,15};Line(18)={15,16};  // top, y-axis

// ---------------- verticals ----------------
Line(21)={1,11}; Line(22)={2,12}; Line(23)={3,13};   // x-axis (r=1,2,4)
Line(24)={4,14}; Line(25)={5,15}; Line(26)={6,16};   // y-axis (r=1,2,4)

// ================= INNER solid  (r = 1 .. 2) =================
Curve Loop(101)={1,13,-2,-11};   Plane Surface(101)={101}; // bottom sector
Curve Loop(102)={4,17,-5,-15};   Plane Surface(102)={102}; // top sector
Curve Loop(103)={1,24,-4,-21};   Surface(103)={103};       // inner cyl  r=1
Curve Loop(104)={2,25,-5,-22};   Surface(104)={104};       // middle cyl r=2  <-- INTERFACE
Curve Loop(105)={11,22,-15,-21}; Plane Surface(105)={105}; // sym_x face (y=0)
Curve Loop(106)={13,25,-17,-24}; Plane Surface(106)={106}; // sym_y face (x=0)
Surface Loop(201)={101,102,103,104,105,106};
Volume(201)={201};

// ================= OUTER solid  (r = 2 .. 4) =================
// Surface 104 (r=2) is reused here -> shared, conformal interface.
Curve Loop(111)={2,14,-3,-12};   Plane Surface(111)={111}; // bottom sector
Curve Loop(112)={5,18,-6,-16};   Plane Surface(112)={112}; // top sector
Curve Loop(114)={3,26,-6,-23};   Surface(114)={114};       // outer cyl r=4
Curve Loop(115)={12,23,-16,-22}; Plane Surface(115)={115}; // sym_x face (y=0)
Curve Loop(116)={14,26,-18,-25}; Plane Surface(116)={116}; // sym_y face (x=0)
Surface Loop(202)={111,112,104,114,115,116};
Volume(202)={202};

// ================= transfinite: inner (103) + middle/interface (104) =================
Transfinite Curve {1,4}          = Ntheta;   // inner arcs (r=1)
Transfinite Curve {2,5}          = Ntheta;   // middle arcs (r=2)
Transfinite Curve {21,24,22,25}  = Nz;       // verticals bounding surfaces 103 & 104
Transfinite Surface {103,104};

// ================= physical groups =================
Physical Volume ("inner_solid",100) = {201};
Physical Volume ("outer_solid",200) = {202};
Physical Surface("inner",     1) = {103};       // r=1
Physical Surface("interface", 2) = {104};       // r=2  (shared)
Physical Surface("outer",     3) = {114};       // r=4
Physical Surface("bottom",    4) = {101,111};
Physical Surface("top",       5) = {102,112};
Physical Surface("sym_x",     6) = {105,115};
Physical Surface("sym_y",     7) = {106,116};

// =====================================================================
//  OPTIONAL: make the WHOLE thing a pure conformal HEX8 mesh.
//  Uncomment the block below (and you can then ignore dx).
//  Nr1/Nr2 = nodes across the inner / outer ring in the radial direction.
// =====================================================================
// Nr1 = 8;  Nr2 = 12;
// Transfinite Curve {1,2,3,4,5,6}       = Ntheta;
// Transfinite Curve {11,13,15,17}       = Nr1;
// Transfinite Curve {12,14,16,18}       = Nr2;
// Transfinite Curve {21,22,23,24,25,26} = Nz;
// Transfinite Surface {101,102,103,104,105,106,111,112,114,115,116};
// Transfinite Volume  {201,202};
// Recombine Surface   {101,102,103,104,105,106,111,112,114,115,116};
