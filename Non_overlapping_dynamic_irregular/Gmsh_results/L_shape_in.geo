//+
SetFactory("OpenCASCADE");
dx=0.01;
 
w=2.0;
h=2.0;
Point(1)={-w / 2,-h/4, 0.0, dx};
Point(2)={-w / 2 + w, -h / 4, 0.0, dx};
Point(3)={-w / 2 + w, -h / 4 + h, 0.0, dx};
Point(4)={-w / 2, -h / 4 + h, 0.0, dx};

//+
Line(1) = {4, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Point(5) = {-w / 4, -h / 4, 0, dx};
Point(6) = {-w / 4, h / 4, 0, dx};
Point(7) = {0.0, h / 4, 0, dx};
Point(8) = {0.0, 0.0, 0, dx};
Point(9) = {w / 4, 0, 0, dx};
Point(10) = {w / 4, -h / 4, 0, dx};

r = 0.1;
Point(11) = {- w / 4 + r, - h / 4 + r + h / 4, 0, dx};
Point(12) = {- w / 4 + r, - h / 4 + h / 4, 0, dx};
Point(13) = {- w / 4, - h / 4 + r + h / 4, 0, dx};

Point(14) = {- w / 4 + r,  h / 4 - r + h / 4, 0, dx};
Point(15) = {- w / 4 + r,  h / 4 + h / 4, 0, dx};
Point(16) = {- w / 4,  h / 4 - r + h / 4, 0, dx};

Point(17) = {0 - r,  h / 4 - r + h / 4, 0, dx};
Point(18) = {0 - r,  h / 4 + h / 4, 0, dx};
Point(19) = {0,  h / 4 - r + h / 4, 0, dx};

Point(20) = {0 + r, 0 + r + h / 4, 0, dx};
Point(21) = {0 + r, 0 + h / 4, 0, dx};
Point(22) = {0, 0 + r + h / 4, 0, dx};

Point(23) = { w / 4 - r, 0 - r + h / 4, 0, dx};
Point(24) = { w / 4 - r, 0 + h / 4, 0, dx};
Point(25) = { w / 4, 0 - r + h / 4, 0, dx};

Point(26) = { w / 4 - r, - h / 4 + r + h / 4, 0, dx};
Point(27) = { w / 4 - r, - h / 4 + h / 4, 0, dx};
Point(28) = { w / 4, - h / 4 + r + h / 4, 0, dx};

//+
Line(5) = {13, 16};
//+
Line(6) = {15, 18};
//+
Line(7) = {19, 22};
//+
Line(8) = {21, 24};
//+
Line(9) = {25, 28};
//+
Line(10) = {27, 12};
//+
Circle(11) = {16, 14, 15};
//+
Circle(12) = {18, 17, 19};
//+
Circle(13) = {22, 20, 21};
//+
Circle(14) = {24, 23, 25};
//+
Circle(15) = {28, 26, 27};
//+
Circle(16) = {12, 11, 13};
//+
Curve Loop(2) = {5, 11, 6, 12, 7, 13, 8, 14, 9, 15, 10, 16};
//+
Plane Surface(2) = {2};
//+
BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; Delete; }

all_surfaces[] = Surface "*";
Printf("Total number of surfaces: %g", #all_surfaces[]);
Printf("=== Surface tag: %g ===", all_surfaces[0]);

//+
Transfinite Curve {19, 17, 18, 20} = 60 Using Progression 1;
//+
Transfinite Curve {6, 7, 8, 9} = 20 Using Progression 1;
//+
Transfinite Curve {11, 12, 13, 14, 15, 16} = 8 Using Progression 1;
//+
Transfinite Curve {5, 10} = 40 Using Progression 1;

// Physical groups
Physical Surface("domain0", 101) = {1};
Physical Curve("BCin", 103) = {6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 5, 10};
Physical Curve("BCout", 102) = {19, 17, 18, 20};


//+
Physical Curve("lc1", 104) = {11};
//+
Physical Curve("lc2", 105) = {16};
//+
Physical Curve("l1", 106) = {5};
//+
Physical Curve("t1", 107) = {6};
//+
Physical Curve("rc1", 108) = {12};
//+
Physical Curve("rc2", 109) = {13};
//+
Physical Curve("rc3", 110) = {14};
//+
Physical Curve("rc4", 111) = {15};
//+
Physical Curve("r1", 112) = {7};
//+
Physical Curve("r2", 113) = {8};
//+
Physical Curve("r3", 114) = {9};
//+
Physical Curve("b1", 115) = {10};
