//+
SetFactory("OpenCASCADE");
dx=0.01;
 
w=2.0;
h=2.0;

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
Transfinite Curve {6, 7, 8, 9} = 20 Using Progression 1;
//+
Transfinite Curve {11, 12, 13, 14, 15, 16} = 8 Using Progression 1;
//+
Transfinite Curve {5, 10} = 40 Using Progression 1;

// Physical groups
Physical Surface("domain0", 1002) = {2};
Physical Curve("BCin", 104) = {6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 5, 10};






