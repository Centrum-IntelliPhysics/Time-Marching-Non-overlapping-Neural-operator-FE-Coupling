//+
dx=0.05;
 
r1=1.0;
r2=2.0;
h=4.0;

Point(1)={r1, 0.0, 0.0, dx};
Point(2)={r2, 0.0, 0.0, dx};
Point(3)={0.0,  r2, 0.0,dx};
Point(4)={0.0,  r1, 0.0,dx};

Point(8)={r1, 0.0, h, dx};
Point(7)={r2, 0.0, h, dx};
Point(6)={0.0,  r2, h,dx};
Point(5)={0.0,  r1, h,dx};

Point(9)={0.0, 0.0, 0.0, dx};
Point(10)={0.0, 0.0, h, dx};

//+
Circle(1) = {1, 9, 4};
//+
Line(2) = {4, 3};
//+
Circle(3) = {3, 9, 2};
//+
Line(4) = {2, 1};
//+
Circle(5) = {8, 10, 5};
//+
Line(6) = {5, 6};
//+
Circle(7) = {6, 10, 7};
//+
Line(8) = {7, 8};
//+
Line(9) = {4, 5};
//+
Line(10) = {6, 3};
//+
Line(11) = {7, 2};
//+
Line(12) = {1, 8};
//+ 底面 z=0 / 顶面 z=h(平面,带弧形边)
Curve Loop(1) = {1, 2, 3, 4};      Plane Surface(1) = {1};
Curve Loop(2) = {5, 6, 7, 8};      Plane Surface(2) = {2};
//+ 内、外圆柱面(曲面,用 Surface 不是 Plane Surface)
Curve Loop(3) = {1, 9, -5, -12};   Surface(3) = {3};
Curve Loop(4) = {3, -11, -7, 10};  Surface(4) = {4};
//+ 两个径向对称面(x=0 和 y=0 平面)
Curve Loop(5) = {2, -10, -6, -9};  Plane Surface(5) = {5};
Curve Loop(6) = {4, 12, -8, 11};   Plane Surface(6) = {6};
//+ 封壳成体
Surface Loop(1) = {1, 2, 3, 4, 5, 6};

Ntheta = 50; // 弧向节点数
Nz     = 80;  // 高度方向节点数
Transfinite Curve {1, 5} = Ntheta;   // 两条内弧
Transfinite Curve {9, 12} = Nz;      // inner 面的两条竖边
Transfinite Surface {3};

Transfinite Curve {3, 7} = Ntheta;   // 两条内弧
Transfinite Curve {11, 10} = Nz;      // inner 面的两条竖边
Transfinite Surface {4};

Volume(1) = {1};

Physical Volume("solid", 100) = {1};
Physical Surface("bottom", 1) = {1};   Physical Surface("top", 2)   = {2};
Physical Surface("inner", 3)  = {3};   Physical Surface("outer", 4) = {4};
Physical Surface("sym_x", 5)  = {5};   Physical Surface("sym_y", 6) = {6};




