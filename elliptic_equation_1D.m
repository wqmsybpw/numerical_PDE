function [U_all,errL1_all,errL2_all,errInf_all,orderL1_all,orderL2_all,orderInf_all]=elliptic_equation_1D
%一维两点边值问题 椭圆型方程 有限差分方法
%-a(x)*u''(x)+b(x)*u'(x)+c(x)*u(x)=f(x), x in (0,1), u(a)=ua,u(b)=ub
clear;clc;
a=0;b=1;M=[10 20 30 40 50 60];
flag=1; %画图

U_all = cell(1,length(M));
h_all = zeros(length(M),1);
errL1_all = zeros(length(M),1);
errL2_all = zeros(length(M),1);
errInf_all = zeros(length(M),1);
orderL1_all = zeros(length(M)-1,1);
orderL2_all = zeros(length(M)-1,1);
orderInf_all = zeros(length(M)-1,1);

for i=1:length(M)
    [U,errL1,errL2,errInf,h] = Solver(a,b,M(i),flag);
    U_all{i} = U;
    errL1_all(i) = errL1;
    errL2_all(i) = errL2;
    errInf_all(i) = errInf;
    h_all(i) = h;
end
%收敛阶
for i=2:length(M)
    orderL1_all(i-1) = log(errL1_all(i)/errL1_all(i-1))/log(h_all(i)/h_all(i-1));
    orderL2_all(i-1) = log(errL2_all(i)/errL2_all(i-1))/log(h_all(i)/h_all(i-1));
    orderInf_all(i-1) = log(errInf_all(i)/errInf_all(i-1))/log(h_all(i)/h_all(i-1));
end

end

function [U,errL1,errL2,errInf,h]=Solver(a,b,M,flag)
%第一步 网格剖分 a=x_0<x_1<...<x_M=b, x_j=j*h, j=0,...,M
h = (b-a)/M;
P=[a:h:b]'; P=P(2:M);
%第二步 参数函数向量化
ax=@(x)x.^2+1;
bx=@(x)-cos(x).*sin(x)+x.^2;
cx=@(x)x.*(1-x);
fx=@(x)(cos(x).*sin(x)-x.^2).*(x.*exp(sin(x))+exp(sin(x)).*(x-1)+x.*exp(sin(x)).*cos(x).*(x-1))+...
    (x.^2+1).*(2.*exp(sin(x))+2.*x.*exp(sin(x)).*cos(x)+2.*exp(sin(x)).*cos(x).*(x-1)...
    +x.*exp(sin(x)).*cos(x).^2.*(x-1)-x.*exp(sin(x)).*sin(x).*(x-1))+x.^2.*exp(sin(x)).*(x-1).^2;
ux=@(x)x.*(1-x).*exp(sin(x)); %精确解
vec_a = feval(ax, P);
vec_b = feval(bx, P);
vec_c = feval(cx, P);
vec_f = feval(fx, P);
vec_ue = feval(ux, P);
%第三步 组装系数矩阵，右端向量
A = sparse(M-1,M-1);
A(1,[1,2]) = [2*vec_a(1)+h^2*vec_c(1), -vec_a(1)+0.5*h*vec_b(1)];
A(M-1,[M-2,M-1]) = [-vec_a(M-1)-0.5*h*vec_b(M-1), 2*vec_a(M-1)+h^2*vec_c(M-1)]; 
for i=2:M-2
   A(i,[i-1,i,i+1]) = [-vec_a(i)-0.5*h*vec_b(i),2*vec_a(i)+h^2*vec_c(i),-vec_a(i)+0.5*h*vec_b(i)];
end
g = zeros(M-1,1);
g(1) = h^2*vec_f(1)+(vec_a(1)+0.5*h*vec_b(1))*feval(ux,a);
g(M-1) = h^2*vec_f(M-1)+(vec_a(M-1)+0.5*h*vec_b(M-1))*feval(ux,b);
for i=2:M-2
    g(i) = h^2*vec_f(i);
end
%第四步 求解AU=g A是M-1阶方阵 U是M-1维解向量 g是M-1维右端向量
U = A\g;
%第五步：计算误差
error_vec = U-vec_ue;
errL1 = norm(error_vec,1);
errL2 = norm(error_vec,2);
errInf = norm(error_vec,inf);
%第六步 画图比较
if flag==1
    lb = feval(ux,a); rb = feval(ux,b); %补上边界
    figure;
    hold on
    plot([a;P;b],[lb;U;rb],'--+g');
    plot([a;P;b],[lb;vec_ue;rb],'-sb');
    legend('numerical solution','exact solution');
    hold off
end
end
