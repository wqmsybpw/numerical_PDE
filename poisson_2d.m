function [errL1_all,errL2_all,errInf_all,orderL1_all,orderL2_all,orderInf_all]=poisson_2d
%二维泊松方程 -u_xx-u_yy-f(x,y) in [x,y]*[x,y] 有限差分方法
clear;clc;
x=0;y=1;
M=[10 20 30 40 50 60]; %网格剖分大小
draw=1;
errL1_all=zeros(length(M),1);
errL2_all=zeros(length(M),1);
errInf_all=zeros(length(M),1);
h_all=zeros(length(M),1);
orderL1_all=zeros(length(M)-1,1);
orderL2_all=zeros(length(M)-1,1);
orderInf_all=zeros(length(M)-1,1);

for i=1:length(M)
    [~,errL1,errL2,errInf,h]=Solver(x,y,M(i),draw);
    errL1_all(i)=errL1;
    errL2_all(i)=errL2;
    errInf_all(i)=errInf;
    h_all(i)=h;
end
%计算收敛阶
for i=2:length(M)
    orderL1_all(i-1) = log(errL1_all(i)/errL1_all(i-1))/log(h_all(i)/h_all(i-1));
    orderL2_all(i-1) = log(errL2_all(i)/errL2_all(i-1))/log(h_all(i)/h_all(i-1));
    orderInf_all(i-1) = log(errInf_all(i)/errInf_all(i-1))/log(h_all(i)/h_all(i-1));
end
end

function [U,errL1,errL2,errInf,h]=Solver(x,y,M,draw)
h=(y-x)/M;
Nv=(M+1)*(M+1); %节点个数
P=zeros(Nv,3); %P(:,3) 表示内部/边界点 0/-1; P(k,1)表示节点k的x坐标 P(k,2)表示y坐标
for i=1:M+1
    xi=x+(i-1)*h;
    for j=1:M+1
        yj=x+(j-1)*h;
        k=i+(j-1)*(M+1); % 按行转换坐标
        P(k,1:2)=[xi,yj];
        if i==1 || i==M+1 || j == 1 || j == M+1 %边界
            P(k,3)=-1; 
        end
    end
end

vec_rhs=zeros(size(P,1),1);
vec_ue=zeros(size(P,1),1);
uexact_2D=@(x,y)sin(x.^2.*(x-1).^2).*tan(y.^2.*(y-1).^2); %测试用
rhs_func_2D=@(x,y)sin(x.^2.*(x - 1).^2).*tan(y.^2.*(y - 1).^2).*(2.*x.*(x - 1).^2 + x.^2.*(2.*x - 2)).^2 ...
    - cos(x.^2.*(x - 1).^2).*tan(y.^2.*(y - 1).^2).*(4.*x.*(2.*x - 2) + 2.*(x - 1)^2 + 2.*x^2)...
    - sin(x^2.*(x - 1)^2).*(tan(y^2.*(y - 1)^2)^2 + 1).*(4.*y.*(2.*y - 2) + 2.*(y - 1)^2 + 2.*y^2) ...
    - 2.*sin(x^2.*(x - 1)^2).*tan(y^2.*(y - 1)^2).*(tan(y^2.*(y - 1)^2)^2 + 1).*(2.*y.*(y - 1)^2 + y^2.*(2.*y - 2))^2;
for i=1:Nv
    vec_rhs(i,1)=feval(rhs_func_2D, P(i,1),P(i,2));
    vec_ue(i,1)=feval(uexact_2D, P(i,1),P(i,2)); 
end

A=sparse(Nv,Nv); %系数矩阵
F=zeros(Nv,1); %右端向量
for k=1:Nv
    if P(k,3)==-1
        A(k,k)=1;
        F(k,1)=0;
    else
        A(k, [k-(M+1), k-1, k, k+1, k+M+1])=[-1 -1 4 -1 -1];
        F(k,1)=h^2*vec_rhs(k);
    end
end

U=A\F; %解出数值解与精确解比较
vec_err=U-vec_ue;
errL1=norm(vec_err,1);
errL2=norm(vec_err,2);
errInf=norm(vec_err,inf);

if draw==1
    figure
    surf(reshape(P(:,1),M+1,M+1),reshape(P(:,2),M+1,M+1),reshape(vec_ue,M+1,M+1));
    xlabel x; ylabel y; zlabel z
    title(strcat('exact hx=',mat2str(h)));
    figure
    surf(reshape(P(:,1),M+1,M+1),reshape(P(:,2),M+1,M+1),reshape(U,M+1,M+1));
    xlabel x; ylabel y; zlabel z
    title(strcat('numerical hx=',mat2str(h)));
end
end