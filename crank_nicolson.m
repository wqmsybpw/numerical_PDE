function [U_all,errL1_all,errL2_all,errInf_all,orderL1_all,orderL2_all,orderInf_all,h_all,k_all]=crank_nicolson(type,flag)
% type = 'cn01' or 'cn02' or 'cn03' or 'cn04' 02,03非对称; flag = 1 or 0 是否画图
%pde:     ut=uxx+f(x,t), x in [a,b], t in [0,T]
%边界条件: u(a)=ua, u(b)=ub, t in [0,T] 初始条件: u(x,0)=v, x in [a,b] % U_ini
% crank-nicolson
% (U^{n+1}_j-U^{n}_j)/k=(1/h^2)*(1/2)*[(U^{n+1}_{j+1}-2U^{n+1}_{j}+U^{n+1}_{j-1})...
% +(U^{n}_{j+1}-2U^{n}_{j}+U^{n}_{j-1})]+(1/2)*(f(xj,tn)+f(xj,tn+1))
% 格式2 
% (1+lambda)*U^{n+1}_j-(lambda/2)(U^{n+1}_{j+1}+U^{n+1}_{j-1})...
% = (1-lambda)*U^{n}_{j}+(lambda/2)(U^{n}_{j+1}+U^{n}_{j-1})+(1/2)*(f(xj,tn)+f(xj,tn+1))
% 格式2
% U^{n+1}_j-(lambda/2)*(U^{n+1}_{j+1}-2U^{n+1}_j+U^{n+1}_{j-1})...
% = U^{n}_j+(lambda/2)*(U^{n}_{j+1}-U^{n}_{j})
a=0;b=1;ini_t=0;end_t=1;
M=[10 20 30 40 80 160 320];
N=[10 20 30 40 80 160 320];
U_all=cell(1,length(M));
errL1_all=zeros(length(M),1);
errL2_all=zeros(length(M),1);
errInf_all=zeros(length(M),1);
h_all=zeros(length(M),1);
k_all=zeros(length(M),1);
orderL1_all=zeros(length(M)-1,1);
orderL2_all=zeros(length(M)-1,1);
orderInf_all=zeros(length(M)-1,1);

for i=1:length(M)
    [U,errL1,errL2,errInf,h,k]=Solver(type,a,b,ini_t,end_t,M(i),N(i),flag);
    U_all{i}=U;
    errL1_all(i)=errL1;
    errL2_all(i)=errL2;
    errInf_all(i)=errInf;
    h_all(i)=h;
    k_all(i)=k;
end

for i=2:length(M)
    orderL1_all(i-1) = log(errL1_all(i)/errL1_all(i-1))/log(h_all(i)/h_all(i-1));
    orderL2_all(i-1) = log(errL2_all(i)/errL2_all(i-1))/log(h_all(i)/h_all(i-1));
    orderInf_all(i-1) = log(errInf_all(i)/errInf_all(i-1))/log(h_all(i)/h_all(i-1));
end

end

function [U_numerical_final,errL1,errL2,errInf,h,k]=Solver(type,a,b,ini_t,end_t,M,N,flag)
%第一步 网格剖分
k=(end_t-ini_t)/N; h=(b-a)/M;
P=[a:h:b-a]';

%第二步 参数函数向量化
u=@(x,t)(x.^2-x).*cos(2*pi*t);
rhs=@(x,t)2*pi*sin(2*pi*t)*(-x.^2+x)-2*cos(2*pi*t);
U_ini=feval(u, P, 0);

%第三步 求解
[U,U_all]=cranknicoson_method_1D(type,U_ini,P,k,N,h,M,u,rhs,a,b);

%第四步：计算误差和收敛阶
U_numerical_final=U;
vector_uexact_final = feval(u,P,end_t);
if strcmp(type,'cn02')||strcmp(type,'cn03')
    ua=feval(u,a,end_t);ub=feval(u,b,end_t);
    U_numerical_final=[ua;U;ub];
    U_all=[ua*ones(1,N);U_all;ub*ones(1,N)];
end
vec_err=U_numerical_final-vector_uexact_final;
errL1=norm(vec_err,1);
errL2=norm(vec_err,2);
errInf=norm(vec_err,inf);

%第五步 画图
if flag==1
    figure; hold on
    for ki=1:100
        plot(P,U_all(:,ki),'--+r');
        Uexact = feval(u,P,ki*k);
        plot(P,Uexact,'-sb');
        legend('numerical solution','exact solution');
    end
    grid on; xlabel x; ylabel y; hold off
    figure
    fsurf(u,[a b ini_t end_t]);
    [t,x]=meshgrid(ini_t:k:end_t-k,a:h:b);
    figure
    surf(x,t,U_all,'EdgeColor','none');
    axis tight
end
end

function [U,U_all]=cranknicoson_method_1D(type,U_ini,P,k,N,h,M,u,rhs,a,b)
Uold=U_ini;
if strcmp(type,'cn01')
    [U,U_all]=solver_CN01(Uold,P,k,h,N,M,u,rhs);
elseif strcmp(type,'cn02')
    [U,U_all]=solver_CN02(Uold,P,k,h,N,M,u,rhs,a,b);
elseif strcmp(type,'cn03')
    [U,U_all]=solver_CN03(Uold,P,k,h,N,M,u,rhs,a,b);
else
    [U,U_all]=solver_CN04(Uold,P,k,h,N,M,rhs);
end
end

function [U,U_all]=solver_CN01(Uold,P,k,h,N,M,u,rhs)
B=sparse(M+1,M+1);A=sparse(M+1,M+1);
F=zeros(M+1,1);
U_all=zeros(M+1,N);
lambda=k/(h^2);
for n=1:N
    tn=n*k;tpre=tn-k;
    B(1,1)=1;F(1,1)=feval(u,P(1),tn);
    B(M+1,M+1)=1;
    F(M+1,1)=feval(u,P(M+1),tn);
    for j=2:M
        B(j,[j-1,j,j+1])=[-lambda/2,1+lambda,-lambda/2];
        A(j,[j-1,j,j+1])=[lambda/2,1-lambda,lambda/2];
        F(j,1)=k*(1/2)*(feval(rhs,P(j),tn)+feval(rhs,P(j),tpre));
    end
    U=B\(A*Uold+F);
    Uold=U;
    U_all(:,n)=U;
end
end


function [U,U_all]=solver_CN02(Uold,P,k,h,N,M,u,rhs,a,b)
    B=sparse(M-1,M-1);A=sparse(M-1,M-1);
    F=zeros(M-1,1);G=zeros(M-1,1);
    U_all=zeros(M-1,N);
    lambda=k/(h^2);
    Uold=Uold(2:length(Uold)-1);
    for n=1:N
        tn=n*k;tpre=tn-k;
        B(1,[1,2])=[1+lambda,-lambda/2];
        B(M-1,[M-2,M-1])=[-lambda/2,1+lambda];
        A(1,[1,2])=[1-lambda,lambda/2];
        A(M-1,[M-2,M-1])=[lambda/2,1-lambda];
        F(1,1)=feval(u,P(1),tn);
        F(M-1,1)=feval(u,P(M-1),tn);
        G(1,1)=lambda*feval(u,a,tn);
        G(M-1,1)=lambda*feval(u,b,tn);
        for j=2:M-2
            B(j,[j-1,j,j+1])=[-lambda/2,1+lambda,-lambda/2];
            A(j,[j-1,j,j+1])=[lambda/2,1-lambda,lambda/2];
            F(j,1)=k*(1/2)*(feval(rhs,P(j+1),tn)+feval(rhs,P(j+1),tpre));
        end
        U=B\(A*Uold+F+G);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        Uold=U;
        U_all(:,n)=U;
    end
end

function [U,U_all]=solver_CN03(Uold,P,k,h,N,M,u,rhs,a,b)
    A=sparse(M-1,M-1);
    F=zeros(M-1,1);G=zeros(M-1,1);
    U_all=zeros(M-1,N);
    lambda=k/(h^2);
    I=eye(M-1);
    Uold=Uold(2:length(Uold)-1);
    for n=1:N
        tn=n*k;tpre=tn-k;
        for j=1:M-1
            F(j,1)=k*(1/2)*(feval(rhs,P(j+1),tn)+feval(rhs,P(j+1),tpre));
            if j==1
                A(j,[j,j+1])=[lambda,-lambda/2];
                G(j)=lambda*feval(u,a,tn);
            elseif j==M-1
                A(j,[j-1,j])=[-lambda/2,lambda];
                G(j)=lambda*feval(u,b,tn);
            else
                A(j,[j-1,j,j+1])=[-lambda/2,lambda,-lambda/2];
            end
        end
        U=(I+A)\((I-A)*Uold+F+G);
        Uold=U;
        U_all(:,n)=U;
    end
end

function [U,U_all]=solver_CN04(Uold,P,k,h,N,M,rhs)
A=sparse(M+1,M+1);
F=zeros(M+1,1);
U_all=zeros(M+1,N);
lambda=k/(h^2);
I=eye(M+1);
for n=1:N
    tn=n*k;tpre=tn-k;
    for j=2:M
        A(j,[j-1,j,j+1])=[lambda/2,-lambda,lambda/2];
        F(j,1)=k*(1/2)*(feval(rhs,P(j),tn)+feval(rhs,P(j),tpre));
    end
    U=(I-A)\((I+A)*Uold+F);
    Uold=U;
    U_all(:,n)=U;
end
end
