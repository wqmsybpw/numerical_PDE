function [U_all,errL1_all,errL2_all,errInf_all,orderL1_all,orderL2_all,orderInf_all,h_all,k_all]=forward_euler
% initional_boundary_value_problem_1D
% ut=uxx+f(x,t), x in [a,b], t in [0,T]
%边界条件: u(a)=ua, u(b)=ub, t in [0,T]
%初始条件: u(x,0)=v, x in [a,b]; U_ini
%离散格式: 欧拉向前有限差分
%U^{n+1}_j=U^{n}_j+(k/h^2)*(U^n_{j+1}-2U^n_{j}+U^n_{j-1})+k*f(xj+tn); k<=0.5*h^2

flag=0; a=0; b=1; ini_t=0; end_t=1;
M=[10 20 30]; N=[1000 4000 9000];
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
    [U,errL1,errL2,errInf,h,k]=Solver(a,b,ini_t,end_t,M(i),N(i),flag);
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

function [U_numerical_final,errL1,errL2,errInf,h,k]=Solver(a,b,ini_t,end_t,M,N,flag)
%第一步 网格剖分
k=(end_t-ini_t)/N; h=(b-a)/M; lambda = k/h^2;
P=[a:h:b-a]';

%第二步 参数函数向量化
uexact_td=@(x,t)(x.^2-x).*cos(2*pi*t);
rhs_td=@(x,t)2*pi*sin(2*pi*t)*(-x.^2+x)-2*cos(2*pi*t);
U_ini=feval(uexact_td, P, 0);

%第三步 求解
U_old=U_ini; U=zeros(M+1,1); U_all=zeros(M+1,N);
for n=1:N
    tn=k*(n-1);
    U(1)=feval(uexact_td,P(1),tn);
    U(M+1)=feval(uexact_td,P(M+1),tn);
    for j=2:M
        U(j)=(1-2*lambda)*U_old(j)+lambda*(U_old(j+1)+U_old(j-1))+k*feval(rhs_td,P(j),tn);
    end
    U_old=U;
    U_all(:,n)=U;
end

%第四步：计算误差和收敛阶
U_numerical_final=U;
vector_uexact_final = feval(uexact_td,P,end_t);
vec_err=U_numerical_final-vector_uexact_final;
errL1=norm(vec_err,1);
errL2=norm(vec_err,2);
errInf=norm(vec_err,inf);

%第五步 画图
if flag==1
    figure; hold on
    for ki=1:100
        plot(P,U_all(:,ki),'--+r');
        Uexact = feval(uexact_td,P,ki*k);
        plot(P,Uexact,'-sb');
        legend('numerical solution','exact solution');
    end
    grid on; xlabel x; ylabel y; hold off
    figure
    fsurf(uexact_td,[a b ini_t end_t]);
    [t,x]=meshgrid(ini_t:k:end_t-k,a:h:b);
    figure
    surf(x,t,U_all,'EdgeColor','none');
    axis tight
end
end
