function out=PLS_SplitBregman_3D(PET,MR,psf,niter,mu)
% 3D partial volume correction using parallel level set method
% PET: input 3D PET image
% MR: input 3D MR image
% psf: System 3D point spread function
% niter: maximum number of iteration
% mu,lambda: regularization parameters
lambdainit=68;% initial lambda, also used to gaurantee convergence
lambda=lambdainit;
%% rescale
%% smooth and rescale
scale_factor_PET=max(PET(:));
MR=MR/max(MR(:));
PET=PET/scale_factor_PET;
%% parameter to smooth MR to avoid zero
MR_smooth=6e-4;
%% initialization
init=zeros([size(PET)]);
d=cell(3,1); b=cell(3,1); % PLS
d(:)={init}; b(:)={init};
u_current=PET; u_old=zeros([size(PET)]); u=PET;
gv=Gradient_3D(MR,'forward');
gv_norm=sqrt(gv{1}.^2+gv{2}.^2+gv{3}.^2+MR_smooth.^2);
gv={gv{1}./gv_norm;gv{2}./gv_norm;gv{3}./gv_norm};
P=1;[C,~]=obj(PET,PET(:),gv,psf,d,b,mu,lambda);eta=0.995;
%% main iteration
stop_counter=0; %check convergence
flag=0; %control the choice of adaptive \lambda or fixed \lambda
epsilon=0.1*length(u(:))/(181*210*181); %stopping tolerance, normalized as 0.1 for phantom with size 181x210x181;
for i=1:niter
    %% solve u subproblem
    u_current=u;
    fun=@(x)obj(PET,x,gv,psf,d,b,mu,lambda); 
    [~,dfx]=fun(u_current(:));
    alpha=line_search(fun,u_current(:),u_old(:),C);
    u=u_current-alpha*reshape(dfx,[size(u_current)]);
    %% solve d sub problems
    gu=Gradient_3D(u,'forward');
    B_gu=Operator_PLS(gv,gu);
    d_old=d;
    d=d_PLS_update(B_gu,b,lambda);
    %% update b
    b=b_update(B_gu,d,b);
    %% update lambda
    [lambda,rp,rd]=lambda_update(d,d_old,lambda,gv,B_gu,b,flag,lambdainit);
%     fprintf('lambda=%.3e\n',lambda);
    %% update parameters for one-step gradient descent
    [fx,~]=obj(PET,u(:),gv,psf,d,b,mu,lambda);
    P_new=eta*P+1; C=(eta*P*C+fx)/P_new;P=P_new; % update C, P
    u_old=u_current;
    fprintf('finish iteration %d: ||x(k)-x(k-1)||_2/||x(k-1)||_2 = %.3f%%\n',i,norm(u(:)-u_current(:))/norm(u_current(:))*100);
    %% stopping criteria
    if(norm(u(:)-u_current(:))/norm(u_current(:))<1e-3)
        stop_counter=stop_counter+1;
    else
        stop_counter=0;
    end
    if(stop_counter>=3&&flag<=1)% no more progress for adaptive lambda, change to fixed lambda until convergence
        flag=flag+1;
        stop_counter=0;
    elseif(rp<epsilon&&flag>1) 
        break
    end
end
out=u*scale_factor_PET;
end

function [fx,dfx]=obj(imgy,u,gv,psf,d,b,mu,lambda)
% compute fx, dfx of objective function for u_sub problem
img_u=reshape(u,[size(imgy)]);
psf_neg=flip(rot90(psf,2),3);
gu=Gradient_3D(img_u,'forward');
%% compute objection function value
B_gu=Operator_PLS(gv,gu);
fx1=convn(img_u,psf,'same')-imgy;
fx2=cellfun(@(x,y,z)(x-y-z).^2,d,B_gu,b,'un',false); fx2=cellfun(@(x)sum(x(:)),fx2);
fx=(mu/2)*sum(fx1(:).^2)+(lambda/2)*sum(fx2);
%% compute gradient of objective function
cell_fx2=cellfun(@(x,y,z)(x-y-z),d,B_gu,b,'un',false);
B_fx2=Operator_PLS(gv,cell_fx2);

dfx1=convn(convn(img_u,psf,'same')-imgy,psf_neg,'same');
dfx2_tmp1=Gradient_3D(B_fx2{1},'backward');
dfx2_tmp2=Gradient_3D(B_fx2{2},'backward');
dfx2_tmp3=Gradient_3D(B_fx2{3},'backward');
dfx2={dfx2_tmp1{1},dfx2_tmp2{2},dfx2_tmp3{3}};
dfx=mu*dfx1+lambda*sum(cat(4,dfx2{:}),4);
dfx=dfx(:);
end

function alpha=line_search(fun,x,x_old,C)
% compute step size for one-step steepest descnet method 
% fun: function handle of u subproblem objective function
% x: u in current step (u^k)
% x_old: u in previous step (u^(k-1))
% C: average function value
    [~,dfx]=fun(x); [~,dfx_old]=fun(x_old);
    s=x-x_old; y=dfx-dfx_old;
    alpha=s'*y/(y'*y);
    epsilon=1e-4;
    rho=0.4;
    test_step=x-alpha*dfx;
    for i=1:8
        [f_test,~]=fun(test_step);
        if(f_test<=C-epsilon*alpha*dfx'*dfx)
            return
        end
        alpha=alpha*rho;
    end
    fprintf('iteration in line search: %d\n',i);
end

function out=symmetricPad(in)
% symmetrically padding 3D image
% in: input 3D image
[H,L,W]=size(in);
out=zeros(H+2,L+2,W+2);
out(2:end-1,2:end-1,2:end-1)=in;
out(1,:,:)=out(3,:,:); out(end,:,:)=out(end-2,:,:); 
out(:,1,:)=out(:,3,:); out(:,end,:)=out(:,end-2,:);
out(:,:,1)=out(:,:,3); out(:,:,end)=out(:,:,end-2);
end

function gx=Gradient_3D(x,choice)
% compute 3D Gradient of image
% x:input 3D image
% choice: choose finite difference method ('forward','backward')
switch choice
    case 'forward' 
        mask=[0,-1,1];
    case 'backward'
        mask=[-1,1,0];
end
X=symmetricPad(x);
gx_x=convn(X,mask,'same'); gx_x=gx_x(2:end-1,2:end-1,2:end-1);
gx_y=convn(X,mask','same'); gx_y=gx_y(2:end-1,2:end-1,2:end-1);
gx_z=convn(X,reshape(mask,1,1,3),'same'); gx_z=gx_z(2:end-1,2:end-1,2:end-1);
gx={gx_x;gx_y;gx_z};
end

function B_out=Operator_PLS(gv,in)
% compute PLS operation: (I-gv*gv^T)x
% gv: normalized MR gradient information
% in: input image, 3x1 cell 
B_out=cell(3,1);
B_out{1}=(1-gv{1}.^2).*in{1}-gv{1}.*gv{2}.*in{2}-gv{1}.*gv{3}.*in{3};
B_out{2}=-gv{1}.*gv{2}.*in{1}+(1-gv{2}.^2).*in{2}-gv{2}.*gv{3}.*in{3};
B_out{3}=-gv{1}.*gv{3}.*in{1}-gv{2}.*gv{3}.*in{2}+(1-gv{3}.^2).*in{3};
end

function d_out=d_PLS_update(B_gu,b,lambda)
% soft-thresholding operator for solving d subproblem
s=sqrt((B_gu{1}+b{1}).^2+(B_gu{2}+b{2}).^2+(B_gu{3}+b{3}).^2);
tmp=s-1/lambda; tmp(tmp<0)=0;
d_out=cell(3,1);
d_out{1}=tmp.*(B_gu{1}+b{1})./(s+eps);
d_out{2}=tmp.*(B_gu{2}+b{2})./(s+eps);
d_out{3}=tmp.*(B_gu{3}+b{3})./(s+eps);
end

function b_out=b_update(B_gu,d,b)
% update Bregman parameter b
b_out=cellfun(@(x,y,z)(x+y-z),b,B_gu,d,'un',false);
end

function [lambda_out,r_prim_norm,r_dual_norm]=lambda_update(d,d_old,lambda,gv,B_gu,b,flag,lambdain)
% update lambda adaptively with residual balancing
%% primal residual
r_prim=cellfun(@(x,y)(x-y),B_gu,d,'un',false); % primal residual
r_prim=cellfun(@(x)norm(x(:)),r_prim,'un',false);
r_prim=norm(cat(1,r_prim{:}));

factor1=cellfun(@(x)norm(x(:)),B_gu,'un',false);%||B\nablau||_2
factor1=norm(cat(1,factor1{:}));
factor2=cellfun(@(x)norm(x(:)),d,'un',false); %||d||_2
factor2=norm(cat(1,factor2{:}));
r_prim_norm=r_prim/max(factor1,factor2); %normlaized primal residual
%% dual residual
d_tmp=cellfun(@(x,y)(x-y),d_old,d,'un',false);
B_dd=Operator_PLS(gv,d_tmp);
r_dual1=Gradient_3D(B_dd{1},'backward');
r_dual2=Gradient_3D(B_dd{2},'backward');
r_dual3=Gradient_3D(B_dd{3},'backward');
r_dual={r_dual1{1},r_dual2{2},r_dual3{3}};% dual residual
lambda_cell={lambda,lambda,lambda};
r_dual=cellfun(@(x,y)(x*y),lambda_cell,r_dual,'un',false);
r_dual=cellfun(@(x)norm(x(:)),r_dual,'un',false);
r_dual=norm(cat(1,r_dual{:}));

B_gb=Operator_PLS(gv,b);
factor1=Gradient_3D(B_gb{1},'backward');
factor2=Gradient_3D(B_gb{2},'backward');
factor3=Gradient_3D(B_gb{3},'backward');
factor={factor1{1},factor2{2},factor3{3}};
factor=cellfun(@(x)norm(x(:)),factor,'un',false);
factor=norm(cat(1,factor{:}));
r_dual_norm=r_dual/factor; %normalized dual factor
%% compute update ratio
alpha_tmp=(r_prim_norm/r_dual_norm)^0.5;
alpha_max=100;
if(alpha_tmp>=1&&alpha_tmp<alpha_max)
    alpha=alpha_tmp;
elseif(alpha_tmp>1/alpha_max&&alpha_tmp<1)
    alpha=1/alpha_tmp;
else
    alpha=alpha_max;
end
%%%%%%%%%%
beta=8;
if(flag==0)
    if(r_prim>beta*r_dual)
        lambda=alpha*lambda;
    elseif(r_dual>beta*r_prim)
        lambda=1/alpha*lambda;
    end
else
    lambda=lambdain;
end
lambda_out=lambda;
end