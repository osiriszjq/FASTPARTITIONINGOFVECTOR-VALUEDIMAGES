function [ v,lambda,mu ] = DownS_pre(f, gamma, tao, mu0)
%DOWNSPRE Summary of this function goes here
%   Detailed explanation goes here
%   First step in ADMM4
%   same inputs as ADMM4
%   
%% downsampling
    [M,N,S] = size(f);
    
    m=mod(1:M,2)==1;
    n=mod(1:N,2)==1;
    f0=f(m,n,:);
    %% ADMM4
    v0 = f0;
    mu = mu0/2;
    stop=sum(sum(sum(f0.^2)))*1e-7;
    lambda0 = zeros(size(f0));
    %u = zeros(size(f0));
    error = +Inf;
    gamma=gamma/2;
    iter = 1;
    while error > TOL
        f1 = f + mu*v - lambda;
        u = UnivPottsVectorParallel(permute(f1,[2,1,3])/(1+mu),2*gamma/(1+mu),ones(N,M));
        u = permute(u,[2,1,3]);
        
        f2= f + mu*u + lambda;
        v = UnivPottsVectorParallel(f2/(1+mu),2*gamma/(1+mu),ones(M,N));

        lambda = lambda + mu*(u-v);
        mu = tau*mu;
        error(iter) = immse(u,v);
        error(iter)
        iter = iter+1;
    end
    %% resize back
    v=[];
    lambda=[];
    for i=1:S
        v(:,:,i) = kron(v0(:,:,i),ones(2,2));
        lambda(:,:,i) = kron(lambda0(:,:,i),ones(2,2));
    end
    v=v(1:M,1:N,:);
    lambda=lambda(1:M,1:N,:);
    mu=mu*2;
end

