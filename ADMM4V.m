function [error, result] = ADMM4V(f, gamma, tau, mu0) 
% In paper, gamma = 0.5 or 2.0, tau = 2, mu0 = 0.01*gamma
    v = f;
    [M,N,S] = size(f);
    mu = mu0;
    lambda = zeros(M,N,S);
    u = zeros(M,N,S);
    error = +Inf;
    TOL = 1e-10*sum(sum(sum(f.^2)));
    
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
    result = v;
end