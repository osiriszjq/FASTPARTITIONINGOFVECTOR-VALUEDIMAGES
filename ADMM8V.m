function [error, result] = ADMM8V(f, gamma, tau, mu0) %in paper, gamma = 0.5 or 2.0, tau = 2, mu0 = 0.01*gamma
    v = f;
    w = f;
    z = f;
    [M,N,S] = size(f);
    mu = mu0;
    
    l1 = zeros(M,N,S);
    l2 = zeros(M,N,S);
    l3 = zeros(M,N,S);
    l4 = zeros(M,N,S);
    l5 = zeros(M,N,S);
    l6 = zeros(M,N,S);
    u = zeros(M,N,S);
    
    error = +Inf;
    wc = sqrt(2)-1;
    wd = 1-sqrt(2)/2;
    
    diag_coordinates = zeros(M*N,2); 
    antidiag_coordinates = zeros(M*N,2);
    start_point = zeros(M+N,1);
    TOL = 1e-13*sum(sum(sum(f.^2)))*6;

    counter = 1;
    for i = 1:M+N-1 
        start_point(i) = counter;
        for j = 1:min([M, N, i, M+N-i]) 
            diag_coordinates(counter,1) = max(i-N+1, 1) + j-1;
            diag_coordinates(counter,2) = i - diag_coordinates(counter,1) + 1;
            counter = counter + 1;
        end
    end
    start_point(end) = start_point(end-1) + 1;
    antidiag_coordinates(:,1) = M+1-diag_coordinates(:,1);
    antidiag_coordinates(:,2) = diag_coordinates(:,2);
    
    wcc = ones(N,M);
    wcc(sum(f,3)==0) = 0; 
    
    iter = 1;
    while error > TOL
        % cols
        f1 = f+2*mu*(v+w+z)-2*(l1+l2+l3);
        
        u = UnivPottsVectorParallel(permute(f1,[2,1,3])./(wcc+6*mu),4*gamma*wc,wcc+mu);
        u = permute(u,[2,1,3]);
        
        f2 = f+2*mu*(u+v+z)+2*(l2+l4-l6);
        for i = 1:M+N-1 
            segment = start_point(i):start_point(i+1)-1;    
            for s = 1:S
                temp_s = squeeze(f2(:,:,s));
                temp(i,1:size(segment,2),s) = temp_s(sub2ind([M,N], diag_coordinates(segment,1), diag_coordinates(segment,2)));
            end
        end
        wdd = ones(M+N-1,min(M,N));
        wdd(sum(temp,3)==0) = 0; 
        temp = UnivPottsVectorParallel(temp./(wdd+6*mu), 4*gamma*wd, wdd+mu);
        for s = 1:S   
            for i = 1:M+N-1
                segment = start_point(i):start_point(i+1)-1;
                temp_s(sub2ind([M,N], diag_coordinates(segment,1), diag_coordinates(segment,2))) = temp(i,1:size(segment,2),s);     
            end
            w(:,:,s) = temp_s;
        end
        
        %row
        f3 = f+2*mu*(u+w+z)+2*(l1-l4-l5);
        v = UnivPottsVectorParallel(f3./(wcc'+6*mu),4*gamma*wc,wcc'+mu);

        f4 = f+2*mu*(u+v+w)+2*(l3+l5+l6);
        for i = 1:M+N-1 
            segment = start_point(i):start_point(i+1)-1;    
            for s = 1:S
                temp_s = squeeze(f4(:,:,s));
                temp(i,1:size(segment,2),s) = temp_s(sub2ind([M,N], antidiag_coordinates(segment,1), antidiag_coordinates(segment,2)));
            end
        end
        wdd = 1*ones(M+N-1,min(M,N));
        wdd(sum(temp,3)==0) = 0; 
        temp = UnivPottsVectorParallel(temp./(wdd+6*mu), 4*gamma*wd, wdd+mu);
        for s = 1:S   
            for i = 1:M+N-1
                segment = start_point(i):start_point(i+1)-1;
                temp_s(sub2ind([M,N], antidiag_coordinates(segment,1), antidiag_coordinates(segment,2))) = temp(i,1:size(segment,2),s);     
            end
            z(:,:,s) = temp_s;
        end
        
        l1 = l1 + mu*(u-v);
        l2 = l2 + mu*(u-w);
        l3 = l3 + mu*(u-z);
        l4 = l4 + mu*(v-w);
        l5 = l5 + mu*(v-z);
        l6 = l6 + mu*(w-z);
        
        mu = tau*mu;
        
        error(iter) = immse(u,v)+immse(u,w)+immse(u,z)+immse(v,w)+immse(v,z)+immse(w,z);
        error(iter)
        iter = iter+1;
    end
    result = round(u,3);
end