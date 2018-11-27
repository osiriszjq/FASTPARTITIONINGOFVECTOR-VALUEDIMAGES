function h = UnivPottsVectorParallel(f, gamma, w)
    % --------------------------------------------------------------------
    % This function finds the solution to the univariate Potts problems
    % by parallel computation.
    %
    % [Input]   f: the data vector f \in R^(Batch_size by N by s)
    %
    %           gamma: model parameter. small gamma results in detail
    %           partitions; large gamma results in partitions having fewer
    %           partitioning regions.
    %
    %           w: weights w \in (R+)^(Batch_size by N)
    %
    % [Output]  h: the resulting global minimizer to the univariate Potts
    %           problem. h \in R^(Batch_size by N by s)
    % --------------------------------------------------------------------
    [Batch_size, N, s] = size(f);
    M = cumsum(w.*f, 2); 
    S = cumsum(w.*f.^2, 2);  
    W = cumsum(w, 2);
    P = zeros(Batch_size, N); 
    for r = 1:N
        P(:,r) = sum((S(:,r,:) - M(:,r,:).^2 ./ W(:,r)), 3);
    end
    J = zeros(Batch_size, N);  
    h = zeros(Batch_size, N, s);
    % --------------------------------------------------------------------
    % Find the optimal jump locations
    for r = 2:N
        for l = r:-1:2
            d = sum((S(:,r,:) - S(:,l-1,:) ...
                - (M(:,r,:) - M(:,l-1,:)).^2./ (W(:,r) - W(:,l-1))) ,3);
            idx = find(P(:,r) >= d + gamma);
            
            if sum(idx) == 0
                break;
            end
            
            p = inf * ones(Batch_size,1);
            p(idx) = P(idx,l-1) + gamma + d(idx);
            idx = find(p<=P(:,r));
            if sum(idx) ~= 0
                P(idx,r) = p(idx);
                J(idx,r) = l-1;
            end
        end
    end
    
    % Reconstruct the minimizer h from the optimal jump locations
    for i = 1:Batch_size
        r = N; l = J(i,r);
        while(l > 0)
            h(i,l+1:r,:) = repmat((M(i,r,:)-M(i,l,:))/(W(i,r)-W(i,l)) ...
                                  , r-l, 1);
            r = l;
            l = J(i,r);
        end
        h(i,1:r,:) = repmat((M(i,r,:))/(W(i,r)), r-l, 1);
    end
end