function [error, end_u] = ADMM8Vd(f, gamma, tau, mu0) %in paper, gamma = 0.5 or 2.0, tau = 2, mu0 = 0.01*gamma
    [M,N,S] = size(f);
    m=mod(1:M,2)==1;
    n=mod(1:N,2)==1;
    f0=f(m,n,:);
    
    [end_u, l10,l20,l30,l40,l50,l60,mu0] = ADMM48V_zjq(f0, gamma/2, tau, mu0/2);
    mu0=mu0*2;
    v=[];
    l1=[];
    l2=[];
    l3=[];
    l4=[];
    l5=[];
    l6=[];
    for i=1:S
        v(:,:,i) = kron(end_u(:,:,i),ones(2,2));
        l1(:,:,i) = kron(l10(:,:,i),ones(2,2));
        l2(:,:,i) = kron(l20(:,:,i),ones(2,2));
        l3(:,:,i) = kron(l30(:,:,i),ones(2,2));
        l4(:,:,i) = kron(l40(:,:,i),ones(2,2));
        l5(:,:,i) = kron(l50(:,:,i),ones(2,2));
        l6(:,:,i) = kron(l60(:,:,i),ones(2,2));
    end
    v=v(1:M,1:N,:);
    l1=l1(1:M,1:N,:);
    l2=l2(1:M,1:N,:);
    l3=l3(1:M,1:N,:);
    l4=l4(1:M,1:N,:);
    l5=l5(1:M,1:N,:);
    l6=l6(1:M,1:N,:);
    w = v;
    z = v;
    mu = mu0;
    stop=sum(sum(sum(f.^2)))*1e-12;
    %u_series = [];
    
    u = f;
    
    error = +Inf;
    wc = sqrt(2)-1;
    wd = 1-sqrt(2)/2;
    
    diag_coordinates = zeros(M*N,2); 
    antidiag_coordinates = zeros(M*N,2);
    start_point = zeros(M+N,1);

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
    
    iter = 1;
    while error > stop
        % col
        f1 = min(max(f,u-0.2),u+0.2)+2*mu*(v+w+z)-2*(l1+l2+l3);
        u = UPVP_zjq(permute(f1,[2,1,3])/(1+6*mu),4*gamma*wc/(1+6*mu));
        u = permute(u,[2,1,3]);
%         for i = 1:N
%             f1 = squeeze(f(:,i,:)+2*mu*(v(:,i,:)+w(:,i,:)+z(:,i,:))-2*(l1(:,i,:)+l2(:,i,:)+l3(:,i,:)));
%             u(:,i,:) = UnivPottsVectorParal(f1/(1+6*mu), 4*gamma*wc/(1+6*mu), wc*ones(M,1));
%         end
        %diagonal 1
        temp = min(max(f,w-0.2),w+0.2)+2*mu*(u+v+z)+2*(l2+l4-l6);
        for i = 1:M+N-1            
            segment = start_point(i):start_point(i+1)-1;
            vec_temp = zeros(length(segment),S);
            for s = 1:S
                temp_s = squeeze(temp(:,:,s));
                vec_temp(:,s) = temp_s(sub2ind([M,N], diag_coordinates(segment,1), diag_coordinates(segment,2)));
            end
            temp_w = UPV_zjq(vec_temp/(1+6*mu), 4*gamma*wd/(1+6*mu));
            for j = 1:length(segment)
                w(diag_coordinates(segment(j),1),diag_coordinates(segment(j),2),:) = temp_w(j,:);
            end
        end
        
        %row
        f3 = min(max(f,v-0.2),v+0.2)+2*mu*(u+w+z)+2*(l1-l4-l5);
        v = UPVP_zjq(f3/(1+6*mu),4*gamma*wc/(1+6*mu));
        %v = permute(v,[2,1,3]);
%         for i = 1:M
%             f3 = squeeze(f(i,:,:)+2*mu*(u(i,:,:)+w(i,:,:)+z(i,:,:))+2*(l1(i,:,:)-l4(i,:,:)-l5(i,:,:)));
%             v(i,:,:) = UnivPottsVector(f3/(1+6*mu), 4*gamma*wc/(1+6*mu), wc*ones(N,1));
%         end
        %diagonal 2
        temp = min(max(f,z-0.2),z+0.2)+2*mu*(u+v+w)+2*(l3+l5+l6);
        for i = 1:M+N-1            
            segment = start_point(i):start_point(i+1)-1;
            vec_temp = zeros(length(segment),S);
            for s = 1:S
                temp_s = squeeze(temp(:,:,s));
                vec_temp(:,s) = temp_s(sub2ind([M,N], antidiag_coordinates(segment,1), antidiag_coordinates(segment,2)));
            end
            temp_z = UPV_zjq(vec_temp/(1+6*mu), 4*gamma*wd/(1+6*mu));
            for j = 1:length(segment)
                z(antidiag_coordinates(segment(j),1), antidiag_coordinates(segment(j),2),:) = temp_z(j,:);
            end
        end
        l1 = l1 + mu*(u-v);
        l2 = l2 + mu*(u-w);
        l3 = l3 + mu*(u-z);
        l4 = l4 + mu*(v-w);
        l5 = l5 + mu*(v-z);
        l6 = l6 + mu*(w-z);
        mu = tau*mu;
        error = immse(u,v)
        %u_series(iter,:,:,:) = u;
        iter = iter+1;
    end
    end_u = u;
end