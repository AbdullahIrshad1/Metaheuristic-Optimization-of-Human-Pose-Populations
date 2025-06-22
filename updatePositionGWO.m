function newPos = updatePositionGWO(pop, alpha, beta, delta, a)
    % Update positions of grey wolves based on alpha, beta, delta
    % pop: [N x D], alpha/beta/delta: [1 x D], a: scalar coefficient
    [N, D] = size(pop);
    newPos = zeros(N, D);
    for i = 1:N
        for j = 1:D
            r1 = rand(); r2 = rand(); A1 = 2*a*r1 - a; C1 = 2*r2;
            D_alpha = abs(C1*alpha(j) - pop(i,j)); X1 = alpha(j) - A1*D_alpha;
            r1 = rand(); r2 = rand(); A2 = 2*a*r1 - a; C2 = 2*r2;
            D_beta  = abs(C2*beta(j)  - pop(i,j)); X2 = beta(j)  - A2*D_beta;
            r1 = rand(); r2 = rand(); A3 = 2*a*r1 - a; C3 = 2*r2;
            D_delta = abs(C3*delta(j) - pop(i,j)); X3 = delta(j) - A3*D_delta;
            newPos(i,j) = (X1 + X2 + X3) / 3;
        end
    end
end
