function fitness = evaluateFitness(positions, GTpose)
    % positions: [N×D] array of candidate poses
    % GTpose   : [1×D] ground-truth pose
    [N, ~] = size(positions);
    joints = numel(GTpose)/3;
    fitness = zeros(N,1);
    for i = 1:N
        cand3 = reshape(positions(i,:), 3, [])';  % [joints×3]
        gt3   = reshape(GTpose,        3, [])';  % [joints×3]
        d     = sqrt(sum((cand3 - gt3).^2, 2));  % per-joint distance
        fitness(i) = sum(d.^2);                  % sum of squared errors
    end
end
