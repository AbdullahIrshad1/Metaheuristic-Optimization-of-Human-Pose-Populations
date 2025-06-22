function visualizePoses(candidate, GTpose)
    % candidate: [1×D] optimized pose
    % GTpose   : [1×D] ground-truth pose
    joints = numel(candidate)/3;
    c3 = reshape(candidate, 3, [])';  % [joints×3]
    g3 = reshape(GTpose,    3, [])';  % [joints×3]
    plot3(g3(:,1), g3(:,2), g3(:,3), 'ro-', 'LineWidth',1.5); hold on;
    plot3(c3(:,1), c3(:,2), c3(:,3), 'bx-', 'LineWidth',1.5);
    xlabel('X'); ylabel('Y'); zlabel('Z');
    legend('Ground Truth','Optimized'); grid on; axis equal;
    view(30,30);
end
