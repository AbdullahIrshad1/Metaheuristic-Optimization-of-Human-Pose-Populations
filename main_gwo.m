%% main_gwo.m
% Grey Wolf Optimization for Human Pose Optimization

clc; clear;

% 1) Load data
d1 = load('datasetPoses.mat');         % contains datasetPoses [93×2462]
population = d1.datasetPoses';         % [2462×93]

d2 = load('GTPose.mat');               % contains queryPose [93×1]
GTPose = d2.queryPose';                % [1×93]

% 2) GWO parameters
maxIter = 100;
[N, D] = size(population);

% 3) Initialize alpha, beta, delta
alpha = zeros(1, D);    alphaScore = inf;
beta  = zeros(1, D);    betaScore  = inf;
delta = zeros(1, D);    deltaScore = inf;
pos    = population;

% 4) Prepare tracking
bestFitPerIter = zeros(1, maxIter);
mjpePerIter    = zeros(1, maxIter);
pckPerIter     = zeros(1, maxIter);
timePerIter    = zeros(1, maxIter);

% Derive number of joints and threshold
numJoints = numel(GTPose) / 3;
threshold = 0.1;

% 5) Main GWO loop
for iter = 1:maxIter
    tStart = tic;
    a = 2 - iter * (2 / maxIter);

    % a) Evaluate fitness for all wolves
    fitness = evaluateFitness(pos, GTPose);

    % b) Update alpha, beta, delta
    [sortedFit, idxSort] = sort(fitness);
    alphaScore = sortedFit(1); alpha = pos(idxSort(1), :);
    betaScore  = sortedFit(2); beta  = pos(idxSort(2), :);
    deltaScore = sortedFit(3); delta = pos(idxSort(3), :);

    % c) Update positions
    pos = updatePositionGWO(pos, alpha, beta, delta, a);

    % d) Record metrics on alpha (best)
    bestFitPerIter(iter) = alphaScore;
    coords  = reshape(alpha, 3, []).';
    gtcoords= reshape(GTPose,3, []).';
    dists   = sqrt(sum((coords - gtcoords).^2,2));
    mjpePerIter(iter) = mean(dists);
    pckPerIter(iter)  = sum(dists < threshold) / numJoints;
    timePerIter(iter) = toc(tStart);

    % e) Log every 10 iters
    if mod(iter,10)==0
        fprintf('Iter %3d/%d — Fit: %.4f, MJPE: %.4f, PCK: %.2f%%, Time: %.3fs\n', ...
            iter, maxIter, bestFitPerIter(iter), mjpePerIter(iter), pckPerIter(iter)*100, timePerIter(iter));
    end
end

% 6) Final results
fprintf('\nGWO Completed. Best Fitness: %.4f, MJPE: %.4f, PCK: %.2f%%\n', ...
    bestFitPerIter(end), mjpePerIter(end), pckPerIter(end)*100);

% 7) Plot convergence
figure;
subplot(3,1,1);
plot(1:maxIter, bestFitPerIter, 'LineWidth',2);
xlabel('Iteration'); ylabel('Fitness'); title('GWO Fitness'); grid on;
subplot(3,1,2);
plot(1:maxIter, mjpePerIter, 'LineWidth',2);
xlabel('Iteration'); ylabel('MJPE'); title('GWO MJPE'); grid on;
subplot(3,1,3);
plot(1:maxIter, pckPerIter*100, 'LineWidth',2);
xlabel('Iteration'); ylabel('PCK (%)'); title('GWO PCK'); grid on;

% 8) Display & save table
Iteration = (1:maxIter).';
BestFit   = bestFitPerIter.';
MJPE      = mjpePerIter.';
PCK       = (pckPerIter*100).';
Time_s    = timePerIter.';
T = table(Iteration, BestFit, MJPE, PCK, Time_s);
disp(T);
writetable(T, 'GWO_Iteration_Metrics.csv');

% 9) Visualize best pose
figure;
visualizePoses(alpha, GTPose);
title('GWO Optimized Pose vs. Ground Truth');
