%% Genetic Algorithm for Human Pose Optimization
% Implementation of Genetic Algorithm for "Metaheuristic Optimization of Human Pose 
% Populations for Enhanced Pose Estimation" project
%
% This algorithm optimizes a population of candidate poses to better match
% ground truth pose data, using GA operations of selection, crossover, and mutation.

% Clear workspace and close figures
clear all;
close all;
clc;

%% Load Data
% Load ground truth poses and initial population
try
    load('GTPose.mat');  % Ground truth pose data
    load('datasetPoses.mat');  % Initial population
    disp('Data loaded successfully.');
catch
    error('Error loading data files. Make sure GTPose.mat and datasetPoses.mat are in the current directory.');
end

% After loading files, assign to proper variables
GTPose = queryPose;       % Ground truth pose from the query pose
PopulationPoses = datasetPoses; % Initial population of poses

% Display basic information about the data
fprintf('Ground Truth Pose dimension: %d x %d\n', size(GTPose));
fprintf('Population Poses dimension: %d x %d\n', size(PopulationPoses));

%% Parameters for Genetic Algorithm
params.populationSize = min(size(PopulationPoses, 2), 100);  % Size of population (use available or cap at 100)
params.maxGenerations = 100;  % Maximum number of generations
params.crossoverProbability = 0.8;  % Probability of crossover
params.mutationProbability = 0.1;  % Probability of mutation
params.eliteCount = 5;  % Number of elites to preserve in each generation
params.tournamentSize = 5;  % Tournament selection size

%% Initialize Variables
% Check if required data is loaded
if ~exist('datasetPoses', 'var')
    error('Population Poses not found in the loaded data');
end

if ~exist('GTPose', 'var')
    error('GTPose not found in the loaded data');
end

% Initialize population
population = PopulationPoses;
[dim, populationSize] = size(population); % Joint parameters in rows, poses in columns

% Adjust parameters if needed
params.populationSize = populationSize;

% Calculate number of joints correctly
numJoints = size(GTPose, 1) / 3;
if mod(numJoints, 1) ~= 0
    error('GTPose dimensions not compatible with 3D joint structure (should be divisible by 3)');
end
numJoints = floor(numJoints);
fprintf('Detected %d joints in the poses\n', numJoints);

% Initialize storage for results and metrics
bestFitness = zeros(params.maxGenerations, 1);
meanFitness = zeros(params.maxGenerations, 1);
bestPose = zeros(params.maxGenerations, dim);
metrics_history = struct('MJPE', zeros(params.maxGenerations, 1), ...
                         'PCK', zeros(params.maxGenerations, 1), ...
                         'PEA', zeros(params.maxGenerations, 1));

% Timer for computational efficiency
tic;

%% Start Genetic Algorithm
fprintf('\n===== Starting Genetic Algorithm for Human Pose Optimization =====\n');

% Initial fitness evaluation
fitness = evaluateFitness(population, GTPose);
[bestFitVal, bestIdx] = min(fitness);
bestPose(1, :) = population(:, bestIdx)';
bestFitness(1) = bestFitVal;
meanFitness(1) = mean(fitness);

% Calculate initial metrics
[metrics] = calculateMetrics(population(:, bestIdx), GTPose);
metrics_history.MJPE(1) = metrics.MJPE;
metrics_history.PCK(1) = metrics.PCK;
metrics_history.PEA(1) = metrics.PEA;

fprintf('Initial Best Fitness: %.4f\n', bestFitVal);
fprintf('Initial MJPE: %.4f, PCK: %.2f%%, PEA: %.4f\n', ...
    metrics.MJPE, metrics.PCK*100, metrics.PEA);

% Main GA loop
for generation = 2:params.maxGenerations
    % Selection - tournament selection
    parents = selection(population, fitness, params);
    
    % Crossover - two-point crossover
    offspring = crossover(parents, params);
    
    % Mutation - gaussian mutation
    offspring = mutation(offspring, params);
    
    % Elitism - preserve the best individuals
    [~, eliteIndices] = sort(fitness);
    elite = population(:, eliteIndices(1:params.eliteCount));
    
    % Create new population
    population(:, eliteIndices(1:params.eliteCount)) = elite;
    population(:, eliteIndices(params.eliteCount+1:end)) = offspring(:, 1:(populationSize-params.eliteCount));
    
    % Evaluate fitness of new population
    fitness = evaluateFitness(population, GTPose);
    
    % Store best and mean fitness
    [bestFitVal, bestIdx] = min(fitness);
    bestPose(generation, :) = population(:, bestIdx)';
    bestFitness(generation) = bestFitVal;
    meanFitness(generation) = mean(fitness);
    
    % Calculate metrics for this generation's best pose
    [metrics] = calculateMetrics(population(:, bestIdx), GTPose);
    metrics_history.MJPE(generation) = metrics.MJPE;
    metrics_history.PCK(generation) = metrics.PCK;
    metrics_history.PEA(generation) = metrics.PEA;
    
    % Display progress
    if mod(generation, 10) == 0 || generation == params.maxGenerations
        fprintf('Generation %d/%d: Best Fitness = %.4f, Mean Fitness = %.4f\n', ...
            generation, params.maxGenerations, bestFitness(generation), meanFitness(generation));
        fprintf('MJPE: %.4f, PCK: %.2f%%, PEA: %.4f\n', ...
            metrics.MJPE, metrics.PCK*100, metrics.PEA);
    end
end

% Record execution time
executionTime = toc;
fprintf('Genetic Algorithm completed in %.2f seconds.\n', executionTime);

%% Evaluation and Results
% Find the best solution from all generations
[finalBestFitness, bestGeneration] = min(bestFitness);
finalBestPose = bestPose(bestGeneration, :);

% Calculate final metrics
[finalMetrics] = calculateMetrics(reshape(finalBestPose', [], 1), GTPose);

% Display comprehensive results
fprintf('\n===== Genetic Algorithm Results =====\n');
fprintf('Best Generation: %d\n', bestGeneration);
fprintf('Best Fitness: %.4f\n', finalBestFitness);
fprintf('Mean Joint Position Error (MJPE): %.4f\n', finalMetrics.MJPE);
fprintf('Percentage of Correct Keypoints (PCK): %.2f%%\n', finalMetrics.PCK*100);
fprintf('Pose Error Area (PEA): %.4f\n', finalMetrics.PEA);
fprintf('Computational Efficiency:\n');
fprintf('  - Total Execution Time: %.2f seconds\n', executionTime);
fprintf('  - Average Time per Generation: %.4f seconds\n', executionTime/params.maxGenerations);
fprintf('  - Average Time per Fitness Evaluation: %.6f seconds\n', ...
    executionTime/(params.maxGenerations*params.populationSize));

%% Plot Results
% Figure 1: Fitness convergence
figure;
subplot(2,1,1);
plot(1:params.maxGenerations, bestFitness, 'b-', 'LineWidth', 2);
hold on;
plot(1:params.maxGenerations, meanFitness, 'r--', 'LineWidth', 1);
xlabel('Generation');
ylabel('Fitness (Error)');
title('Genetic Algorithm Convergence');
legend('Best Fitness', 'Mean Fitness');
grid on;

% Figure 2: Metrics over generations
subplot(2,1,2);
plot(1:params.maxGenerations, metrics_history.MJPE, 'b-', 'LineWidth', 2); hold on;
plot(1:params.maxGenerations, metrics_history.PEA/10, 'g-', 'LineWidth', 1);
plot(1:params.maxGenerations, metrics_history.PCK, 'r-', 'LineWidth', 1);
xlabel('Generation');
ylabel('Metric Value');
title('Performance Metrics Over Generations');
legend('MJPE', 'PEA/10', 'PCK');
grid on;

% Figure 3: Visualize the best pose vs ground truth
figure;
visualizePoses(finalBestPose, GTPose, 'Genetic Algorithm: Best Pose vs Ground Truth');

% Save results for comparison with other algorithms
save('GA_Results.mat', 'finalBestPose', 'bestFitness', 'meanFitness', ...
     'metrics_history', 'executionTime', 'finalMetrics', 'params');

%% Helper Functions

function fitness = evaluateFitness(population, GTpose)
    % Calculate fitness as the sum of squared distances between joints
    % Lower fitness is better (representing less error)
    [dim, populationSize] = size(population);
    fitness = zeros(1, populationSize);
    
    % Calculate number of joints
    numJoints = size(GTpose, 1) / 3;
    numJoints = floor(numJoints);
    
    for i = 1:populationSize
        % Get candidate pose
        candidatePose = population(:, i);
        
        % Reshape poses to 3 x numJoints format for comparison
        reshapedCandidate = reshape(candidatePose, 3, int32(numJoints));
        reshapedGT = reshape(GTpose, 3, int32(numJoints));
        
        % Calculate joint-wise Euclidean distances
        diff = reshapedCandidate - reshapedGT;
        squaredDist = sum(diff.^2, 1); % Sum across x,y,z dimensions
        jointDistances = sqrt(squaredDist);
        
        % Fitness is sum of squared distances (lower is better)
        fitness(i) = sum(jointDistances.^2);
    end
end

function parents = selection(population, fitness, params)
    % Tournament selection: select parents for crossover
    [dim, populationSize] = size(population);
    parents = zeros(size(population));
    
    for i = 1:populationSize
        % Select random candidates for tournament
        candidates = randperm(populationSize, params.tournamentSize);
        [~, idx] = min(fitness(candidates));
        winner = candidates(idx);
        parents(:, i) = population(:, winner);
    end
end

function offspring = crossover(parents, params)
    % Two-point crossover operation
    [dim, numParents] = size(parents);
    offspring = parents; % Initialize with parents
    
    for i = 1:2:numParents-1
        if rand < params.crossoverProbability
            % Select two crossover points
            points = sort(randperm(dim, 2));
            
            % Perform two-point crossover
            temp = offspring(points(1):points(2), i);
            offspring(points(1):points(2), i) = offspring(points(1):points(2), i+1);
            offspring(points(1):points(2), i+1) = temp;
        end
    end
end

function offspring = mutation(offspring, params)
    % Gaussian mutation operation
    [dim, numOffspring] = size(offspring);
    
    for i = 1:numOffspring
        for j = 1:dim
            if rand < params.mutationProbability
                % Apply gaussian mutation
                offspring(j, i) = offspring(j, i) + randn * 0.1;
            end
        end
    end
end

function [metrics] = calculateMetrics(pose, GTpose)
    % Calculate evaluation metrics as defined in the project requirements
    
    % Calculate number of joints
    numJoints = size(GTpose, 1) / 3;
    numJoints = floor(numJoints);
    
    % Reshape poses for metric calculation
    reshapedPose = reshape(pose, 3, int32(numJoints));
    reshapedGT = reshape(GTpose, 3, int32(numJoints));
    
    % Calculate Euclidean distances for each joint
    diff = reshapedPose - reshapedGT;
    squaredDist = sum(diff.^2, 1);
    jointDistances = sqrt(squaredDist);
    
    % Mean Joint Position Error (MJPE)
    metrics.MJPE = mean(jointDistances);
    
    % Percentage of Correct Keypoints (PCK)
    threshold = 0.2; % Can be adjusted based on requirements
    metrics.PCK = sum(jointDistances < threshold) / numJoints;
    
    % Pose Error Area (PEA)
    metrics.PEA = sum(jointDistances);
end

function visualizePoses(candidatePose, GTpose, titleStr)
    % Visualize the ground truth and optimized poses
    
    % Calculate number of joints
    numJoints = length(GTpose) / 3;
    numJoints = floor(numJoints);
    
    % Reshape poses for visualization (to [numJoints x 3] for easier plotting)
    if size(candidatePose, 1) == 1 % If candidatePose is a row vector
        candidatePose = candidatePose';
    end
    
    reshapedCandidate = reshape(candidatePose, 3, int32(numJoints))';
    reshapedGT = reshape(GTpose, 3, int32(numJoints))';
    
    % Create 3D plot
    figure;
    plot3(reshapedGT(:,1), reshapedGT(:,2), reshapedGT(:,3), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    hold on;
    plot3(reshapedCandidate(:,1), reshapedCandidate(:,2), reshapedCandidate(:,3), 'bx', 'MarkerSize', 8, 'LineWidth', 2);
    
    % Connect joints to visualize skeleton structure
    % This creates a simple skeleton - adjust based on actual joint structure
    % For full human skeleton, you may need a more complex connection pattern
    for i = 1:(numJoints-1)
        line([reshapedGT(i,1), reshapedGT(i+1,1)], ...
             [reshapedGT(i,2), reshapedGT(i+1,2)], ...
             [reshapedGT(i,3), reshapedGT(i+1,3)], 'Color', 'r', 'LineWidth', 1);
        line([reshapedCandidate(i,1), reshapedCandidate(i+1,1)], ...
             [reshapedCandidate(i,2), reshapedCandidate(i+1,2)], ...
             [reshapedCandidate(i,3), reshapedCandidate(i+1,3)], 'Color', 'b', 'LineWidth', 1);
    end
    
    % Set plot properties
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title(titleStr);
    legend('Ground Truth', 'Optimized Pose');
    grid on;
    axis equal;
    view(3); % Set 3D view
    rotate3d on; % Enable 3D rotation
end