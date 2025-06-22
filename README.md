# Metaheuristic-Optimization-of-Human-Pose-Populations
Refining 3D human pose estimation using metaheuristic algorithms (GA, PSO, ACO, GWO) on motion capture data with custom fitness metrics.

An advanced AI project focused on refining 3D human pose estimations using population-based metaheuristic optimization algorithms. The system enhances baseline predictions from motion capture data by minimizing pose errors with evolutionary algorithms such as Genetic Algorithm (GA), Particle Swarm Optimization (PSO), Ant Colony Optimization (ACO), and Grey Wolf Optimizer (GWO).

📌 Project Overview
This project improves the accuracy of human pose estimation models by refining predicted pose populations through metaheuristic optimization. It evaluates and compares multiple bio-inspired optimization strategies to align predicted poses more closely with ground truth data from motion capture systems.

We utilize real motion datasets (GTpose.mat, PopulationPoses.mat) and develop a robust evaluation framework incorporating both numerical metrics and qualitative visualizations.

🚀 Objectives
Apply and compare four metaheuristic algorithms for human pose refinement:

Genetic Algorithm (GA)

Particle Swarm Optimization (PSO)

Ant Colony Optimization (ACO)

Grey Wolf Optimizer (GWO)

Minimize pose deviation using a custom fitness function

Visualize pose improvement before and after optimization

Perform cluster-wise analysis of optimized poses

📂 Dataset
GTpose.mat — Ground truth 3D joint positions

PopulationPoses.mat — Initial population of candidate poses

Format: MATLAB .mat files (processed using SciPy / MATLAB engine)

⚙️ Optimization Metrics
The optimization is guided using a multi-factor fitness function based on:

Mean Joint Position Error (MJPE)

Percentage of Correct Keypoints (PCK)

Pose Estimation Accuracy (PEA)

These ensure the optimized poses are not only numerically accurate but also visually aligned with the ground truth.

📈 Algorithms Used
Algorithm	Highlights
GA (Genetic Algorithm)	Evolution-inspired; crossover and mutation for global search
PSO (Particle Swarm Optimization)	Particle-based velocity updates for fast convergence
ACO (Ant Colony Optimization)	Path-optimization with pheromone trail dynamics
GWO (Grey Wolf Optimizer)	Leadership hierarchy-based search behavior

🖼️ Visual Analysis
3D pose skeleton overlays (before vs. after)

Comparative convergence plots across algorithms

Clustering of optimized poses by performance groups

🧪 Evaluation Pipeline
Data Loading: Read and preprocess .mat files

Population Initialization: Retrieve candidate poses

Fitness Evaluation: MJPE + PCK + PEA metrics

Optimization Execution: Apply selected algorithm

Visualization & Comparison: Output overlays and stats

🛠️ Tech Stack
Languages: Python, MATLAB (for dataset prep)

Libraries: NumPy, SciPy, matplotlib, scikit-learn

Optimization Frameworks: Custom implementations for GA, PSO, ACO, GWO

🧠 Key Contributions
Designed a flexible fitness function to evaluate pose accuracy

Developed independent pipelines for each optimization strategy

Performed comparative benchmarking across algorithms

Visualized improvements in pose quality through 3D plots


