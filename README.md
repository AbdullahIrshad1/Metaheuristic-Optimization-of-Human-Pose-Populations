# Metaheuristic-Optimization-of-Human-Pose-Populations
Refining 3D human pose estimation using metaheuristic algorithms (GA, PSO, ACO, GWO) on motion capture data with custom fitness metrics.

An advanced AI project focused on refining 3D human pose estimations using population-based metaheuristic optimization algorithms. The system enhances baseline predictions from motion capture data by minimizing pose errors with evolutionary algorithms such as Genetic Algorithm (GA), Particle Swarm Optimization (PSO), Ant Colony Optimization (ACO), and Grey Wolf Optimizer (GWO).

## 📌 Project Overview

This advanced AI project enhances the accuracy of 3D human pose estimation by refining predicted pose populations using population-based metaheuristic optimization techniques. Baseline motion capture predictions are improved by minimizing pose estimation errors via Genetic Algorithm (GA), Particle Swarm Optimization (PSO), Ant Colony Optimization (ACO), and Grey Wolf Optimizer (GWO).  
The system uses real-world motion capture datasets and a multi-factor fitness function to optimize predicted human poses and visualize the improvements through 3D plots and clustering analysis.

---

## 🚀 Objectives

| Goal | Description |
|------|-------------|
| **🔁 Multi-Algorithm Optimization** | Apply and compare four bio-inspired metaheuristic algorithms for human pose refinement. |
| **📉 Error Minimization** | Use custom fitness metrics to reduce Mean Joint Position Error (MJPE) and increase keypoint accuracy. |
| **📊 Visual Improvement** | Overlay 3D skeletons to compare before vs. after optimization performance. |
| **🔬 Cluster-wise Evaluation** | Group and analyze optimized pose populations based on performance similarity. |

---

## 📂 Dataset

| File | Description |
|------|-------------|
| `GTpose.mat` | Ground truth 3D joint positions (motion capture-based) |
| `PopulationPoses.mat` | Initial population of estimated poses |
| **Format** | MATLAB `.mat` files processed via Python using `SciPy` or MATLAB Engine API |

---

## ⚙️ Optimization Metrics

The optimization process is guided by a composite fitness function integrating:

- **MJPE** – Mean Joint Position Error  
- **PCK** – Percentage of Correct Keypoints  
- **PEA** – Pose Estimation Accuracy  

These ensure both numerical precision and visual fidelity of the refined human poses.

---

## 🧠 Algorithms Used

| Algorithm | Description |
|-----------|-------------|
| **🧬 Genetic Algorithm (GA)** | Evolution-based search using crossover and mutation operators |
| **🌪 Particle Swarm Optimization (PSO)** | Particle-based search using velocity updates for convergence |
| **🐜 Ant Colony Optimization (ACO)** | Simulates ant pathfinding via pheromone-based reinforcement |
| **🐺 Grey Wolf Optimizer (GWO)** | Mimics social hierarchy and hunting behavior of grey wolves |

---

## 🖼️ Visual Analysis

| Visualization | Description |
|---------------|-------------|
| **🎯 3D Pose Overlays** | Side-by-side skeleton comparisons before and after optimization |
| **📈 Convergence Plots** | Graphical analysis of algorithm convergence over generations |
| **🧩 Clustering Analysis** | Group optimized poses by performance metrics using unsupervised clustering |

---

## 🧪 Evaluation Pipeline

1. **Data Loading** – Read and preprocess `.mat` files  
2. **Population Initialization** – Retrieve candidate poses for optimization  
3. **Fitness Evaluation** – Compute MJPE, PCK, and PEA  
4. **Optimization Execution** – Run GA / PSO / ACO / GWO on population  
5. **Visualization & Comparison** – Generate plots and overlay results

---

## 🛠️ Tech Stack

- **Languages**: Python, MATLAB (for preprocessing)
- **Libraries**: NumPy, SciPy, matplotlib, scikit-learn
- **Optimization Frameworks**: Custom implementations for GA, PSO, ACO, GWO

---

## ✨ Key Contributions

- Designed a flexible fitness function to balance multiple error metrics  
- Developed modular pipelines for each optimization strategy  
- Performed comparative benchmarking across all algorithms  
- Visualized improvements with comprehensive 3D analysis and clustering

Performed comparative benchmarking across algorithms

Visualized improvements in pose quality through 3D plots



