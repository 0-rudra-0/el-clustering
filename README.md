# EL-Clustering: Code and Experiments

This repository contains the implementation and experimental suite for the paper **"EL-Clustering: Clustering with Equitable Load Constraints"** (under review). The tools provided allow for the complete replication of the results discussed in the paper.

This implementation validates a combination algorithm that is not only theoretically sound but also highly efficient, stable, and robust in practice. This repository contains the tools to explore and verify these findings.

---

## 1. Summary of Key Empirical Findings

Our comprehensive experiments across three distinct datasets validate the practical utility of the EL-Clustering algorithm. The key findings are:

1.  **High Practical Efficiency:** The algorithm consistently and significantly outperforms its theoretical cost guarantee. Even in its worst-performing scenarios, the final clustering cost is typically an order of magnitude lower than the theoretical maximum, showcasing a vast gap between the worst-case bound and the algorithm's excellent performance in practice.

<p align="center">
  <img src="plots/Combined cost comparison with theoretical guarantee (absolute).png" alt="Absolute Cost vs Theoretical Guarantee" width="45%"/>
  &nbsp;&nbsp;
  <img src="plots/Combined cost comparison with theoretical guarantee.png" alt="Relative Cost vs Theoretical Guarantee" width="45%"/>
</p>

2.  **Stable and Predictable Cost:** The core combination step introduces a minimal and predictable cost increase. The distribution of this increase is tight and centered near 1.1, indicating that unpredictable performance is rare. Across all datasets, the final cost is, in over 95% of cases, less than **1.169 times** the cost of the max(cost_S_L, cost_S_U) of initial heuristic solution.

<p align="center">
  <img src="plots/Adults (Obtained cost distribution).png" alt="Adults Obtained Cost Distribution" width="45%"/>
  &nbsp;&nbsp;
  <img src="plots/Bank (Obtained cost distribution).png" alt="Bank Obtained Cost Distribution" width="45%"/>
  &nbsp;&nbsp;
  <img src="plots/Diabetes (Obtained cost distribution).png" alt="Diabetes Obtained Cost Distribution" width="45%"/>
</p>

3.  **Robust Under Challenging Conditions:** The algorithm remains robust even under difficult parameterizations. Performance degradation (i.e., upper bound violation) is most likely to occur under a predictable combination of high `k` (a large number of clusters) and tight constraints (a small gap between `L` and `U`), demonstrating that the algorithm's behavior is well-understood and not erratic.


<p align="center">
  <img src="plots/Adults Avg. Violation Heatmap.png" alt="Adults Obtained Cost Distribution" width="45%"/>
  &nbsp;&nbsp;
  <img src="plots/Bank Avg. Violation Heatmap.png" alt="Bank Obtained Cost Distribution" width="45%"/>
  &nbsp;&nbsp;
  <img src="plots/Diabetes Avg. Violation Heatmap.png" alt="Diabetes Obtained Cost Distribution" width="45%"/>
</p>

---

## 2. Guide to Reproducing Results

The repository is structured to allow for easy and complete replication of our findings.

### Step 1: Setup
The code requires Python 3.8+ and standard scientific libraries.

```bash
# Install dependencies
pip install pandas matplotlib seaborn
```
## **Step 2: Running Experiments**  
  
The run_experiments.py script automates the full parameter sweep and generates .csv log files with detailed metrics for each run.
```Bash  
  
# Run experiments on all datasets and save logs to the ./logs/ directory  
# For a straightforward implementation, refer to comments starting at line 222
python run_experiments.py --datasets all --output_dir ./logs/  
```  
## **Step 3: Generating Analysis**  
  
The run_analysis.py script processes the log files to generate all analytical plots.  
```Bash  
  
# Generate all plots from the logs in ./logs/ and save them to ./plots/  
python run_analysis.py --log_dir ./logs/ --output_dir ./plots/  
```  
  
## **3. Algorithm and Code Description**  
  
  
## **Code Structure**  
  
The project is organized into two primary scripts:  
* run_experiments.py: The main experimental runner. It loads data, generates the S_L and S_U heuristic solutions, executes the core combination algorithm, and logs all results.  
* run_analysis.py: The plotting script. It reads the .csv logs and produces the analytical figures.  
  
## **Heuristic Algorithms**  
  
The two initial solutions are generated as follows:  
* **Lower-Bounded Heuristic (S_L):** This heuristic begins with a standard unconstrained k-Median assignment, which minimizes initial cost but may leave some clusters undersized. A repair phase follows, where clients from larger clusters are iteratively moved to the nearest facility that is below the lower-bound (L). This greedy reassignment continues until every facility's cluster size is at least L, guaranteeing compliance at the cost of a potential increase in the total distance.

* **Upper-Bounded Heuristic (S_U):** Upper-Bounded Heuristic (S_U): This method processes clients in a randomized sequential order to prevent pathological assignments. Each client is assigned to its nearest facility that has not yet reached its upper-bound (U) capacity. If the closest facility is full, the client considers its second-closest, third-closest, and so on, until an available facility is found. This ensures no upper bounds are violated, trading strict optimality for guaranteed capacity compliance.
  
## **Combination Algorithm**  
  
The core technique merges the two heuristic solutions (S_L, S_U) through a structured, graph-based approach to resolve assignment conflicts.

A directed "dependency graph" is built where an edge from facility i to facility j signifies that i's cluster in S_L contains clients that are assigned to j's group in S_U. A cycle in this graph represents a processing deadlock. To resolve this, cycles are systematically broken by re-routing a minimal number of client assignments in the S_L solution, which untangles the dependency at its weakest point while preserving all lower-bound guarantees.

With a deterministic order established via a topological sort of this graph, each facility group is processed. The final assignments are made by bundling clients to satisfy the L constraint, using the S_L facilities as guaranteed backstops to absorb remaining clients. This structured process ensures the final clustering respects both L and U constraints with a provably bounded increase in cost.  
  
  
## **4. Data Specification**  
  
  
## **Input Data Format (.py)**  
  
The run_experiments.py script expects a simple text file containing a Python-style list of lists, where each inner list represents a multi-dimensional data point.  
**Example (sample_data.py):**  
```python
[[0.15, 0.52, 0.28], [0.91, 0.33, 0.45], [0.22, 0.81, 0.79]]  
```
 
## **Internal Data Format (Python variable with JSON-style content)**  

The core combination algorithm operates on an intermediate Python file containing a variable in **JSON-style format**.  
This file is **generated automatically** by run_experiments.py. A user does not need to create this file manually.  
It bundles the heuristic solutions for processing.  

**Example (internal_run_data.py):**  

```python
data = {  
  "parameters": {"L": 50, "U": 150},  
  "clients": {  
    "c0": [0.15, 0.52], "c1": [0.91, 0.33]  
  },  
  "sl_facilities": {  
    "f1": [0.2, 0.5], "f2": [0.8, 0.4]  
  },  
  "su_facilities": {  
    "f3": [0.1, 0.6], "f4": [0.9, 0.2]  
  },  
  "sl_assignment": {  
    "c0": "f1", "c1": "f2"  
  },  
  "su_assignment": {  
    "c0": "f3", "c1": "f4"  
  }  
}
```
## 5. Datasets Used  

The experiments were conducted on three publicly available datasets from the **[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)**.  
To make them directly usable for clustering, the raw datasets were **preprocessed into Python list-of-lists**, where each inner list is a point with float-valued coordinates.

1. **Adult Income Dataset:** [Link](https://archive.ics.uci.edu/ml/datasets/adult) — Derived from U.S. Census data, using 6 continuous numerical features.  
2. **Pima Indians Diabetes Dataset:** [Link](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes) — Based on 10 years of clinical care data, using 10 numerical features.  
3. **Bank Marketing Dataset:** [Link](https://archive.ics.uci.edu/ml/datasets/bank+marketing) — From marketing campaigns of a Portuguese bank, using 7 numerical features.  

*Note: The datasets have been modified and provided in the form of Python lists of float coordinate points for clustering. Some features may have been removed or transformed.*
