# Nodal Network Metrics

These scripts compute **region-by-region connectivity characteristics** for all 246 Brainnectome regions, identifying which brain areas change most and their network roles.

## What is "Nodal"?

Nodal metrics characterize **individual brain regions** within the network:
- **Node strength**: Total connectivity of a region (sum of edge weights)
- **Degree**: Number of connections to other regions
- **Betweenness centrality**: How often a region lies on shortest paths (information routing)
- **Participation coefficient**: Balance between local and global connections
- **Within-module z-score**: Hub status within functional modules
- **Hub classification**: Provincial (local hub) vs Connector (global hub) vs Other

Result: **A value for each of 246 regions per subject per timepoint**

## Scripts in This Category

### 01_nodal_hub_identification.py
**Purpose:** Identify hub regions and classify their network role

**Input:**
- Connectivity matrices (246×246 per subject)

**Output:**
- Node-level metrics for all 246 regions
- Hub classification tables
- Temporal trajectories showing how each region evolves

**When to use:**
- To answer: "Which brain regions change most?"
- To answer: "Which regions are hubs? Do they change?"
- To identify candidate regions for further investigation

**Key outputs:**
- `node_metrics.parquet` - All metrics for 246 regions
- `hub_classification.parquet` - Hub type per region
- `regional_trajectories/` - Time-series changes per region

---

### 02_nodal_temporal_trajectories.py
**Purpose:** Analyze how individual regions change over the 4 timepoints (3 post-baseline)

**Input:**
- Node-level metrics from 01_nodal_hub_identification.py

**Output:**
- Statistical tests of temporal patterns
- Visualizations of regional change trajectories
- Group × time interactions

**When to use:**
- To understand temporal dynamics at the region level
- To test if intervention effects vary by region
- To identify regions with linear vs nonlinear changes

---

### 03_nodal_multivariate_analysis.py
**Purpose:** Analyze correlations and multivariate patterns across node-level metrics

**Input:**
- Node-level metrics

**Output:**
- Principal Component Analysis (PCA) of node metrics
- Correlation matrices
- Multivariate group comparisons

**When to use:**
- To understand relationships between different node metrics
- To reduce dimensionality while preserving information
- To test multivariate group differences

---

## Global vs Nodal: Quick Comparison

| Question | Script Category |
|----------|-----------------|
| "Is the whole network more efficient?" | Global metrics |
| "Which 5 regions change the most?" | Nodal metrics |
| "Do hubs differ from non-hubs?" | Nodal metrics |
| "Is network integration improved?" | Global metrics |
| "Which region drives the change?" | Nodal metrics |

## Typical Workflow

1. Run **01_nodal_hub_identification.py** → Understand regional change magnitude
2. Run **02_nodal_temporal_trajectories.py** → Characterize temporal patterns
3. Run **03_nodal_multivariate_analysis.py** → Understand relationships between metrics

## Next Steps

After nodal analysis:
- **4_statistical_classification/** - Can we predict group from regional patterns?
- **5_responder_analysis/** - Do responders vs non-responders show different nodal changes?
- **6_visualization/** - Visualize regional changes on brain surface

---

## Technical Notes

- 246 regions from Brainnectome atlas (BN_Atlas_246_1mm.nii.gz)
- Hub classification uses module detection (Louvain algorithm)
- Participation coefficient normalized by module size
- All metrics computed on weighted, undirected networks

