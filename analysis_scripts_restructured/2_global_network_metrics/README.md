# Global Network Metrics

These scripts compute **whole-brain connectivity characteristics** that summarize the entire network into single metrics per subject per timepoint.

## What is "Global"?

Global metrics characterize the **entire network as a single entity**:
- **Path length**: Average shortest path between all brain regions
- **Global efficiency**: How efficiently information flows across the whole network
- **Small-worldness**: Balance between segregation and integration
- **Clustering coefficient**: Tendency to form local clusters

Result: **One number per subject per timepoint** (e.g., global efficiency = 0.45)

## Scripts in This Category

### 01_global_basic_metrics.py
**Purpose:** Compute basic global connectivity metrics and perform dimensionality reduction

**Input:**
- Connectivity matrices from Test_matrizen/ses-{1,2,3,4}/

**Output:**
- Global metrics table (path length, efficiency, small-worldness, etc.)
- UMAP coordinates for trajectory visualization
- Trajectory plots showing how network topology changes over time

**When to use:**
- First step in understanding overall network changes
- To see if network becomes more/less efficiently organized

**Key outputs:**
- `global_metrics.parquet` - Numeric metrics for each subject × timepoint
- `umap_coordinates.parquet` - Low-dimensional representation
- `trajectory_plots/` - Visualization folder

---

## Next Steps

After running global metrics, proceed to:
- **3_nodal_network_metrics/** - Understand *which regions* drive these changes
- **4_statistical_classification/** - Can ML predict group from these metrics?

---

## Technical Notes

- Uses `bct` (Brain Connectivity Toolbox) for graph theory metrics
- Requires connected graph (excludes isolated nodes)
- Path length computed on undirected, binarized networks
- UMAP uses Euclidean distance on standardized metrics

