"""
Enhanced Brain Connectivity Toolbox Analysis Module
Supports DSI Studio BIDS output, multiple file formats, comprehensive BCT metrics,
and matrix preprocessing options.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import bct
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import warnings
warnings.filterwarnings('ignore')


class BCTAnalyzer:
    """
    Enhanced BCT analysis engine with support for:
    - DSI Studio BIDS folder structure
    - Multiple file formats (CSV, NPY, MAT, Excel, TSV)
    - Comprehensive BCT metrics (50+ metrics)
    - Matrix preprocessing (thresholding, normalization, binarization)
    - Flexible atlas and metric selection
    """
    
    # Supported file formats
    SUPPORTED_FORMATS = ['.npy', '.csv', '.mat', '.xlsx', '.xls', '.tsv', '.txt']
    
    # DSI Studio metrics available
    DSI_STUDIO_METRICS = [
        'count', 'ncount', 'ncount2', 'mean_length',
        'fa', 'qa', 'ad', 'rd', 'md', 'iso', 'rdi', 'ndi',
        'dti_fa', 'dti_ad', 'dti_rd', 'dti_md'
    ]
    
    # Known brain atlases with their typical region counts
    ATLAS_INFO = {
        68: ("Desikan-Killiany (FreeSurfer)", "DK68"),
        78: ("Brodmann", "Brodmann"),
        82: ("AAL", "AAL"),
        84: ("FreeSurferDKT Cortical", "DKT84"),
        90: ("AAL90", "AAL90"),
        116: ("AAL116", "AAL116"),
        120: ("AAL2", "AAL2"),
        166: ("AAL3", "AAL3"),
        170: ("AAL3", "AAL3"),
        200: ("Schaefer200", "Schaefer200"),
        246: ("Brainnetome", "BN246"),
        264: ("Power264", "Power264"),
        333: ("Gordon333", "Gordon333"),
        360: ("HCP-MMP (Glasser)", "HCP360"),
        379: ("HCP-MMP", "HCP379"),
        400: ("Schaefer400", "Schaefer400"),
        100: ("Schaefer100", "Schaefer100"),
        384: ("AICHA", "AICHA"),
    }
    
    # Comprehensive metric groups
    METRIC_GROUPS = {
        'degree': {
            'description': 'Node degree and strength measures',
            'metrics': ['degree', 'strength', 'in_degree', 'out_degree', 'in_strength', 'out_strength']
        },
        'density': {
            'description': 'Network density and sparsity',
            'metrics': ['density', 'connection_count', 'sparsity']
        },
        'clustering': {
            'description': 'Clustering and transitivity',
            'metrics': ['clustering_coef', 'transitivity', 'local_efficiency']
        },
        'modularity': {
            'description': 'Community structure',
            'metrics': ['modularity', 'participation_coef', 'module_degree_zscore', 'community_structure']
        },
        'efficiency': {
            'description': 'Network efficiency measures',
            'metrics': ['global_efficiency', 'local_efficiency', 'nodal_efficiency']
        },
        'centrality': {
            'description': 'Centrality measures',
            'metrics': ['betweenness', 'eigenvector_centrality', 'pagerank', 'subgraph_centrality', 'kcore']
        },
        'path': {
            'description': 'Path length and distance measures',
            'metrics': ['char_path_length', 'eccentricity', 'radius', 'diameter']
        },
        'assortativity': {
            'description': 'Degree correlations',
            'metrics': ['assortativity', 'rich_club']
        },
        'smallworld': {
            'description': 'Small-world properties',
            'metrics': ['small_worldness', 'sigma', 'omega']
        },
        'resilience': {
            'description': 'Network resilience',
            'metrics': ['largest_component', 'vulnerability']
        }
    }
    
    def __init__(self, 
                 output_callback=None, 
                 selected_metrics: Optional[List[str]] = None,
                 output_format: str = 'parquet',
                 atlas_filter: Optional[str] = None,
                 dsi_metric: str = 'count',
                 threshold: Optional[float] = None,
                 threshold_type: str = 'absolute',
                 normalize: bool = False,
                 binarize: bool = False):
        """
        Initialize Enhanced BCT Analyzer
        
        Args:
            output_callback: Optional callback function for logging
            selected_metrics: List of metric groups to calculate
            output_format: Output format ('parquet', 'feather', 'hdf5', 'csv', 'excel')
            atlas_filter: Filter for specific atlas (e.g., 'AAL3', 'Schaefer200')
            dsi_metric: DSI Studio metric to use ('count', 'fa', 'qa', etc.)
            threshold: Threshold value for matrix (None = no threshold)
            threshold_type: 'absolute', 'proportional', 'density'
            normalize: Whether to normalize weights
            binarize: Whether to binarize the matrix
        """
        self.output_callback = output_callback or self._default_log
        self.output_format = output_format.lower()
        self.atlas_filter = atlas_filter
        self.dsi_metric = dsi_metric
        self.threshold = threshold
        self.threshold_type = threshold_type
        self.normalize = normalize
        self.binarize = binarize
        
        # Validate output format
        valid_formats = ['parquet', 'feather', 'hdf5', 'csv', 'excel']
        if self.output_format not in valid_formats:
            self.log(f"⚠️ Invalid output format '{output_format}', defaulting to 'parquet'")
            self.output_format = 'parquet'
        
        # Set selected metrics
        if not selected_metrics:
            self.selected_metrics = list(self.METRIC_GROUPS.keys())
        else:
            self.selected_metrics = selected_metrics
    
    def _default_log(self, message: str):
        """Default logging function"""
        print(message)
    
    def log(self, message: str):
        """Log message via callback"""
        self.output_callback(message)
    
    # =========================================================================
    # FILE LOADING METHODS
    # =========================================================================
    
    def load_matrix(self, filepath: str) -> Optional[np.ndarray]:
        """
        Load connectivity matrix from various file formats
        
        Args:
            filepath: Path to matrix file
            
        Returns:
            Numpy array or None if loading fails
        """
        filepath = Path(filepath)
        ext = filepath.suffix.lower()
        
        try:
            if ext == '.npy':
                return np.load(filepath)
            
            elif ext == '.csv':
                # Try different CSV formats
                try:
                    # DSI Studio format with headers
                    df = pd.read_csv(filepath, index_col=0)
                    return df.values.astype(float)
                except:
                    # Simple CSV without headers
                    return np.loadtxt(filepath, delimiter=',')
            
            elif ext == '.mat':
                try:
                    from scipy.io import loadmat
                    mat_data = loadmat(filepath)
                    # Find the matrix variable (usually the largest array)
                    for key, value in mat_data.items():
                        if not key.startswith('_') and isinstance(value, np.ndarray):
                            if len(value.shape) == 2 and value.shape[0] == value.shape[1]:
                                return value.astype(float)
                except ImportError:
                    self.log("⚠️ scipy not available for .mat files")
                    return None
            
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(filepath, index_col=0, header=0)
                return df.values.astype(float)
            
            elif ext == '.tsv':
                df = pd.read_csv(filepath, sep='\t', index_col=0)
                return df.values.astype(float)
            
            elif ext == '.txt':
                # Try space/tab delimited
                try:
                    return np.loadtxt(filepath)
                except:
                    return np.loadtxt(filepath, delimiter='\t')
            
            else:
                self.log(f"⚠️ Unsupported file format: {ext}")
                return None
                
        except Exception as e:
            self.log(f"❌ Error loading {filepath}: {e}")
            return None
    
    # =========================================================================
    # BIDS/DSI STUDIO PARSING
    # =========================================================================
    
    def parse_bids_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse BIDS-style filename to extract subject, session, etc.
        
        Handles formats like:
        - sub-1293175_ses-1.odf.qsdr_Brodmann.tt.gz.Brodmann.count..end.connectivity.csv
        - sub-119BPAF161001_ses-1_preproc.nii.gz...connectogram.npy
        
        Returns:
            Dictionary with 'subject', 'session', 'atlas', 'metric', etc.
        """
        info = {
            'subject': 'unknown',
            'session': 'unknown',
            'atlas': 'unknown',
            'metric': 'unknown',
            'file_type': 'unknown'
        }
        
        # Extract subject (sub-XXXXX)
        sub_match = re.search(r'sub-([A-Za-z0-9]+)', filename)
        if sub_match:
            info['subject'] = f"sub-{sub_match.group(1)}"
        
        # Extract session (ses-X)
        ses_match = re.search(r'ses-(\d+)', filename)
        if ses_match:
            info['session'] = f"ses-{ses_match.group(1)}"
        
        # Extract atlas name
        atlas_patterns = [
            r'\.([A-Za-z0-9\-]+)\.tt\.gz',  # DSI Studio format
            r'\.([A-Za-z0-9\-]+)\.count\.',
            r'\.([A-Za-z0-9\-]+)\.(fa|qa|ad|rd|md|ncount)',
            r'Brodmann|AAL\d*|Schaefer\d+|HCP-MMP|Power\d+|Gordon\d+|FreeSurfer|AICHA|Talairach'
        ]
        
        for pattern in atlas_patterns:
            atlas_match = re.search(pattern, filename, re.IGNORECASE)
            if atlas_match:
                info['atlas'] = atlas_match.group(1) if atlas_match.lastindex else atlas_match.group(0)
                break
        
        # Extract DSI Studio metric
        for metric in self.DSI_STUDIO_METRICS:
            if f'.{metric}.' in filename.lower():
                info['metric'] = metric
                break
        
        # Determine file type
        if 'connectivity' in filename.lower():
            info['file_type'] = 'connectivity'
        elif 'connectogram' in filename.lower():
            info['file_type'] = 'connectogram'
        elif 'network_measures' in filename.lower():
            info['file_type'] = 'network_measures'
        
        return info
    
    def discover_dsi_studio_structure(self, input_dir: str) -> Dict:
        """
        Discover DSI Studio output structure
        
        Args:
            input_dir: Root directory to search
            
        Returns:
            Dictionary describing found structure
        """
        structure = {
            'subjects': [],
            'sessions': set(),
            'atlases': set(),
            'metrics': set(),
            'files_by_subject': {},
            'total_files': 0
        }
        
        input_path = Path(input_dir)
        
        # Look for subject folders (sub-XXXX_ses-X.odf.qsdr_timestamp format)
        for item in input_path.iterdir():
            if item.is_dir() and 'sub-' in item.name:
                bids_info = self.parse_bids_filename(item.name)
                subject = bids_info['subject']
                session = bids_info['session']
                
                structure['subjects'].append(subject)
                structure['sessions'].add(session)
                
                if subject not in structure['files_by_subject']:
                    structure['files_by_subject'][subject] = {}
                if session not in structure['files_by_subject'][subject]:
                    structure['files_by_subject'][subject][session] = []
                
                # Look for atlas folders
                atlas_path = item / 'tracks_1000k_streamline' / 'by_atlas'
                if atlas_path.exists():
                    for atlas_dir in atlas_path.iterdir():
                        if atlas_dir.is_dir():
                            atlas_name = atlas_dir.name
                            
                            # Find connectivity files (only non-simple CSV files)
                            csv_files = [f for f in atlas_dir.glob('*connectivity.csv') 
                                        if '.simple.' not in f.name.lower()]
                            
                            # Only add atlas if it has connectivity files
                            if csv_files:
                                structure['atlases'].add(atlas_name)
                                
                                for f in csv_files:
                                    file_info = self.parse_bids_filename(f.name)
                                    structure['metrics'].add(file_info['metric'])
                                    structure['files_by_subject'][subject][session].append({
                                        'path': str(f),
                                        'atlas': atlas_name,
                                        'metric': file_info['metric'],
                                        'type': file_info['file_type']
                                    })
                                    structure['total_files'] += 1
        
        # Convert sets to sorted lists
        structure['subjects'] = sorted(set(structure['subjects']))
        structure['sessions'] = sorted(structure['sessions'])
        structure['atlases'] = sorted(structure['atlases'])
        structure['metrics'] = sorted(structure['metrics'])
        
        return structure
    
    # =========================================================================
    # MATRIX PREPROCESSING
    # =========================================================================
    
    def preprocess_matrix(self, A: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing steps to connectivity matrix
        
        Args:
            A: Raw connectivity matrix
            
        Returns:
            Preprocessed matrix
        """
        A = A.copy().astype(float)
        
        # Ensure diagonal is zero
        np.fill_diagonal(A, 0)
        
        # Remove negative values (unusual for structural connectivity)
        A[A < 0] = 0
        
        # Apply threshold
        if self.threshold is not None:
            A = self._apply_threshold(A)
        
        # Normalize
        if self.normalize:
            A = self._normalize_matrix(A)
        
        # Binarize
        if self.binarize:
            A = (A > 0).astype(float)
        
        return A
    
    def _apply_threshold(self, A: np.ndarray) -> np.ndarray:
        """Apply threshold to matrix"""
        if self.threshold_type == 'absolute':
            # Remove edges below threshold value
            A[A < self.threshold] = 0
        
        elif self.threshold_type == 'proportional':
            # Keep top X% of edges
            if 0 < self.threshold <= 1:
                threshold_val = np.percentile(A[A > 0], (1 - self.threshold) * 100)
                A[A < threshold_val] = 0
        
        elif self.threshold_type == 'density':
            # Target specific edge density
            n = A.shape[0]
            max_edges = n * (n - 1) / 2
            target_edges = int(self.threshold * max_edges)
            
            # Get threshold that achieves target density
            flat = A[np.triu_indices(n, k=1)]
            flat_sorted = np.sort(flat)[::-1]
            if target_edges < len(flat_sorted):
                threshold_val = flat_sorted[target_edges]
                A[A < threshold_val] = 0
        
        return A
    
    def _normalize_matrix(self, A: np.ndarray) -> np.ndarray:
        """Normalize matrix weights to [0, 1]"""
        max_val = np.max(A)
        if max_val > 0:
            A = A / max_val
        return A
    
    # =========================================================================
    # QUALITY CONTROL
    # =========================================================================
    
    def quality_check(self, A: np.ndarray, filename: str = "") -> Dict:
        """
        Comprehensive quality check on connectivity matrix
        """
        qc = {
            "filename": filename,
            "passed": True,
            "warnings": [],
            "errors": []
        }
        
        # Check: Square matrix
        if A.shape[0] != A.shape[1]:
            qc["errors"].append(f"Not square: shape {A.shape}")
            qc["passed"] = False
            return qc
        
        n_nodes = A.shape[0]
        qc["n_nodes"] = n_nodes
        
        # Detect atlas from matrix size
        if n_nodes in self.ATLAS_INFO:
            qc["likely_atlas"] = self.ATLAS_INFO[n_nodes][0]
            qc["atlas_code"] = self.ATLAS_INFO[n_nodes][1]
        else:
            qc["likely_atlas"] = f"Unknown ({n_nodes} regions)"
            qc["atlas_code"] = f"N{n_nodes}"
        
        # Check diagonal
        diag_values = np.diag(A)
        qc["diagonal_ok"] = np.allclose(diag_values, 0)
        if not qc["diagonal_ok"]:
            qc["warnings"].append(f"Non-zero diagonal values detected")
        
        # Check symmetry
        qc["is_symmetric"] = np.allclose(A, A.T)
        
        # Sparsity and density
        upper_tri = np.triu(A, k=1)
        max_edges = n_nodes * (n_nodes - 1) // 2
        nonzero_count = np.count_nonzero(upper_tri)
        qc["sparsity"] = float(1 - nonzero_count / max_edges)
        qc["edge_density"] = float(nonzero_count / max_edges)
        qc["edge_count"] = int(nonzero_count)
        
        # Weight statistics
        nonzero_weights = A[A > 0]
        if len(nonzero_weights) > 0:
            qc["weight_min"] = float(np.min(nonzero_weights))
            qc["weight_max"] = float(np.max(nonzero_weights))
            qc["weight_mean"] = float(np.mean(nonzero_weights))
            qc["weight_std"] = float(np.std(nonzero_weights))
            qc["weight_median"] = float(np.median(nonzero_weights))
        else:
            qc["errors"].append("Matrix has no connections")
            qc["passed"] = False
        
        # Check for isolated nodes
        node_degree = np.sum(A > 0, axis=0) + np.sum(A > 0, axis=1)
        qc["isolated_nodes"] = int(np.sum(node_degree == 0))
        if qc["isolated_nodes"] > 0:
            qc["warnings"].append(f"{qc['isolated_nodes']} isolated nodes")
        
        # Check for NaN/Inf
        if np.any(np.isnan(A)):
            qc["errors"].append("Contains NaN values")
            qc["passed"] = False
        if np.any(np.isinf(A)):
            qc["errors"].append("Contains Inf values")
            qc["passed"] = False
        
        # Check for negative values
        qc["has_negative"] = bool(np.any(A < 0))
        if qc["has_negative"]:
            qc["warnings"].append("Contains negative weights")
        
        return qc
    
    # =========================================================================
    # BCT METRIC CALCULATIONS
    # =========================================================================
    
    @staticmethod
    def detect_matrix_type(A: np.ndarray) -> str:
        """Detect matrix type: BU, WU, BD, WD"""
        if A.shape[0] != A.shape[1]:
            return "unknown"
        
        is_symmetric = np.allclose(A, A.T)
        is_binary = np.allclose(A, A.astype(bool).astype(float))
        
        if is_symmetric:
            return "BU" if is_binary else "WU"
        else:
            return "BD" if is_binary else "WD"
    
    def calculate_comprehensive_metrics(self, A: np.ndarray, matrix_type: str) -> Dict:
        """
        Calculate comprehensive BCT metrics
        
        Includes 50+ metrics across all categories
        """
        metrics = {}
        A = np.array(A, dtype=float)
        A_bin = (A > 0).astype(float)
        n = A.shape[0]
        is_weighted = matrix_type in ["WU", "WD"]
        is_undirected = matrix_type in ["BU", "WU"]
        
        # =================================================================
        # DEGREE AND STRENGTH
        # =================================================================
        if 'degree' in self.selected_metrics:
            try:
                if is_undirected:
                    deg = bct.degrees_und(A_bin)
                    metrics['degree'] = deg
                    metrics['avg_degree'] = float(np.mean(deg))
                    metrics['max_degree'] = float(np.max(deg))
                    metrics['min_degree'] = float(np.min(deg))
                    metrics['std_degree'] = float(np.std(deg))
                    
                    if is_weighted:
                        strength = bct.strengths_und(A)
                        metrics['strength'] = strength
                        metrics['avg_strength'] = float(np.mean(strength))
                        metrics['max_strength'] = float(np.max(strength))
                else:
                    in_deg, out_deg, _ = bct.degrees_dir(A_bin)
                    metrics['in_degree'] = in_deg
                    metrics['out_degree'] = out_deg
                    metrics['avg_in_degree'] = float(np.mean(in_deg))
                    metrics['avg_out_degree'] = float(np.mean(out_deg))
                    
                    if is_weighted:
                        in_str, out_str, _ = bct.strengths_dir(A)
                        metrics['in_strength'] = in_str
                        metrics['out_strength'] = out_str
            except Exception as e:
                self.log(f"⚠️ Degree calculation error: {e}")
        
        # =================================================================
        # DENSITY
        # =================================================================
        if 'density' in self.selected_metrics:
            try:
                if is_undirected:
                    dens = bct.density_und(A_bin)
                else:
                    dens = bct.density_dir(A_bin)
                metrics['density'] = float(dens[0]) if isinstance(dens, tuple) else float(dens)
            except Exception as e:
                self.log(f"⚠️ Density calculation error: {e}")
        
        # =================================================================
        # CLUSTERING AND TRANSITIVITY
        # =================================================================
        if 'clustering' in self.selected_metrics:
            try:
                if matrix_type == "BU":
                    cc = bct.clustering_coef_bu(A_bin)
                    trans = bct.transitivity_bu(A_bin)
                elif matrix_type == "WU":
                    cc = bct.clustering_coef_wu(A)
                    trans = bct.transitivity_wu(A)
                elif matrix_type == "BD":
                    cc = bct.clustering_coef_bd(A_bin)
                    trans = bct.transitivity_bd(A_bin)
                elif matrix_type == "WD":
                    cc = bct.clustering_coef_wd(A)
                    trans = bct.transitivity_wd(A)
                
                metrics['clustering_coef'] = cc
                metrics['avg_clustering'] = float(np.mean(cc))
                metrics['transitivity'] = float(trans)
            except Exception as e:
                self.log(f"⚠️ Clustering calculation error: {e}")
        
        # =================================================================
        # EFFICIENCY
        # =================================================================
        if 'efficiency' in self.selected_metrics:
            try:
                if is_weighted:
                    # Convert to connection lengths (invert weights)
                    L = A.copy()
                    L[L > 0] = 1 / L[L > 0]
                    ge = bct.efficiency_wei(A)
                    le = bct.efficiency_wei(A, local=True)
                else:
                    ge = bct.efficiency_bin(A_bin)
                    le = bct.efficiency_bin(A_bin, local=True)
                
                metrics['global_efficiency'] = float(ge)
                metrics['local_efficiency'] = le
                metrics['avg_local_efficiency'] = float(np.mean(le))
            except Exception as e:
                self.log(f"⚠️ Efficiency calculation error: {e}")
        
        # =================================================================
        # PATH LENGTH
        # =================================================================
        if 'path' in self.selected_metrics:
            try:
                if is_weighted:
                    # Convert to lengths
                    L = A.copy()
                    L[L > 0] = 1 / L[L > 0]
                    D = bct.distance_wei(L)
                else:
                    D = bct.distance_bin(A_bin)
                
                D_result = D[0] if isinstance(D, tuple) else D
                
                # Replace inf with nan for calculations
                D_finite = D_result.copy()
                D_finite[np.isinf(D_finite)] = np.nan
                
                charpath = bct.charpath(D_result, include_diagonal=False, include_infinite=False)
                if isinstance(charpath, tuple):
                    metrics['char_path_length'] = float(charpath[0])
                    metrics['efficiency_from_charpath'] = float(charpath[1]) if len(charpath) > 1 else np.nan
                    metrics['eccentricity'] = charpath[2] if len(charpath) > 2 else None
                else:
                    metrics['char_path_length'] = float(charpath)
                
                # Eccentricity, radius, diameter
                ecc = np.nanmax(D_finite, axis=1)
                metrics['eccentricity'] = ecc
                metrics['radius'] = float(np.nanmin(ecc[ecc > 0])) if np.any(ecc > 0) else np.nan
                metrics['diameter'] = float(np.nanmax(ecc[np.isfinite(ecc)])) if np.any(np.isfinite(ecc)) else np.nan
                
            except Exception as e:
                self.log(f"⚠️ Path length calculation error: {e}")
        
        # =================================================================
        # CENTRALITY
        # =================================================================
        if 'centrality' in self.selected_metrics:
            try:
                # Betweenness centrality
                if is_weighted:
                    L = A.copy()
                    L[L > 0] = 1 / L[L > 0]
                    bc = bct.betweenness_wei(L)
                else:
                    bc = bct.betweenness_bin(A_bin)
                
                metrics['betweenness'] = bc
                metrics['avg_betweenness'] = float(np.mean(bc))
                metrics['max_betweenness'] = float(np.max(bc))
                
                # Eigenvector centrality (undirected only)
                if is_undirected:
                    try:
                        ec = bct.eigenvector_centrality_und(A if is_weighted else A_bin)
                        metrics['eigenvector_centrality'] = ec
                        metrics['avg_eigenvector'] = float(np.mean(ec))
                    except:
                        pass
                
                # Subgraph centrality
                try:
                    sc = bct.subgraph_centrality(A_bin)
                    metrics['subgraph_centrality'] = sc
                    metrics['avg_subgraph_centrality'] = float(np.mean(sc))
                except:
                    pass
                
                # K-core
                try:
                    if is_undirected:
                        kcore, kn, peelorder, peellevel = bct.kcore_bu(A_bin)
                        metrics['kcore'] = kcore
                        metrics['max_kcore'] = int(np.max(kcore))
                except:
                    pass
                
            except Exception as e:
                self.log(f"⚠️ Centrality calculation error: {e}")
        
        # =================================================================
        # MODULARITY AND COMMUNITY
        # =================================================================
        if 'modularity' in self.selected_metrics:
            try:
                # Community detection using Louvain
                if is_weighted:
                    ci, q = bct.community_louvain(A)
                else:
                    ci, q = bct.community_louvain(A_bin)
                
                metrics['modularity'] = float(q)
                metrics['community_structure'] = ci
                metrics['n_communities'] = int(len(np.unique(ci)))
                
                # Participation coefficient
                try:
                    pc = bct.participation_coef(A if is_weighted else A_bin, ci)
                    metrics['participation_coef'] = pc
                    metrics['avg_participation'] = float(np.mean(pc))
                except:
                    pass
                
                # Module degree z-score
                try:
                    mdz = bct.module_degree_zscore(A if is_weighted else A_bin, ci)
                    metrics['module_degree_zscore'] = mdz
                    metrics['avg_module_zscore'] = float(np.mean(mdz))
                except:
                    pass
                
            except Exception as e:
                self.log(f"⚠️ Modularity calculation error: {e}")
        
        # =================================================================
        # ASSORTATIVITY
        # =================================================================
        if 'assortativity' in self.selected_metrics:
            try:
                if matrix_type == "BU":
                    assort = bct.assortativity_bin(A_bin, flag=0)
                elif matrix_type == "WU":
                    assort = bct.assortativity_wei(A, flag=0)
                else:
                    assort = np.nan
                
                metrics['assortativity'] = float(assort) if not isinstance(assort, tuple) else float(assort[0])
                
                # Rich club coefficient
                if is_undirected:
                    try:
                        rc = bct.rich_club_bu(A_bin)
                        if isinstance(rc, dict):
                            metrics['rich_club'] = rc
                        else:
                            metrics['rich_club'] = rc
                    except:
                        pass
                
            except Exception as e:
                self.log(f"⚠️ Assortativity calculation error: {e}")
        
        # =================================================================
        # SMALL-WORLD PROPERTIES
        # =================================================================
        if 'smallworld' in self.selected_metrics:
            try:
                # Small-world sigma (requires random network comparison)
                # We'll compute a basic version using clustering and path length
                if 'avg_clustering' in metrics and 'char_path_length' in metrics:
                    C = metrics['avg_clustering']
                    L = metrics['char_path_length']
                    
                    # Estimate random network values
                    k = metrics.get('avg_degree', 2 * metrics.get('density', 0.1) * (n - 1))
                    C_rand = k / n  # Expected clustering for random network
                    L_rand = np.log(n) / np.log(k) if k > 1 else n  # Expected path length
                    
                    if C_rand > 0 and L_rand > 0:
                        gamma = C / C_rand
                        lam = L / L_rand
                        sigma = gamma / lam if lam > 0 else np.nan
                        
                        metrics['gamma'] = float(gamma)
                        metrics['lambda'] = float(lam)
                        metrics['small_world_sigma'] = float(sigma)
            except Exception as e:
                self.log(f"⚠️ Small-world calculation error: {e}")
        
        # =================================================================
        # RESILIENCE
        # =================================================================
        if 'resilience' in self.selected_metrics:
            try:
                # Get components
                comp, comp_sizes = bct.get_components(A_bin)
                metrics['n_components'] = int(len(np.unique(comp)))
                metrics['largest_component_size'] = int(np.max(np.bincount(comp)))
                metrics['largest_component_fraction'] = float(metrics['largest_component_size'] / n)
                
            except Exception as e:
                self.log(f"⚠️ Resilience calculation error: {e}")
        
        return metrics
    
    # =========================================================================
    # MAIN ANALYSIS FUNCTIONS
    # =========================================================================
    
    def analyze_dsi_studio(self, input_dir: str, output_dir: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Analyze DSI Studio output with BIDS structure
        
        Args:
            input_dir: Root directory containing DSI Studio output
            output_dir: Directory for output files
            
        Returns:
            Tuple of (results_dataframe, summary)
        """
        self.log(f"📁 Input directory: {input_dir}")
        self.log(f"📁 Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Discover structure
        self.log("\n🔍 Discovering DSI Studio structure...")
        structure = self.discover_dsi_studio_structure(input_dir)
        
        if not structure['subjects']:
            self.log("⚠️ No DSI Studio subject folders found")
            self.log("   Looking for standard folder structure...")
            return self.analyze_standard_structure(input_dir, output_dir)
        
        self.log(f"✓ Found {len(structure['subjects'])} subjects")
        self.log(f"✓ Sessions: {', '.join(structure['sessions'])}")
        self.log(f"✓ Atlases: {', '.join(structure['atlases'])}")
        self.log(f"✓ Metrics: {', '.join(structure['metrics'])}")
        self.log(f"✓ Total files: {structure['total_files']}")
        
        # Filter by atlas if specified
        target_atlas = self.atlas_filter
        if target_atlas:
            self.log(f"\n🎯 Filtering for atlas: {target_atlas}")
        
        # Process files
        all_data = []
        qc_summary = {
            'total_files': 0,
            'passed_qc': 0,
            'warnings': 0,
            'errors': 0,
            'atlases_used': {}
        }
        
        self.log(f"\n📊 Processing matrices (metric: {self.dsi_metric})...\n")
        
        files_checked = 0
        files_skipped_atlas = 0
        files_skipped_metric = 0
        files_skipped_simple = 0
        
        for subject in structure['subjects']:
            for session in structure['files_by_subject'].get(subject, {}):
                files = structure['files_by_subject'][subject][session]
                
                for file_info in files:
                    files_checked += 1
                    
                    # Filter by atlas (case-insensitive)
                    if target_atlas:
                        if file_info['atlas'].lower() != target_atlas.lower():
                            files_skipped_atlas += 1
                            continue
                    
                    # Filter by metric (case-insensitive)
                    if file_info['metric'].lower() != self.dsi_metric.lower():
                        files_skipped_metric += 1
                        continue
                    
                    # Skip simple CSV files (use the labeled ones)
                    if '.simple.' in file_info['path'].lower():
                        files_skipped_simple += 1
                        continue
                    
                    filepath = file_info['path']
                    self.log(f"  → {subject} / {session} / {file_info['atlas']}")
                    
                    # Load matrix
                    A = self.load_matrix(filepath)
                    if A is None:
                        qc_summary['errors'] += 1
                        continue
                    
                    # Quality check
                    qc = self.quality_check(A, os.path.basename(filepath))
                    qc_summary['total_files'] += 1
                    
                    if qc['passed']:
                        qc_summary['passed_qc'] += 1
                    qc_summary['warnings'] += len(qc['warnings'])
                    
                    atlas_name = file_info['atlas']
                    qc_summary['atlases_used'][atlas_name] = qc_summary['atlases_used'].get(atlas_name, 0) + 1
                    
                    # Preprocess
                    A = self.preprocess_matrix(A)
                    
                    # Detect matrix type
                    matrix_type = self.detect_matrix_type(A)
                    
                    # Calculate metrics
                    metrics = self.calculate_comprehensive_metrics(A, matrix_type)
                    
                    # Build data row
                    data_row = {
                        'subject': subject,
                        'session': session,
                        'atlas': atlas_name,
                        'dsi_metric': file_info['metric'],
                        'matrix_type': matrix_type,
                        'n_nodes': qc['n_nodes'],
                        'edge_count': qc.get('edge_count', 0),
                        'edge_density': qc.get('edge_density', 0),
                        'qc_passed': qc['passed'],
                        'qc_warnings': len(qc['warnings']),
                        'weight_mean': qc.get('weight_mean', np.nan),
                        'weight_std': qc.get('weight_std', np.nan),
                    }
                    
                    # Add computed metrics (scalars only for main table)
                    for key, value in metrics.items():
                        if isinstance(value, (int, float, np.floating, np.integer)):
                            data_row[key] = float(value)
                        elif isinstance(value, np.ndarray):
                            # Store summary stats for vector metrics
                            data_row[f'{key}_mean'] = float(np.mean(value))
                            data_row[f'{key}_std'] = float(np.std(value))
                    
                    all_data.append(data_row)
        
        # Log filtering stats
        self.log(f"\n📈 Filter statistics:")
        self.log(f"   Files checked: {files_checked}")
        if target_atlas:
            self.log(f"   Skipped (atlas mismatch): {files_skipped_atlas}")
        self.log(f"   Skipped (metric mismatch): {files_skipped_metric}")
        self.log(f"   Skipped (simple files): {files_skipped_simple}")
        
        # Create results dataframe
        if all_data:
            df_results = pd.DataFrame(all_data)
            self.log(f"\n✓ Processed {len(all_data)} matrices")
            
            # Save results
            self._save_results(df_results, output_dir)
            
            # Save QC report
            self._save_qc_report(qc_summary, output_dir)
            
            summary = {
                'total_matrices': len(all_data),
                'subjects': df_results['subject'].unique().tolist(),
                'sessions': df_results['session'].unique().tolist(),
                'atlases': list(qc_summary['atlases_used'].keys()),
                'quality_control': qc_summary
            }
            
            return df_results, summary
        else:
            self.log("❌ No matrices were successfully processed")
            return pd.DataFrame(), {}
    
    def analyze_standard_structure(self, input_dir: str, output_dir: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Analyze standard folder structure (ses-1, ses-2, etc. or flat)
        
        Supports multiple file formats
        """
        self.log(f"📁 Analyzing standard structure: {input_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        all_data = []
        qc_summary = {'total_files': 0, 'passed_qc': 0, 'warnings': 0, 'errors': 0, 'atlases_used': {}}
        
        input_path = Path(input_dir)
        
        # Check for session folders
        session_folders = sorted([d for d in input_path.iterdir() if d.is_dir() and 'ses' in d.name.lower()])
        
        if session_folders:
            self.log(f"✓ Found session folders: {[f.name for f in session_folders]}")
            search_dirs = session_folders
        else:
            self.log("  No session folders, searching in root directory")
            search_dirs = [input_path]
        
        for search_dir in search_dirs:
            session = search_dir.name if search_dir != input_path else 'unknown'
            
            # Find all supported files
            for ext in self.SUPPORTED_FORMATS:
                for filepath in search_dir.glob(f'*{ext}'):
                    # Skip if it's a network_measures or non-matrix file
                    if 'network_measures' in filepath.name or 'connectogram' in filepath.name:
                        continue
                    
                    self.log(f"  → {filepath.name}")
                    
                    # Load matrix
                    A = self.load_matrix(str(filepath))
                    if A is None:
                        qc_summary['errors'] += 1
                        continue
                    
                    # Quality check
                    qc = self.quality_check(A, filepath.name)
                    qc_summary['total_files'] += 1
                    
                    if qc['passed']:
                        qc_summary['passed_qc'] += 1
                    qc_summary['warnings'] += len(qc['warnings'])
                    
                    # Parse filename for subject info
                    bids_info = self.parse_bids_filename(filepath.name)
                    
                    # Preprocess
                    A = self.preprocess_matrix(A)
                    
                    # Detect matrix type
                    matrix_type = self.detect_matrix_type(A)
                    
                    # Calculate metrics
                    metrics = self.calculate_comprehensive_metrics(A, matrix_type)
                    
                    # Build data row
                    data_row = {
                        'subject': bids_info['subject'],
                        'session': session if session != 'unknown' else bids_info['session'],
                        'filename': filepath.name,
                        'atlas': qc.get('likely_atlas', 'unknown'),
                        'matrix_type': matrix_type,
                        'n_nodes': qc['n_nodes'],
                        'edge_density': qc.get('edge_density', 0),
                        'qc_passed': qc['passed'],
                    }
                    
                    # Add metrics
                    for key, value in metrics.items():
                        if isinstance(value, (int, float, np.floating, np.integer)):
                            data_row[key] = float(value)
                        elif isinstance(value, np.ndarray):
                            data_row[f'{key}_mean'] = float(np.mean(value))
                    
                    all_data.append(data_row)
        
        if all_data:
            df_results = pd.DataFrame(all_data)
            self.log(f"\n✓ Processed {len(all_data)} matrices")
            self._save_results(df_results, output_dir)
            self._save_qc_report(qc_summary, output_dir)
            
            return df_results, {'total_matrices': len(all_data), 'quality_control': qc_summary}
        
        return pd.DataFrame(), {}
    
    def analyze_matrices(self, input_dir: str, output_dir: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Main entry point - automatically detects structure type
        """
        input_path = Path(input_dir)
        
        # Check if it's a DSI Studio structure
        has_dsi_structure = any(
            'sub-' in item.name and item.is_dir() 
            for item in input_path.iterdir()
        )
        
        if has_dsi_structure:
            return self.analyze_dsi_studio(input_dir, output_dir)
        else:
            return self.analyze_standard_structure(input_dir, output_dir)
    
    # =========================================================================
    # OUTPUT METHODS
    # =========================================================================
    
    def _save_results(self, df_results: pd.DataFrame, output_dir: str):
        """Save results in specified format"""
        try:
            if self.output_format == 'parquet':
                output_file = os.path.join(output_dir, "bct_analysis_results.parquet")
                df_results.to_parquet(output_file, index=False)
                self.log(f"✓ Results saved to: {output_file}")
                
            elif self.output_format == 'feather':
                output_file = os.path.join(output_dir, "bct_analysis_results.feather")
                df_results.to_feather(output_file)
                self.log(f"✓ Results saved to: {output_file}")
                
            elif self.output_format == 'hdf5':
                import h5py
                output_file = os.path.join(output_dir, "bct_analysis_results.h5")
                df_results.to_hdf(output_file, key='bct_results', mode='w')
                self.log(f"✓ Results saved to: {output_file}")
                
            elif self.output_format == 'csv':
                output_file = os.path.join(output_dir, "bct_analysis_results.csv")
                df_results.to_csv(output_file, index=False)
                self.log(f"✓ Results saved to: {output_file}")
                
            elif self.output_format == 'excel':
                output_file = os.path.join(output_dir, "bct_analysis_results.xlsx")
                df_results.to_excel(output_file, index=False)
                self.log(f"✓ Results saved to: {output_file}")
                
        except Exception as e:
            self.log(f"⚠️ Error saving in {self.output_format} format: {e}")
            self.log("Falling back to CSV...")
            output_file = os.path.join(output_dir, "bct_analysis_results.csv")
            df_results.to_csv(output_file, index=False)
            self.log(f"✓ Results saved to: {output_file}")
    
    def _save_qc_report(self, qc_summary: Dict, output_dir: str):
        """Save quality control report"""
        qc_report_path = os.path.join(output_dir, "quality_control_report.txt")
        with open(qc_report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("QUALITY CONTROL REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total files processed: {qc_summary['total_files']}\n")
            f.write(f"Passed QC: {qc_summary['passed_qc']}\n")
            f.write(f"Total warnings: {qc_summary['warnings']}\n")
            f.write(f"Total errors: {qc_summary['errors']}\n\n")
            f.write("Atlases used:\n")
            for atlas, count in qc_summary.get('atlases_used', {}).items():
                f.write(f"  - {atlas}: {count} files\n")
            f.write("\n" + "=" * 60 + "\n")
        
        self.log(f"✓ QC report saved to: {qc_report_path}")
