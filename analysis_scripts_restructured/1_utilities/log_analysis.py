#!/usr/bin/env python3
"""
Analysis Logging Utility

Automatically log analysis execution details for reproducibility.
Creates timestamped entries in ANALYSIS_LOG.json with all parameters.

Usage:
    from log_analysis import AnalysisLogger
    
    logger = AnalysisLogger()
    logger.start_analysis(
        script_name="my_analysis.py",
        description="Brief description",
        parameters={"param1": value1, "param2": value2},
        inputs=["input_file1.parquet", "input_file2.csv"],
        outputs=["output_file1.parquet"]
    )
    
    # ... run analysis ...
    
    logger.finish_analysis(
        success=True,
        results_summary={"metric": value, "count": n}
    )
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
import socket
import getpass
import hashlib

class AnalysisLogger:
    """Log analysis execution for reproducibility"""
    
    def __init__(self, log_file="ANALYSIS_LOG.json"):
        """
        Initialize logger
        
        Parameters
        ----------
        log_file : str
            Path to JSON log file (relative to script directory)
        """
        self.script_dir = Path(__file__).parent.parent
        self.log_path = self.script_dir / log_file
        self.current_entry = None
        
    def _load_log(self):
        """Load existing log or create new one"""
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                return json.load(f)
        else:
            return {"analyses": [], "created": datetime.now().isoformat()}
    
    def _save_log(self, log_data):
        """Save log to disk"""
        with open(self.log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def _get_file_hash(self, filepath):
        """Compute SHA256 hash of file for version tracking"""
        try:
            hasher = hashlib.sha256()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()[:16]  # Short hash
        except Exception:
            return None
    
    def _get_git_commit(self):
        """Get current git commit hash if available"""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                cwd=self.script_dir,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def start_analysis(self, script_name, description, parameters=None, 
                      inputs=None, outputs=None, notes=None):
        """
        Start logging an analysis
        
        Parameters
        ----------
        script_name : str
            Name of analysis script (e.g., "node_comprehensive_analysis.py")
        description : str
            Brief description of what the analysis does
        parameters : dict, optional
            Analysis parameters as key-value pairs
        inputs : list of str, optional
            List of input file paths
        outputs : list of str, optional
            List of expected output file paths
        notes : str, optional
            Additional notes or context
        """
        # Get script hash for versioning
        script_path = self.script_dir / "analysis_scripts" / script_name
        script_hash = self._get_file_hash(script_path) if script_path.exists() else None
        
        self.current_entry = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "script_name": script_name,
            "description": description,
            "parameters": parameters or {},
            "inputs": inputs or [],
            "outputs": outputs or [],
            "notes": notes,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "success": None,
            "results_summary": {},
            "environment": {
                "hostname": socket.gethostname(),
                "user": getpass.getuser(),
                "python_version": sys.version.split()[0],
                "working_directory": str(Path.cwd()),
                "script_hash": script_hash,
                "git_commit": self._get_git_commit()
            }
        }
        
        print(f"[AnalysisLogger] Started: {script_name}")
        print(f"[AnalysisLogger] ID: {self.current_entry['id']}")
        print(f"[AnalysisLogger] Time: {self.current_entry['start_time']}")
        
        return self.current_entry['id']
    
    def finish_analysis(self, success=True, results_summary=None, error_message=None):
        """
        Finish logging current analysis
        
        Parameters
        ----------
        success : bool
            Whether analysis completed successfully
        results_summary : dict, optional
            Summary statistics or key results
        error_message : str, optional
            Error message if analysis failed
        """
        if self.current_entry is None:
            raise RuntimeError("No analysis started. Call start_analysis() first.")
        
        end_time = datetime.now()
        start_time = datetime.fromisoformat(self.current_entry['start_time'])
        duration = (end_time - start_time).total_seconds()
        
        self.current_entry['end_time'] = end_time.isoformat()
        self.current_entry['duration_seconds'] = duration
        self.current_entry['success'] = success
        self.current_entry['results_summary'] = results_summary or {}
        
        if error_message:
            self.current_entry['error_message'] = error_message
        
        # Verify output files exist
        output_status = {}
        for output_file in self.current_entry['outputs']:
            exists = Path(output_file).exists()
            output_status[output_file] = {
                "exists": exists,
                "size_bytes": Path(output_file).stat().st_size if exists else None
            }
        self.current_entry['output_verification'] = output_status
        
        # Save to log
        log_data = self._load_log()
        log_data['analyses'].append(self.current_entry)
        self._save_log(log_data)
        
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"[AnalysisLogger] {status}: {self.current_entry['script_name']}")
        print(f"[AnalysisLogger] Duration: {duration:.1f} seconds")
        print(f"[AnalysisLogger] Log updated: {self.log_path}")
        
        self.current_entry = None
    
    def add_note(self, note):
        """Add a note to current analysis"""
        if self.current_entry:
            if self.current_entry['notes']:
                self.current_entry['notes'] += f"\n{note}"
            else:
                self.current_entry['notes'] = note
    
    def get_last_analysis(self, script_name=None):
        """
        Get the most recent analysis entry
        
        Parameters
        ----------
        script_name : str, optional
            Filter by script name
        
        Returns
        -------
        dict or None
            Most recent analysis entry matching criteria
        """
        log_data = self._load_log()
        
        if not log_data['analyses']:
            return None
        
        if script_name:
            matching = [a for a in log_data['analyses'] if a['script_name'] == script_name]
            return matching[-1] if matching else None
        else:
            return log_data['analyses'][-1]
    
    def print_summary(self, n=10):
        """Print summary of recent analyses"""
        log_data = self._load_log()
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS LOG SUMMARY (most recent {n})")
        print(f"{'='*80}\n")
        
        for entry in log_data['analyses'][-n:]:
            status = "✓" if entry['success'] else "✗"
            duration = entry['duration_seconds']
            duration_str = f"{duration:.1f}s" if duration else "running"
            
            print(f"{status} {entry['id']} | {entry['script_name']}")
            print(f"   {entry['description']}")
            print(f"   Duration: {duration_str} | {entry['start_time']}")
            
            if entry['results_summary']:
                print(f"   Results: {entry['results_summary']}")
            
            print()


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="View analysis log")
    parser.add_argument('--summary', action='store_true', 
                       help='Print summary of recent analyses')
    parser.add_argument('--n', type=int, default=10,
                       help='Number of recent analyses to show')
    parser.add_argument('--script', type=str,
                       help='Filter by script name')
    parser.add_argument('--json', action='store_true',
                       help='Output full log as JSON')
    
    args = parser.parse_args()
    
    logger = AnalysisLogger()
    
    if args.json:
        with open(logger.log_path, 'r') as f:
            print(f.read())
    elif args.summary:
        logger.print_summary(n=args.n)
    elif args.script:
        entry = logger.get_last_analysis(script_name=args.script)
        if entry:
            print(json.dumps(entry, indent=2))
        else:
            print(f"No entries found for script: {args.script}")
    else:
        parser.print_help()
