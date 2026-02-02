#!/usr/bin/env python
"""
Setup script for bctpy_mrtrix project
Creates a virtual environment and installs all dependencies using UV
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, description=None):
    """Run a shell command and handle errors"""
    if description:
        print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False

def check_uv_installed():
    """Check if UV is installed"""
    try:
        subprocess.run("uv --version", shell=True, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def install_uv():
    """Install UV package manager"""
    print("UV not found. Installing UV...")
    system = platform.system()
    
    if system in ["Linux", "Darwin"]:  # Linux or macOS
        return run_command(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "Installing UV"
        )
    elif system == "Windows":
        return run_command(
            "powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\"",
            "Installing UV"
        )
    else:
        print(f"❌ Unsupported OS: {system}")
        return False

def create_venv(project_root):
    """Create virtual environment"""
    venv_path = project_root / ".venv"
    if venv_path.exists():
        print(f"✓ Virtual environment already exists at {venv_path}")
        return True
    
    return run_command(
        f"cd \"{project_root}\" && uv venv",
        "Creating virtual environment"
    )

def install_dependencies(project_root):
    """Install project dependencies"""
    # Ensure pip is up to date inside the UV-managed environment
    run_command(
        f"cd \"{project_root}\" && uv pip install --upgrade pip",
        "Upgrading pip in virtual environment"
    )

    dependencies = [
        "numpy",
        "pandas",
        "bctpy",
        "scipy",
        "statsmodels",
        "openpyxl",
        "flask",
        "waitress",
        "pyarrow",
        "h5py"
    ]
    
    cmd = f"cd \"{project_root}\" && uv pip install {' '.join(dependencies)}"
    return run_command(cmd, "Installing dependencies")

def main():
    """Main setup function"""
    print("=" * 50)
    print("Setting up bctpy_mrtrix environment")
    print("=" * 50)
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print(f"\nProject root: {project_root}")
    print(f"Python version: {sys.version}")
    
    # Check UV
    print("\nChecking for UV package manager...")
    if not check_uv_installed():
        if not install_uv():
            print("❌ Failed to install UV")
            sys.exit(1)
        print("✓ UV installed successfully")
    else:
        uv_version = subprocess.run(
            "uv --version",
            shell=True,
            capture_output=True,
            text=True
        ).stdout.strip()
        print(f"✓ UV is installed: {uv_version}")
    
    # Create venv
    print("\nSetting up virtual environment...")
    if not create_venv(project_root):
        print("❌ Failed to create virtual environment")
        sys.exit(1)
    print("✓ Virtual environment ready")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    if not install_dependencies(project_root):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    print("✓ Dependencies installed")
    
    # Success message
    print("\n" + "=" * 50)
    print("✅ Setup complete!")
    print("=" * 50)
    print("\nTo activate the environment, run:")
    if platform.system() == "Windows":
        print(f"  {project_root}\\.venv\\Scripts\\activate")
    else:
        print(f"  source {project_root}/.venv/bin/activate")
    print("\nOr use UV to run scripts directly:")
    print("  uv run python bct_test.py")
    print()

if __name__ == "__main__":
    main()
