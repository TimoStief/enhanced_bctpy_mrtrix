# Quick Start Guide - BCT Analysis Web Interface

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
Run the setup script once:

```bash
# Option 1: Bash (macOS/Linux)
bash scripts/setup_env.sh

# Option 2: Python (All platforms)
python scripts/setup_env.py
```

This will:
- Install UV package manager (if needed)
- Create a virtual environment
- Install all required packages

### 2. Launch Web Interface

```bash
# Option 1: PowerShell (Windows)
.\scripts\run_web_app.ps1

# Option 2: Bash (macOS/Linux)
bash scripts/run_web_app.sh

# Option 3: Direct Python
cd web_app && python app.py
```

The browser will automatically open at `http://127.0.0.1:5000`

### 3. Run Analysis

1. **Enter Input Folder Path**: Type or paste the full path to your data folder, then click "Validate":
   ```
   data/
   ├── ses-1/    (containing .npy files)
   ├── ses-2/    (containing .npy files)
   ├── ses-3/    (containing .npy files)
   └── ses-4/    (containing .npy files)
   ```
  Example: `/Users/karl/work/github/bctpy_mrtrix/Test_matrizen`

2. **Confirm Sessions**: The app will automatically detect and validate sessions

3. **Start Analysis**: Click "Start Analysis" and watch the terminal

4. **Download Results**: When complete, download the Excel results file

---

## What the Web Interface Does

### Analysis Pipeline
The unified analyzer combines functionality from all three original scripts:

```
Input: .npy connectivity matrices
  ↓
[BCT Analyzer]
  ├─ Detects matrix type (Binary/Weighted, Directed/Undirected)
  ├─ Calculates metrics:
  │  ├─ Degree & Strength
  │  ├─ Density & Efficiency
  │  ├─ Clustering Coefficient
  │  ├─ Transitivity
  │  ├─ Community Detection
  │  └─ And more...
  ├─ Aggregates by session
  └─ Organizes by subject
  ↓
Output: bct_analysis_results.xlsx
```

### Real-Time Features
- 📁 **Directory Browser**: Navigate folders easily
- ✓ **Session Validation**: Auto-detect session structure
- 📊 **Live Terminal**: Real-time analysis output
- 📈 **Results Summary**: Matrices processed, sessions, subjects
- 💾 **Auto-Download**: Export results as Excel

---

## File Structure

```
bctpy_mrtrix/
├── web_app/                    # ← Web interface
│   ├── app.py                  # Flask application
│   ├── bct_analyzer.py         # Analysis engine
│   ├── README.md               # Detailed documentation
│   ├── templates/
│   │   ├── base.html
│   │   └── index.html
│   └── static/
│       └── css/main.css
├── scripts/
│   ├── setup_env.py            # Setup script (Python)
│   ├── setup_env.sh            # Setup script (Bash)
│   └── run_web_app.sh          # Web app launcher
├── Test_matrizen/              # Sample test data
├── pyproject.toml              # Dependencies
└── README.md                   # Main documentation
```

---

## Common Use Cases

### Case 1: First Time Setup
```bash
# Install everything
python scripts/setup_env.py

# Run the app
bash scripts/run_web_app.sh
```

### Case 2: Run After First Setup
```bash
# Just launch the app (venv already exists)
bash scripts/run_web_app.sh
```

### Case 3: Using in Different Directory
```bash
# Copy data to Test_matrizen/ses-1, ses-2, etc.
# Or in the web UI, use the file browser to navigate
```

### Case 4: Command Line Alternative
```bash
# Activate venv
source .venv/bin/activate

# Run analysis directly
python bct_test.py        # Basic analysis
python bct_all_test.py    # Full analysis
```

---

## Troubleshooting

**Q: "Virtual environment not found"**
```bash
python scripts/setup_env.py
```

**Q: "Port already in use"**
The app automatically finds a free port. If it still fails, restart your terminal.

**Q: "Permission denied" on data folder**
```bash
chmod -R +r /path/to/your/data
```

**Q: How do I find my folder path?**
On macOS: Drag the folder into the Terminal or use `pwd` in the folder's directory
On Windows: Right-click folder → "Copy as path"
On Linux: Use `pwd` command in the folder's directory

**Q: Large analysis taking too long?**
This is normal! 100+ matrices can take 5-10 minutes depending on your hardware.

---

## What's Different from Original Scripts?

| Feature | Original | Web Interface |
|---------|----------|---------------|
| Cross-platform paths | ❌ Hardcoded Windows | ✅ Dynamic paths |
| Directory selection | ❌ Manual coding | ✅ Visual browser |
| Real-time feedback | ❌ Console only | ✅ Web terminal |
| Session validation | ❌ Manual | ✅ Auto-detect |
| Results export | ⚠️ Auto save | ✅ Download button |
| GUI | ❌ No | ✅ Modern web UI |
| Setup automation | ⚠️ Manual | ✅ One command |

---

## Next Steps

1. **Set up dependencies** with `python scripts/setup_env.py`
2. **Launch the web interface** with `bash scripts/run_web_app.sh`
3. **Browse to** `http://127.0.0.1:5000` (opens automatically)
4. **Select your data folder** and start analyzing!

For detailed information, see [web_app/README.md](web_app/README.md)

---

## Branch Information

You're on the `enhanced-handling` branch which includes:
- ✅ Fixed cross-platform paths
- ✅ UV virtual environment setup
- ✅ Web interface with Flask
- ✅ Unified analysis engine
- ✅ Live terminal output
- ✅ Results export

Ready to merge to main when happy!
