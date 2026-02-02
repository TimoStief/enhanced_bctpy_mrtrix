# BCT Analysis Web Interface

A modern web-based interface for running Brain Connectivity Toolbox (BCT) analysis on connectivity matrices.

## Features

- **Interactive Web Interface**: Browser-based GUI for easy analysis setup
- **Directory Browser**: Select input and output folders with visual folder navigation
- **Session Detection**: Automatically detect and validate session structure (ses-1, ses-2, ses-3, ses-4)
- **Live Terminal**: Real-time analysis output in a terminal-like interface
- **Comprehensive Analysis**: Combined functionality from all three original scripts
  - Matrix type detection (BU, WU, BD, WD)
  - Calculate degree, strength, density, clustering
  - Community detection and modularity
  - And many more BCT metrics
- **Results Export**: Download analysis results as Excel files
- **Responsive Design**: Works on desktop and mobile browsers

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Project dependencies installed (see installation section)

### Installation

1. Set up the Python environment:
```bash
bash scripts/setup_env.sh
# or
python scripts/setup_env.py
```

2. Activate the virtual environment (optional, if not using `uv run`):
```bash
source .venv/bin/activate
```

### Running the Web Interface

**Option 1: Using the launcher script**
```bash
bash scripts/run_web_app.sh
```

**Option 2: Direct Python**
```bash
cd web_app
python app.py
```

**Option 3: Using UV**
```bash
cd web_app
uv run python app.py
```

The application will:
- Start a local web server (typically at `http://127.0.0.1:5000`)
- Automatically open your default web browser
- Display the analysis interface

## Usage

### Step 1: Select Input Directory
1. **Enter the folder path**: Type or paste the full path to your data folder in the "Input Directory" field
2. Click the **Validate** button to verify the path is valid
3. The interface will automatically detect if the directory has the expected session structure (ses-1, ses-2, ses-3, ses-4)
4. Confirmed sessions will be displayed in green

**Example paths:**
- macOS: `/Users/username/data/connectivity_data`
- Linux: `/home/username/data/connectivity_data`
- Windows: `C:\Users\username\data\connectivity_data`

### Step 2: Configure Analysis
- **Output Directory** (optional): Where results will be saved
  - If left empty, results will be saved in `input_dir/bct_output`
- **Analysis Type**: Choose between full analysis or matrix-only analysis

### Step 3: Run Analysis
1. Click **Start Analysis**
2. Watch the terminal output for real-time progress
3. Analysis results will appear automatically when complete

### Step 4: Download Results
- Once analysis is complete, results are available in the "Analysis Results" section
- Click **Download Results** to get an Excel file with all metrics

## API Endpoints

The application provides the following REST endpoints:

- `GET /` - Main web interface
- `POST /api/analyze` - Start analysis with specified directories
- `GET /api/logs` - Get current analysis logs
- `POST /api/get-directory` - List directories for browsing
- `POST /api/check-sessions` - Validate session structure in a directory
- `GET /api/download/<filename>` - Download result file
- `POST /shutdown` - Shutdown the application

## Input Directory Structure

Expected structure for input directories:

```
data_folder/
├── ses-1/
│   ├── subject_001_matrix.npy
│   ├── subject_002_matrix.npy
│   └── ...
├── ses-2/
│   ├── subject_001_matrix.npy
│   ├── subject_002_matrix.npy
│   └── ...
├── ses-3/
│   └── ...
└── ses-4/
    └── ...
```

## Output Files

Analysis produces the following output:

- **bct_analysis_results.xlsx** - Complete analysis results in Excel format
  - Subject and session information
  - Matrix type and shape
  - All calculated metrics (degree, strength, density, clustering, etc.)

## Technology Stack

- **Backend**: Flask 3.1.2, Waitress 3.0.2
- **Frontend**: Bootstrap 5.1.3, Font Awesome 6.0.0
- **Data Processing**: NumPy, Pandas, SciPy
- **Analysis**: Brain Connectivity Toolbox (bctpy)
- **Excel**: openpyxl

## Architecture

### Components

1. **Flask App** (`app.py`)
   - REST API endpoints
   - File serving
   - Server management

2. **BCT Analyzer** (`bct_analyzer.py`)
   - Unified analysis engine
   - Matrix processing
   - Metric calculation
   - Result aggregation

3. **Web Interface** (HTML/CSS/JavaScript)
   - Directory browser
   - Real-time terminal output
   - Results display
   - File download

## Troubleshooting

### Virtual Environment Not Found
```bash
bash scripts/setup_env.sh
```

### Port Already in Use
The application tries to find a free port automatically. If issues persist, edit `web_app/app.py` and change the port number.

### Permission Denied
Make sure you have read permissions on the input directory:
```bash
chmod -R +r /path/to/data
```

### Analysis Takes Too Long
This is normal for large datasets. Monitor progress in the terminal output.

## Project Structure

```
web_app/
├── app.py                      # Flask application
├── bct_analyzer.py            # Analysis engine
├── templates/
│   ├── base.html              # Base template
│   └── index.html             # Main interface
└── static/
    └── css/
        └── main.css           # Styling
```

## Environment Variables

Currently, the application uses no environment variables. Configuration is done through the web interface.

## Performance Notes

- Large datasets (100+ matrices) may take several minutes to process
- Memory usage scales with matrix size (typically 50-200 MB for 100 matrices)
- Recommended minimum: 4GB RAM, 2GB free disk space for output

## Contributing

To modify the analysis engine, edit `bct_analyzer.py`. Changes to the web interface can be made in `templates/` and `static/`.

## License

Check the main project repository for license information.

## Support

For issues with:
- **BCT metrics**: See [bctpy documentation](https://github.com/aestrivex/bctpy)
- **Web interface**: Check Flask and Waitress documentation
- **Data format**: Ensure .npy files are valid NumPy arrays

## Authors

Built with BCT (Brain Connectivity Toolbox) for neuroscience research.
