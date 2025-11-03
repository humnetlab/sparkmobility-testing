# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**sparkmobility** is a human mobility modeling package that implements geospatial mobility models including TimeGeo (a time-geographic simulation framework) and EPR (Exploration and Preferential Return). The package processes mobility data using PySpark and integrates with C++ modules for computationally intensive parameter generation.

## Environment Setup

### Create Conda Environment
```bash
conda env create -f environment.yml
conda activate timegeo
```

### Install Package Dependencies
```bash
pip install -e .
```

The package automatically downloads and configures:
- Apache Spark (version 3.5.7) to `~/.spark/` on first import
- sparkmobility JAR file to `sparkmobility/lib/` from GCS

## Development Commands

### Code Formatting
```bash
./autoformat.sh
```
This runs:
- `autoflake` for removing unused imports
- `black` for code formatting
- `isort` for import sorting

### Testing
```bash
# Test the TimeGeo organized package
python sparkmobility/models/timegeo_organized/test_package.py

# Run model tests in Jupyter
jupyter notebook model_test.ipynb
```

## Architecture

### Package Structure

```
sparkmobility/
├── __init__.py          # Auto-downloads Spark & JAR, sets up environment
├── settings.py          # Config management and Spark installation
├── config/              # Configuration loading
├── models/
│   ├── epr.py          # EPR mobility model (pure Python)
│   ├── gravity.py      # Gravity model
│   └── timegeo_organized/  # TimeGeo simulation framework
│       ├── config.py        # TimeGeoConfig dataclass with all parameters
│       ├── workflow.py      # Main TimeGeoWorkflow orchestration
│       ├── data_processor.py # Input validation and processing
│       ├── module_2_3_1     # C++ binary for parameter generation
│       └── utils/
│           ├── cpp_utils.py              # C++ module interface
│           ├── parquet_utils.py          # Parquet I/O with fallback support
│           ├── file_utils.py             # File/directory operations
│           ├── SRFiltered_to_SimInput.py # Data preparation
│           ├── Simulation_Preparation.py # Simulation input generation
│           ├── Simulation_Mapper.py      # Parallel simulation execution
│           ├── Simulation_PostProcessing.py # Result compression
│           └── Aggregated_Plots.py       # Visualization and analysis
├── processing/          # Data processing utilities
├── utils/              # General utilities
└── visualization/      # Plotting functions
```

### TimeGeo Workflow

The TimeGeo model is the core simulation framework with a multi-stage pipeline:

1. **Data Preparation**: Extract frequent users from Parquet input with required columns (`caid`, `stay_start_timestamp`, `type`, `h3_id_region`)
2. **Parameter Generation**: C++ module (`module_2_3_1`) computes mobility parameters for commuters and non-commuters
3. **Simulation Input**: Generate location lists, time distributions, and simulation parameters
4. **Simulation Execution**: Parallel simulation using generated parameters
5. **Post-Processing**: Map results and compress output
6. **Analysis**: Generate validation plots and mobility pattern analysis

**Key Design**: The workflow uses a modular class-based design with `TimeGeoConfig` for configuration and `TimeGeoWorkflow` for orchestration. All intermediate outputs are stored in structured directories under a base output path.

### EPR Model

The EPR (Exploration and Preferential Return) model is implemented as a standalone class at `sparkmobility/models/epr.py`:
- Follows scikit-learn-like API: `fit()` learns OD matrix from data, `generate()` produces synthetic trajectories
- Works with H3 hexagon tessellations for spatial discretization
- Parameters: `rho` (exploration vs return), `gamma` (exploration decay), `beta` (waiting time exponent), `tau` (time scale)

### PySpark Integration

The package manages its own Spark installation:
- On import, `sparkmobility/__init__.py` triggers automatic Spark download and setup via `settings.py`
- Environment variables (`SPARK_HOME`, `PATH`) are configured programmatically
- Custom JAR (`sparkmobility010.jar`) provides additional Spark functionality

### C++ Module Integration

TimeGeo uses a compiled C++ binary (`module_2_3_1`) for computationally intensive parameter generation:
- Located at `sparkmobility/models/timegeo_organized/module_2_3_1`
- Wrapped by `CppModuleHandler` in `cpp_utils.py`
- Runs as subprocess with timeout handling
- Generates parameters for both commuters and non-commuters

## Working with TimeGeo

### Basic Usage Pattern
```python
from sparkmobility.models.timegeo_organized.config import TimeGeoConfig
from sparkmobility.models.timegeo_organized.workflow import TimeGeoWorkflow

# Configure with required input path
config = TimeGeoConfig(
    input_parquet_path="/path/to/data.parquet",
    num_cpus=16,
    num_days=7
)

# Run workflow
workflow = TimeGeoWorkflow(config)
success = workflow.run()
```

### Input Data Requirements
Parquet file must have exactly four columns:
- `caid`: User identifier
- `stay_start_timestamp`: Stay begin timestamp
- `type`: Stay type ("home", "work", "other")
- `h3_id_region`: H3 index at resolution 16

### Output Directory Structure
```
results/
├── Parameters/              # C++ generated parameters
│   ├── Commuters/
│   └── NonCommuters/
├── SRFiltered_to_SimInput/  # Processed data
├── Simulation/              # Simulation outputs
│   ├── Locations/
│   ├── Parameters/
│   ├── Mapped/
│   └── Compressed/
├── Analysis/                # Analysis results
└── figs/                    # Validation plots
```

## Important Implementation Details

### Parquet Engine Handling
The `ParquetHandler` class auto-detects and falls back between PyArrow and FastParquet for compatibility across environments.

### Module Imports and Reloading
`TimeGeoWorkflow._import_modules()` dynamically imports and reloads utility modules from the `utils/` directory to ensure latest code is used.

### Error Handling
All workflow components include comprehensive validation:
- Input data validation in `DataProcessor`
- Parameter validation in `TimeGeoConfig.validate()`
- Graceful error recovery with detailed messages

### Parallel Processing
The workflow uses Python multiprocessing with configurable `num_cpus` parameter for parallel execution in simulation stages.
