#!/usr/bin/env python3
"""
Parquet-optimized TimeGeo Workflow
This script runs the complete timegeo workflow for processing parquet data directly.

Key improvements:
- Direct parquet processing throughout the pipeline
- No unnecessary text file conversions
- Consistent user ID handling
- Faster processing with pandas vectorized operations
"""

import sys
import os
import pandas as pd
import h3
from multiprocessing import Pool
import importlib
import shutil
import subprocess

# Configure parquet engine to handle PyArrow issues
def configure_parquet_engine():
    """Configure the best available parquet engine."""
    try:
        import pyarrow.parquet
        print("Using PyArrow parquet engine")
        return 'pyarrow'
    except (ImportError, ValueError) as e:
        print(f"PyArrow not available or has issues: {e}")
        try:
            import fastparquet
            print("Falling back to fastparquet engine")
            return 'fastparquet'
        except ImportError:
            print("Neither PyArrow nor fastparquet available. Installing fastparquet...")
            os.system("conda install fastparquet -y")
            return 'fastparquet'

# Set the parquet engine globally
PARQUET_ENGINE = configure_parquet_engine()

# Wrapper function for robust parquet reading
def read_parquet_robust(file_path):
    """Read parquet file with fallback engine support."""
    try:
        return pd.read_parquet(file_path, engine=PARQUET_ENGINE)
    except Exception as e:
        print(f"Error reading {file_path} with {PARQUET_ENGINE}: {e}")
        # Try alternative engine
        alt_engine = 'fastparquet' if PARQUET_ENGINE == 'pyarrow' else 'pyarrow'
        print(f"Trying alternative engine: {alt_engine}")
        return pd.read_parquet(file_path, engine=alt_engine)

# Wrapper function for robust parquet writing
def to_parquet_robust(df, file_path, **kwargs):
    """Write parquet file with fallback engine support."""
    try:
        return df.to_parquet(file_path, engine=PARQUET_ENGINE, **kwargs)
    except Exception as e:
        print(f"Error writing {file_path} with {PARQUET_ENGINE}: {e}")
        # Try alternative engine
        alt_engine = 'fastparquet' if PARQUET_ENGINE == 'pyarrow' else 'pyarrow'
        print(f"Trying alternative engine: {alt_engine}")
        return df.to_parquet(file_path, engine=alt_engine, **kwargs)

# Add current directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Set working directory to script location if not already there
if os.getcwd() != current_dir:
    print(f"Changing working directory from {os.getcwd()} to {current_dir}")
    os.chdir(current_dir)

# Import and reload parquet-optimized modules
import src_parquet.SRFiltered_to_SimInput
importlib.reload(src_parquet.SRFiltered_to_SimInput)
import src_parquet.Simulation_Preparation
importlib.reload(src_parquet.Simulation_Preparation)
import src_parquet.Simulation_Mapper
importlib.reload(src_parquet.Simulation_Mapper)
import src_parquet.Simulation_PostProcessing
importlib.reload(src_parquet.Simulation_PostProcessing)
import src_parquet.Aggregated_Plots
importlib.reload(src_parquet.Aggregated_Plots)

# Also import text-based functions for compatibility where needed
import src.SRFiltered_to_SimInput
importlib.reload(src.SRFiltered_to_SimInput)
import src.Simulation_Preparation
importlib.reload(src.Simulation_Preparation)
import src.Simulation_Mapper
importlib.reload(src.Simulation_Mapper)

print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
print(f"Using parquet engine: {PARQUET_ENGINE}")

# Import the C++ module for parameter generation
import module_2_3_1
print(f"Module function: {module_2_3_1.run_DT_simulation}")

# Import specific functions from parquet-optimized modules
from src_parquet.SRFiltered_to_SimInput import (
    decode_and_write_parameters, 
    remove_redundant_stays_parquet, 
    extract_frequent_users_parquet, 
    extract_stay_regions_for_frequent_users_parquet, 
    clean_and_format_fa_users_parquet
)
from src_parquet.Simulation_Preparation import (
    generate_simulation_input_parquet, 
    generate_simulation_parameters, 
    split_simulation_inputs, 
    activeness, 
    otherLocations
)
from src_parquet.Simulation_Mapper import simulate, simulate_all_parallel
from src_parquet.Simulation_PostProcessing import (
    compress_simulation_results,
    export_simulation_results_to_parquet,
    analyze_simulation_results_parquet,
    compress_and_export_simulation_results
)
from src_parquet.Aggregated_Plots import (
    plot_hourly_trip_counts, 
    plot_dept_validation,
    plot_stay_durations_parquet,
    analyze_mobility_patterns_parquet
)

# Import text-based functions for backward compatibility
from src.SRFiltered_to_SimInput import (
    remove_redundant_stays, 
    extract_frequent_users, 
    extract_stay_regions_for_frequent_users, 
    clean_and_format_fa_users
)
from src.Simulation_Preparation import (
    generate_simulation_input
)


def data_alignment(df_path):
    '''
    Convert input parquet to the required format for the pipeline.
    
    The data_alignment function expects your input Parquet table to include exactly these four columns:
    1. caid - a unique identifier for each user (any hashable type)
    2. stay_start_timestamp - the timestamp when the stay begins  
    3. type - a label for the kind of stay (e.g. "home", "work", "other")
    4. h3_id_region - an H3 index at resolution 16
    
    Returns:
        str: Path to the aligned parquet file
    '''
    # Load consolidated parquet file
    print(f"Processing data alignment for: {df_path}")
    df = read_parquet_robust(df_path)
    print(f"Loaded {len(df)} records from {df_path}")

    # Convert H3 integer to hex string, get lat/lng
    df['h3_id_region_16'] = df['h3_id_region'].astype(int).apply(lambda x: format(x, 'x'))
    lat_lng = list(map(h3.h3_to_geo, df['h3_id_region_16']))
    df['Latitude'], df['Longitude'] = zip(*lat_lng)
    
    # Convert timestamp to UNIX seconds
    df['timestamp'] = pd.to_datetime(df['stay_start_timestamp']).astype(int) // 10**9

    # Insert constant zero column for compatibility
    df['zero'] = 0

    # Keep original user IDs as hex strings for compatibility with parameter generation
    df['caid'] = df['caid'].astype(str)

    # Reorder for export
    out_cols = ['caid', 'timestamp', 'type', 'zero', 'h3_id_region', 'Longitude', 'Latitude']
    
    # Save as parquet for optimized processing
    out_path_parquet = './Data_CDR_to_SRFiltered/StayRegionsFiltered.parquet'
    to_parquet_robust(df[out_cols], out_path_parquet, index=False)
    
    print(f"Data alignment complete. Saved to {out_path_parquet}")
    return out_path_parquet


def ensure_dir(directory):
    '''Ensure that the given directory exists.'''
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")


def consolidate_partitioned_data(input_path, output_path):
    """
    Consolidate partitioned parquet data into a single file for C++ module processing.
    
    Args:
        input_path: Path to partitioned parquet directory or single file
        output_path: Path for consolidated output file
        
    Returns:
        str: Path to consolidated file
    """
    if os.path.isdir(input_path):
        print(f"Consolidating partitioned parquet data from: {input_path}")
        # Read all partitions into a single DataFrame
        df = read_parquet_robust(input_path)
        print(f"Loaded {len(df)} records from {len([f for f in os.listdir(input_path) if f.endswith('.parquet')])} partition files")
        
        # Save as single consolidated file
        to_parquet_robust(df, output_path, index=False)
        print(f"Consolidated data saved to: {output_path}")
        return output_path
    else:
        print(f"Input is already a single file: {input_path}")
        return input_path


def run_cpp_module_direct(input_path, output_dir, commuter_mode=False, min_num_stay=2, max_num_stay=3000, nw_thres=1.0, slot_interval=600, rho=0.6, gamma=-0.21):
    """
    Call the C++ module directly using subprocess instead of Python binding.
    This bypasses memory issues in the Python wrapper.
    
    Command line format: ./module_2_3_1 <input_path> <output_dir> <commuter_mode> <min_num_stay> <max_num_stay> <nw_thres> <slot_interval> <rho> <gamma>
    """
    print("Using direct C++ binary call to bypass Python binding memory issues...")
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Build command arguments using the correct format from the C++ source
    cmd = [
        './module_2_3_1',
        input_path,
        output_dir,
        '1' if commuter_mode else '0',  # Convert boolean to string
        str(min_num_stay),
        str(max_num_stay),
        str(nw_thres),
        str(slot_interval),
        str(rho),
        str(gamma)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the C++ binary directly
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("Direct C++ binary execution completed successfully")
            print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return True
        else:
            print(f"Direct C++ binary failed with return code: {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"Error calling direct C++ binary: {e}")
        return False


def main_workflow(input_parquet_path, num_cpus=16):
    """
    Main workflow function for the parquet-optimized timegeo pipeline.
    
    Args:
        input_parquet_path: Path to input parquet file or directory
        num_cpus: Number of CPUs to use for parallel processing
    """
    
    print("="*80)
    print("STARTING PARQUET-OPTIMIZED TIMEGEO WORKFLOW")
    print("="*80)
    
    # Validate input path
    if not os.path.exists(input_parquet_path):
        raise FileNotFoundError(f"Input path does not exist: {input_parquet_path}")
    
    print(f"Input path: {input_parquet_path}")
    print(f"Input is directory: {os.path.isdir(input_parquet_path)}")
    
    # Create necessary directories
    directories = [
        "./results/Parameters",
        "./results/Parameters/NonCommuters", 
        "./results/Parameters/Commuters",
        "./results/SRFiltered_to_SimInput",
        "./results/Simulation",
        "./results/Simulation/Locations",
        "./results/Simulation/Parameters", 
        "./results/Simulation/Mapped",
        "./results/Simulation/Compressed",
        "./results/Analysis",
        "./results/figs",
        "./Data_CDR_to_SRFiltered"
    ]
    
    for directory in directories:
        ensure_dir(directory)

    # Step 1: Data Alignment - Process the parquet data for the pipeline
    print("\n" + "="*60)
    print("STEP 1: DATA PROCESSING AND ALIGNMENT")
    print("="*60)
    
    aligned_parquet_path = data_alignment(input_parquet_path)

    # Step 2: Parameter Generation using module_2_3_1 on original data
    print("\n" + "="*60)
    print("STEP 2: PARAMETER GENERATION")
    print("="*60)
    
    # Use original parquet file for C++ module
    print(f"Running C++ parameter generation with original file: {input_parquet_path}")
    
    # CRITICAL: Run parameter generation for BOTH non-commuters AND commuters
    # This matches the notebook workflow and enables work locations in simulation
    
    print("2a: Generating parameters for NON-COMMUTERS...")
    success_noncomm = run_cpp_module_direct(
        input_path=input_parquet_path,
        output_dir="./results/Parameters",
        commuter_mode=False,  # Non-commuters first
        min_num_stay=2,
        max_num_stay=3000,
        nw_thres=1.0,
        slot_interval=600,
        rho=0.6,
        gamma=-0.21
    )
    
    if not success_noncomm:
        print("Direct binary call failed for non-commuters, trying Python binding...")
        try:
            import module_2_3_1
            result = module_2_3_1.run_DT_simulation(
                input_path=input_parquet_path,
                output_dir="./results/Parameters",
                commuter_mode=False,
                min_num_stay=2,
                max_num_stay=3000,
                nw_thres=1.0,
                slot_interval=600,
                rho=0.6,
                gamma=-0.21
            )
            print(f"C++ module (non-commuters) completed successfully")
        except Exception as e:
            print(f"C++ module (non-commuters) failed: {e}")
            return False
    
    print("2b: Generating parameters for COMMUTERS...")
    success_comm = run_cpp_module_direct(
        input_path=input_parquet_path,
        output_dir="./results/Parameters",
        commuter_mode=True,  # Commuters second - THIS WAS MISSING!
        min_num_stay=2,
        max_num_stay=3000,
        nw_thres=1.0,
        slot_interval=600,
        rho=0.6,
        gamma=-0.21
    )
    
    if not success_comm:
        print("Direct binary call failed for commuters, trying Python binding...")
        try:
            import module_2_3_1
            result = module_2_3_1.run_DT_simulation(
                input_path=input_parquet_path,
                output_dir="./results/Parameters",
                commuter_mode=True,
                min_num_stay=2,
                max_num_stay=3000,
                nw_thres=1.0,
                slot_interval=600,
                rho=0.6,
                gamma=-0.21
            )
            print(f"C++ module (commuters) completed successfully")
        except Exception as e:
            print(f"C++ module (commuters) failed: {e}")
            return False

    # Step 3: Decode Parameter Values (only if parameter files have content)
    print("\n" + "="*60)
    print("STEP 3: PARAMETER VALUE DECODING")
    print("="*60)
    
    # Check if parameter files have content before decoding
    commuter_param_file = './results/Parameters/Commuters/ParametersCommuters.txt'
    noncommuter_param_file = './results/Parameters/NonCommuters/ParametersNonCommuters.txt'
    
    # Get file sizes
    commuter_size = os.path.getsize(commuter_param_file) if os.path.exists(commuter_param_file) else 0
    noncommuter_size = os.path.getsize(noncommuter_param_file) if os.path.exists(noncommuter_param_file) else 0
    
    print(f"Parameter file sizes: Commuters={commuter_size}, NonCommuters={noncommuter_size}")
    
    # Verify we have both commuter and non-commuter parameters
    if commuter_size == 0:
        print("⚠️  Warning: No commuter parameters found - this will result in no work locations!")
    if noncommuter_size == 0:
        print("⚠️  Warning: No non-commuter parameters found!")
    
    if commuter_size > 0 or noncommuter_size > 0:
        b1_array = list(range(1, 21))
        b2_array = list(range(1, 21))
        
        decode_and_write_parameters(
            b1_array=b1_array,
            b2_array=b2_array,
            commuter_input_path=commuter_param_file,
            noncommuter_input_path=noncommuter_param_file,
            commuter_output_path='./results/Parameters/ParametersCommuters.txt',
            noncommuter_output_path='./results/Parameters/ParametersNonCommuters.txt'
        )
        print("Parameter decoding completed")
        
        # Verify decoded files
        final_comm_size = os.path.getsize('./results/Parameters/ParametersCommuters.txt') if os.path.exists('./results/Parameters/ParametersCommuters.txt') else 0
        final_noncomm_size = os.path.getsize('./results/Parameters/ParametersNonCommuters.txt') if os.path.exists('./results/Parameters/ParametersNonCommuters.txt') else 0
        print(f"Decoded parameter files: Commuters={final_comm_size}, NonCommuters={final_noncomm_size}")
        
    else:
        print("❌ ERROR: No parameter files found - workflow cannot continue!")
        return False

    # Step 4: Data Processing Pipeline using PARQUET files
    print("\n" + "="*60)
    print("STEP 4: DATA PROCESSING PIPELINE (PARQUET)")
    print("="*60)
    
    # 4a: Remove redundant stays using parquet
    print("4a: Removing redundant stays...")
    remove_redundant_stays_parquet(
        input_path=aligned_parquet_path,
        output_path='./results/SRFiltered_to_SimInput/FilteredStayRegions_set.parquet'
    )
    
    # 4b: Extract frequent users using parquet
    print("4b: Extracting frequent users...")
    extract_frequent_users_parquet(
        input_path='./results/SRFiltered_to_SimInput/FilteredStayRegions_set.parquet', 
        output_path='./results/SRFiltered_to_SimInput/FAUsers.parquet', 
        num_stays_threshold=15
    )
    
    # 4c: Extract stay regions for frequent users using parquet
    print("4c: Extracting stay regions for frequent users...")
    extract_stay_regions_for_frequent_users_parquet(
        fa_users_path='./results/SRFiltered_to_SimInput/FAUsers.parquet',
        input_path='./results/SRFiltered_to_SimInput/FilteredStayRegions_set.parquet',
        output_path='./results/SRFiltered_to_SimInput/FAUsers_StayRegions.parquet'
    )
    
    # 4d: Clean and format user data using parquet
    print("4d: Cleaning and formatting user data...")
    clean_and_format_fa_users_parquet(
        input_path='./results/SRFiltered_to_SimInput/FAUsers_StayRegions.parquet',
        output_path='./results/SRFiltered_to_SimInput/FAUsers_Cleaned_Formatted.parquet'
    )

    # Step 5: Simulation Preparation
    print("\n" + "="*60)
    print("STEP 5: SIMULATION PREPARATION")
    print("="*60)
    
    # 5a: Generating simulation input...
    print("5a: Generating simulation input...")
    generate_simulation_input_parquet(
        input_path='./results/SRFiltered_to_SimInput/FAUsers_Cleaned_Formatted.parquet',
        output_path='./results/Simulation/simulation_location.txt'
    )

    # 5b: Generate simulation parameters and ensure user ID consistency
    print("5b: Generating simulation parameters...")
    generate_simulation_parameters( 
        commuter_path='./results/Parameters/ParametersCommuters.txt',
        noncommuter_path='./results/Parameters/ParametersNonCommuters.txt',
        output_path='./results/Simulation/simulation_parameter.txt',
        work_prob_weekday=0.829,
        work_prob_weekend=0.354,
        num_days=1,
        reg_prob=0.846,
        gmm_group_index=0
    )
    
    # Use the original parameter file - no need for alignment as we'll fix the splitting
    aligned_param_file = './results/Simulation/simulation_parameter.txt'

    # 5c: Splitting simulation inputs...
    print("5c: Splitting simulation inputs...")
    split_simulation_inputs(
        parameter_path=aligned_param_file,  # Use aligned parameter file
        location_path='./results/Simulation/simulation_location.txt',
        formatted_user_path='./results/SRFiltered_to_SimInput/FAUsers_Cleaned_Formatted.parquet',
        output_dir='./results/Simulation',
        num_cpus=num_cpus
    )
    
    # 5d: Generate activeness patterns
    print("5d: Generating activeness patterns...")
    activeness(
        noncomm_daily_path='./results/Parameters/NonComm_pt_daily.txt', 
        noncomm_weekly_path='./results/Parameters/NonComm_pt_weekly.txt', 
        comm_daily_path='./results/Parameters/Comm_pt_daily.txt', 
        comm_weekly_path='./results/Parameters/Comm_pt_weekly.txt', 
        output_path='./results/Simulation/activeness.txt'
    )
    
    # 5e: Generate other locations
    print("5e: Generating other locations...")
    otherLocations(
        input_path='./results/Simulation/simulation_location.txt', 
        output_path='./results/Simulation/otherlocation.txt', 
        sample_fraction=0.02
    )

    # Step 6: Run Simulation
    print("\n" + "="*60)
    print("STEP 6: RUNNING SIMULATION")
    print("="*60)
    
    print(f"Running optimized simulation with {num_cpus} parallel processes...")
    
    # Use the original working simulation function
    simulate_all_parallel(
        num_cpus=num_cpus,
        other_locations_file='./results/Simulation/otherlocation.txt',
        activeness_file='./results/Simulation/activeness.txt',
        num_days=1,
        start_slot=0,
        users_locations_dir='./results/Simulation/Locations',
        users_parameters_dir='./results/Simulation/Parameters',
        output_dir='./results/Simulation/Mapped'
    )
    
    # Validate simulation results
    print("Validating simulation results...")
    import glob
    result_files = glob.glob('./results/Simulation/Mapped/simulationResults_*.txt')
    total_users_simulated = 0
    
    for result_file in result_files:
        if os.path.getsize(result_file) > 0:
            with open(result_file, 'r') as f:
                user_count = 0
                for line in f:
                    if len(line.strip().split()) == 1:  # User ID line
                        user_count += 1
                total_users_simulated += user_count
                print(f"  {os.path.basename(result_file)}: {user_count} users simulated")
        else:
            print(f"  {os.path.basename(result_file)}: EMPTY")
    
    print(f"Total users simulated: {total_users_simulated}")
    
    if total_users_simulated == 0:
        print("ERROR: No users were simulated! Check the user ID alignment and parameter files.")
        return False

    # Step 7: Post-processing and Analysis
    print("\n" + "="*60)
    print("STEP 7: POST-PROCESSING AND ANALYSIS")
    print("="*60)
    
    # 7a: Comprehensive post-processing
    print("7a: Running comprehensive post-processing...")
    simulation_df, analysis_results = compress_and_export_simulation_results(
        input_folder='./results/Simulation/Mapped/',
        compressed_folder='./results/Simulation/Compressed/',
        parquet_file='./results/Simulation/simulation_results.parquet',
        analysis_dir='./results/Analysis/'
    )

    # Step 8: Generate Plots and Visualizations
    print("\n" + "="*60)
    print("STEP 8: GENERATING PLOTS AND VISUALIZATIONS")
    print("="*60)
    
    # 8a: Hourly trip counts
    print("8a: Generating hourly trip count plots...")
    plot_hourly_trip_counts(
        mapped_dir='./results/Simulation/Mapped/',
        output='./results/figs/1-HourlyTripCount.png'
    )

    # 8b: Stay duration analysis
    print("8b: Generating stay duration plots...")
    plot_stay_durations_parquet(
        sim_parquet='./results/Simulation/simulation_results.parquet',
        cdr_parquet='./results/SRFiltered_to_SimInput/FAUsers_Cleaned_Formatted.parquet',
        output_file='./results/figs/2-StayDuration_All.png'
    )

    # 8c: Comprehensive mobility pattern analysis (including missing plots)
    print("8c: Generating comprehensive mobility analysis...")
    analyze_mobility_patterns_parquet(
        sim_parquet='./results/Simulation/simulation_results.parquet',
        cdr_parquet='./results/SRFiltered_to_SimInput/FAUsers_Cleaned_Formatted.parquet',
        output_dir='./results/figs/'
    )
    
    # Additional validation plots
    print("8d: Generating additional validation plots...")
    
    # Generate department validation if data is available
    try:
        plot_dept_validation(
            mapped_dir='./results/Simulation/Mapped/',
            output='./results/figs/6-dept_validation.png'
        )
        print("Department validation plot generated successfully")
    except Exception as e:
        print(f"Could not generate department validation plot: {e}")

    print("\n" + "="*80)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Results available in:")
    print(f"  - Parameters: ./results/Parameters/")
    print(f"  - Simulation results: ./results/Simulation/")
    print(f"  - Analysis: ./results/Analysis/")
    print(f"  - Plots: ./results/figs/")
    print(f"    * 1-HourlyTripCount.png: Hourly trip patterns")
    print(f"    * 2-StayDuration_All.png: Stay duration distributions")
    print(f"    * 3-TripDistance.png: Trip distance distributions")
    print(f"    * 4-numVisitedLocations.png: Daily visited location counts")
    print(f"    * 5-LocationRank-User1.png: Location visit frequency rankings")
    print(f"    * 6-dept_validation.png: Departure time validation")
    print("="*80)
    
    return True


if __name__ == "__main__":
    # Example usage
    
    # You can specify the input parquet file path here
    # Replace this with your actual input parquet file path
    input_parquet_path = "/data_1/aparimit/imelda_data/outputs_imelda/2019112900/2019112900_1/work_locations.parquet"
    
    # Run the complete workflow
    main_workflow(
        input_parquet_path=input_parquet_path,
        num_cpus=16    # Number of parallel processes
    ) 