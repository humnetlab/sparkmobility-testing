import os
import sys
import importlib
from typing import Optional, Dict, Any
try:
    from .config import TimeGeoConfig
    from .utils.parquet_utils import ParquetHandler
    from .utils.file_utils import FileHandler
    from .utils.cpp_utils import CppModuleHandler
    from .data_processor import DataProcessor
except ImportError:
    # Fallback for direct execution
    from config import TimeGeoConfig
    from utils.parquet_utils import ParquetHandler
    from utils.file_utils import FileHandler
    from utils.cpp_utils import CppModuleHandler
    from data_processor import DataProcessor


class TimeGeoWorkflow:
    """
    Main TimeGeo workflow class that orchestrates the complete pipeline.
    
    This class provides a production-ready interface for running the TimeGeo
    mobility simulation workflow with configurable parameters and robust error handling.
    """
    
    def __init__(self, config: TimeGeoConfig):
        """
        Initialize the TimeGeo workflow.
        
        Args:
            config: TimeGeoConfig instance with all parameters
        """
        self.config = config
        self.config.validate()
        
        # Initialize utility classes
        self.parquet_handler = ParquetHandler(config.parquet_engine)
        self.file_handler = FileHandler()
        self.cpp_handler = CppModuleHandler()
        self.data_processor = DataProcessor(self.parquet_handler)
        
        # Setup output directories
        self.output_dirs = config.get_output_directories()
        self._setup_directories()
        
        # Import required modules
        self._import_modules()
        
        print(f"TimeGeo workflow initialized with {config.num_cpus} CPUs")
        print(f"Output directory: {config.output_base_dir}")
    
    def _setup_directories(self) -> None:
        """Create all required output directories."""
        directories = list(self.output_dirs.values())
        self.file_handler.ensure_directories(directories)
        print(f"Created {len(directories)} output directories")
    
    def _import_modules(self) -> None:
        """Import and reload required modules."""
        
        # Import parquet-optimized modules
        modules_to_import = [
            "utils.SRFiltered_to_SimInput",
            "utils.Simulation_Preparation", 
            "utils.Simulation_Mapper",
            "utils.Simulation_PostProcessing",
            "utils.Aggregated_Plots"
        ]
        
        for module_name in modules_to_import:
            try:
                module = importlib.import_module(module_name)
                importlib.reload(module)
                print(f"Imported and reloaded: {module_name}")
            except ImportError as e:
                print(f"Warning: Could not import {module_name}: {e}")
        
        # Import specific functions
        try:
            from utils.Aggregated_Plots import (
                analyze_mobility_patterns_parquet,
                plot_dept_validation,
                plot_hourly_trip_counts,
                plot_stay_durations_parquet
            )
            from utils.Simulation_Mapper import simulate_all_parallel
            from utils.Simulation_PostProcessing import (
                compress_and_export_simulation_results
            )
            from utils.Simulation_Preparation import (
                activeness, generate_simulation_input_parquet,
                generate_simulation_parameters, otherLocations, split_simulation_inputs
            )
            from utils.SRFiltered_to_SimInput import (
                clean_and_format_fa_users_parquet, decode_and_write_parameters,
                extract_frequent_users_parquet,
                extract_stay_regions_for_frequent_users_parquet,
                remove_redundant_stays_parquet
            )
            
            # Store imported functions
            self._imported_functions = {
                'analyze_mobility_patterns_parquet': analyze_mobility_patterns_parquet,
                'plot_dept_validation': plot_dept_validation,
                'plot_hourly_trip_counts': plot_hourly_trip_counts,
                'plot_stay_durations_parquet': plot_stay_durations_parquet,
                'simulate_all_parallel': simulate_all_parallel,
                'compress_and_export_simulation_results': compress_and_export_simulation_results,
                'activeness': activeness,
                'generate_simulation_input_parquet': generate_simulation_input_parquet,
                'generate_simulation_parameters': generate_simulation_parameters,
                'otherLocations': otherLocations,
                'split_simulation_inputs': split_simulation_inputs,
                'clean_and_format_fa_users_parquet': clean_and_format_fa_users_parquet,
                'decode_and_write_parameters': decode_and_write_parameters,
                'extract_frequent_users_parquet': extract_frequent_users_parquet,
                'extract_stay_regions_for_frequent_users_parquet': extract_stay_regions_for_frequent_users_parquet,
                'remove_redundant_stays_parquet': remove_redundant_stays_parquet
            }
            
            print("Successfully imported all required functions")
            
        except ImportError as e:
            print(f"Error importing required functions: {e}")
            raise
    
    def run(self) -> bool:
        """
        Run the complete TimeGeo workflow.
        
        Returns:
            bool: True if workflow completed successfully, False otherwise
        """
        try:
            print("=" * 80)
            print("STARTING TIMEGEO WORKFLOW")
            print("=" * 80)
            
            # Step 1: Data Processing and Alignment
            if not self._step1_data_processing():
                return False
            
            # Step 2: Parameter Generation
            if not self._step2_parameter_generation():
                return False
            
            # Step 3: Parameter Value Decoding
            if not self._step3_parameter_decoding():
                return False
            
            # Step 4: Simulation Preparation
            if not self._step4_simulation_preparation():
                return False
            
            # Step 5: Run Simulation
            if not self._step5_run_simulation():
                return False
            
            # Step 6: Post-processing and Analysis
            if not self._step6_post_processing():
                return False
            
            # Step 7: Generate Plots and Visualizations
            if not self._step7_generate_plots():
                return False
            
            print("\n" + "=" * 80)
            print("WORKFLOW COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            self._print_results_summary()
            
            return True
            
        except Exception as e:
            print(f"\nERROR: Workflow failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _step1_data_processing(self) -> bool:
        """Step 1: Data Processing and Alignment"""
        print("\n" + "=" * 60)
        print("STEP 1: DATA PROCESSING AND ALIGNMENT")
        print("=" * 60)
        
        try:
            # Validate input data
            validation = self.data_processor.validate_input_data(self.config.input_parquet_path)
            if not validation["valid"]:
                print("Input data validation failed:")
                for error in validation["errors"]:
                    print(f"  ERROR: {error}")
                return False
            
            if validation["warnings"]:
                print("Input data validation warnings:")
                for warning in validation["warnings"]:
                    print(f"  WARNING: {warning}")
            
            # Align data
            aligned_path = os.path.join(self.output_dirs["data_cdr"], "StayRegionsFiltered.parquet")
            self.data_processor.align_data(self.config.input_parquet_path, aligned_path)
            
            # Remove redundant stays
            filtered_path = os.path.join(self.output_dirs["srfiltered"], "FilteredStayRegions_set.parquet")
            self._imported_functions['remove_redundant_stays_parquet'](
                input_path=aligned_path,
                output_path=filtered_path
            )
            
            # Extract frequent users
            fa_users_path = os.path.join(self.output_dirs["srfiltered"], "FAUsers.parquet")
            self._imported_functions['extract_frequent_users_parquet'](
                input_path=filtered_path,
                output_path=fa_users_path,
                num_stays_threshold=self.config.num_stays_threshold
            )
            
            # Extract stay regions for frequent users
            fa_stay_regions_path = os.path.join(self.output_dirs["srfiltered"], "FAUsers_StayRegions.parquet")
            self._imported_functions['extract_stay_regions_for_frequent_users_parquet'](
                fa_users_path=fa_users_path,
                input_path=filtered_path,
                output_path=fa_stay_regions_path
            )
            
            # Clean and format user data
            cleaned_path = os.path.join(self.output_dirs["srfiltered"], "FAUsers_Cleaned_Formatted.parquet")
            self._imported_functions['clean_and_format_fa_users_parquet'](
                input_path=fa_stay_regions_path,
                output_path=cleaned_path
            )
            
            print("Step 1 completed successfully")
            return True
            
        except Exception as e:
            print(f"Step 1 failed: {e}")
            return False
    
    def _step2_parameter_generation(self) -> bool:
        """Step 2: Parameter Generation"""
        print("\n" + "=" * 60)
        print("STEP 2: PARAMETER GENERATION")
        print("=" * 60)
        
        try:
            # Get frequent user IDs and create filtered dataset
            cleaned_path = os.path.join(self.output_dirs["srfiltered"], "FAUsers_Cleaned_Formatted.parquet")
            import pandas as pd
            fa_users_df = self.parquet_handler.read_parquet(cleaned_path)
            frequent_user_ids = set(fa_users_df['caid'].unique())
            print(f"Found {len(frequent_user_ids)} frequent users for parameter generation")
            
            # Create filtered dataset for parameter generation
            original_df = self.parquet_handler.read_parquet(self.config.input_parquet_path)
            filtered_df = original_df[original_df['caid'].isin(frequent_user_ids)]
            filtered_path = os.path.join(self.output_dirs["srfiltered"], "FilteredForParameters.parquet")
            self.parquet_handler.write_parquet(filtered_df, filtered_path, index=False)
            print(f"Created filtered dataset with {len(filtered_df)} records for {len(frequent_user_ids)} users")
            
            # Generate parameters for non-commuters
            print("2a: Generating parameters for NON-COMMUTERS...")
            success_noncomm = self.cpp_handler.run_parameter_generation(
                input_path=filtered_path,
                output_dir=self.output_dirs["parameters"],
                commuter_mode=False,
                min_num_stay=self.config.min_num_stay,
                max_num_stay=self.config.max_num_stay,
                nw_thres=self.config.nw_thres,
                slot_interval=self.config.slot_interval,
                rho=self.config.rho,
                gamma=self.config.gamma
            )
            
            if not success_noncomm:
                print("Failed to generate non-commuter parameters")
                return False
            
            # Generate parameters for commuters
            print("2b: Generating parameters for COMMUTERS...")
            success_comm = self.cpp_handler.run_parameter_generation(
                input_path=filtered_path,
                output_dir=self.output_dirs["parameters"],
                commuter_mode=True,
                min_num_stay=self.config.min_num_stay,
                max_num_stay=self.config.max_num_stay,
                nw_thres=self.config.nw_thres,
                slot_interval=self.config.slot_interval,
                rho=self.config.rho,
                gamma=self.config.gamma
            )
            
            if not success_comm:
                print("Failed to generate commuter parameters")
                return False
            
            print("Step 2 completed successfully")
            
            # Clean up any parameter files that might be left in the current directory
            self.cpp_handler.cleanup_parameter_files_from_current_dir()
            
            return True
            
        except Exception as e:
            print(f"Step 2 failed: {e}")
            return False
    
    def _step3_parameter_decoding(self) -> bool:
        """Step 3: Parameter Value Decoding"""
        print("\n" + "=" * 60)
        print("STEP 3: PARAMETER VALUE DECODING")
        print("=" * 60)
        
        try:
            # Check parameter file sizes - files should be in the main parameters directory
            commuter_param_file = os.path.join(self.output_dirs["parameters"], "Commuters", "ParametersCommuters.txt")
            noncommuter_param_file = os.path.join(self.output_dirs["parameters"], "NonCommuters", "ParametersNonCommuters.txt")
            
            commuter_size = self.file_handler.get_file_size(commuter_param_file)
            noncommuter_size = self.file_handler.get_file_size(noncommuter_param_file)
            
            print(f"Parameter file sizes: Commuters={commuter_size}, NonCommuters={noncommuter_size}")
            
            if commuter_size == 0:
                print("⚠️  Warning: No commuter parameters found - this will result in no work locations!")
            if noncommuter_size == 0:
                print("⚠️  Warning: No non-commuter parameters found!")
            
            if commuter_size > 0 or noncommuter_size > 0:
                self._imported_functions['decode_and_write_parameters'](
                    b1_array=self.config.b1_array,
                    b2_array=self.config.b2_array,
                    commuter_input_path=commuter_param_file,
                    noncommuter_input_path=noncommuter_param_file,
                    commuter_output_path=os.path.join(self.output_dirs["parameters"], "ParametersCommuters.txt"),
                    noncommuter_output_path=os.path.join(self.output_dirs["parameters"], "ParametersNonCommuters.txt")
                )
                print("Parameter decoding completed")
            else:
                print("❌ ERROR: No parameter files found - workflow cannot continue!")
                return False
            
            print("Step 3 completed successfully")
            return True
            
        except Exception as e:
            print(f"Step 3 failed: {e}")
            return False
    
    def _step4_simulation_preparation(self) -> bool:
        """Step 4: Simulation Preparation"""
        print("\n" + "=" * 60)
        print("STEP 4: SIMULATION PREPARATION")
        print("=" * 60)
        
        try:
            # Generate simulation input
            cleaned_path = os.path.join(self.output_dirs["srfiltered"], "FAUsers_Cleaned_Formatted.parquet")
            simulation_location_path = os.path.join(self.output_dirs["simulation"], "simulation_location.txt")
            
            self._imported_functions['generate_simulation_input_parquet'](
                input_path=cleaned_path,
                output_path=simulation_location_path
            )
            
            # Generate simulation parameters
            commuter_param_path = os.path.join(self.output_dirs["parameters"], "ParametersCommuters.txt")
            noncommuter_param_path = os.path.join(self.output_dirs["parameters"], "ParametersNonCommuters.txt")
            simulation_param_path = os.path.join(self.output_dirs["simulation"], "simulation_parameter.txt")
            
            self._imported_functions['generate_simulation_parameters'](
                commuter_path=commuter_param_path,
                noncommuter_path=noncommuter_param_path,
                output_path=simulation_param_path,
                work_prob_weekday=self.config.work_prob_weekday,
                work_prob_weekend=self.config.work_prob_weekend,
                num_days=self.config.num_days,
                reg_prob=self.config.reg_prob,
                gmm_group_index=self.config.gmm_group_index
            )
            
            # Split simulation inputs
            self._imported_functions['split_simulation_inputs'](
                parameter_path=simulation_param_path,
                location_path=simulation_location_path,
                formatted_user_path=cleaned_path,
                output_dir=self.output_dirs["simulation"],
                num_cpus=self.config.num_cpus
            )
            
            # Generate activeness patterns - files should be in the main parameters directory
            activeness_path = os.path.join(self.output_dirs["simulation"], "activeness.txt")
            self._imported_functions['activeness'](
                noncomm_daily_path=os.path.join(self.output_dirs["parameters"], "NonComm_pt_daily.txt"),
                noncomm_weekly_path=os.path.join(self.output_dirs["parameters"], "NonComm_pt_weekly.txt"),
                comm_daily_path=os.path.join(self.output_dirs["parameters"], "Comm_pt_daily.txt"),
                comm_weekly_path=os.path.join(self.output_dirs["parameters"], "Comm_pt_weekly.txt"),
                output_path=activeness_path
            )
            
            # Generate other locations
            other_locations_path = os.path.join(self.output_dirs["simulation"], "otherlocation.txt")
            self._imported_functions['otherLocations'](
                input_path=simulation_location_path,
                output_path=other_locations_path,
                sample_fraction=self.config.sample_fraction
            )
            
            print("Step 4 completed successfully")
            return True
            
        except Exception as e:
            print(f"Step 4 failed: {e}")
            return False
    
    def _step5_run_simulation(self) -> bool:
        """Step 5: Run Simulation"""
        print("\n" + "=" * 60)
        print("STEP 5: RUNNING SIMULATION")
        print("=" * 60)
        
        try:
            print(f"Running optimized simulation with {self.config.num_cpus} parallel processes...")
            
            self._imported_functions['simulate_all_parallel'](
                num_cpus=self.config.num_cpus,
                other_locations_file=os.path.join(self.output_dirs["simulation"], "otherlocation.txt"),
                activeness_file=os.path.join(self.output_dirs["simulation"], "activeness.txt"),
                num_days=self.config.num_days,
                start_slot=0,
                users_locations_dir=self.output_dirs["simulation_locations"],
                users_parameters_dir=self.output_dirs["simulation_parameters"],
                output_dir=self.output_dirs["simulation_mapped"]
            )
            
            # Validate simulation results
            print("Validating simulation results...")
            result_files = self.file_handler.list_files_with_pattern(
                self.output_dirs["simulation_mapped"], "simulationResults_*.txt"
            )
            
            total_users_simulated = 0
            for result_file in result_files:
                if self.file_handler.file_exists_and_not_empty(result_file):
                    user_count = self.file_handler.count_lines_in_file(result_file)
                    total_users_simulated += user_count
                    print(f"  {os.path.basename(result_file)}: {user_count} users simulated")
                else:
                    print(f"  {os.path.basename(result_file)}: EMPTY")
            
            print(f"Total users simulated: {total_users_simulated}")
            
            if total_users_simulated == 0:
                print("ERROR: No users were simulated! Check the user ID alignment and parameter files.")
                return False
            
            print("Step 5 completed successfully")
            return True
            
        except Exception as e:
            print(f"Step 5 failed: {e}")
            return False
    
    def _step6_post_processing(self) -> bool:
        """Step 6: Post-processing and Analysis"""
        print("\n" + "=" * 60)
        print("STEP 6: POST-PROCESSING AND ANALYSIS")
        print("=" * 60)
        
        try:
            # Comprehensive post-processing
            simulation_df, analysis_results = self._imported_functions['compress_and_export_simulation_results'](
                input_folder=self.output_dirs["simulation_mapped"] + "/",
                compressed_folder=self.output_dirs["simulation_compressed"] + "/",
                parquet_file=os.path.join(self.output_dirs["simulation"], "simulation_results.parquet"),
                analysis_dir=self.output_dirs["analysis"] + "/"
            )
            
            print("Step 6 completed successfully")
            return True
            
        except Exception as e:
            print(f"Step 6 failed: {e}")
            return False
    
    def _step7_generate_plots(self) -> bool:
        """Step 7: Generate Plots and Visualizations"""
        print("\n" + "=" * 60)
        print("STEP 7: GENERATING PLOTS AND VISUALIZATIONS")
        print("=" * 60)
        
        try:
            # Hourly trip counts
            self._imported_functions['plot_hourly_trip_counts'](
                mapped_dir=self.output_dirs["simulation_mapped"] + "/",
                output=os.path.join(self.output_dirs["figs"], "1-HourlyTripCount.png")
            )
            
            # Stay duration analysis
            self._imported_functions['plot_stay_durations_parquet'](
                sim_parquet=os.path.join(self.output_dirs["simulation"], "simulation_results.parquet"),
                cdr_parquet=os.path.join(self.output_dirs["srfiltered"], "FAUsers_Cleaned_Formatted.parquet"),
                output_file=os.path.join(self.output_dirs["figs"], "2-StayDuration_All.png")
            )
            
            # Comprehensive mobility pattern analysis
            self._imported_functions['analyze_mobility_patterns_parquet'](
                sim_parquet=os.path.join(self.output_dirs["simulation"], "simulation_results.parquet"),
                cdr_parquet=os.path.join(self.output_dirs["srfiltered"], "FAUsers_Cleaned_Formatted.parquet"),
                output_dir=self.output_dirs["figs"] + "/"
            )
            
            # Additional validation plots
            try:
                self._imported_functions['plot_dept_validation'](
                    mapped_dir=self.output_dirs["simulation_mapped"] + "/",
                    output=os.path.join(self.output_dirs["figs"], "6-dept_validation.png")
                )
                print("Department validation plot generated successfully")
            except Exception as e:
                print(f"Could not generate department validation plot: {e}")
            
            print("Step 7 completed successfully")
            return True
            
        except Exception as e:
            print(f"Step 7 failed: {e}")
            return False
    
    def _print_results_summary(self) -> None:
        """Print a summary of the results."""
        print(f"Results available in:")
        print(f"  - Parameters: {self.output_dirs['parameters']}")
        print(f"  - Simulation results: {self.output_dirs['simulation']}")
        print(f"  - Analysis: {self.output_dirs['analysis']}")
        print(f"  - Plots: {self.output_dirs['figs']}")
        print(f"    * 1-HourlyTripCount.png: Hourly trip patterns")
        print(f"    * 2-StayDuration_All.png: Stay duration distributions")
        print(f"    * 3-TripDistance.png: Trip distance distributions")
        print(f"    * 4-numVisitedLocations.png: Daily visited location counts")
        print(f"    * 5-LocationRank-User1.png: Location visit frequency rankings")
        print(f"    * 6-dept_validation.png: Departure time validation")
        print("=" * 80)
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get the current status of the workflow.
        
        Returns:
            dict: Status information about the workflow
        """
        status = {
            "config": {
                "input_path": self.config.input_parquet_path,
                "output_dir": self.config.output_base_dir,
                "num_cpus": self.config.num_cpus
            },
            "directories": self.output_dirs,
            "module_availability": self.cpp_handler.check_module_availability(),
            "data_summary": None
        }
        
        # Get data summary if input file exists
        if os.path.exists(self.config.input_parquet_path):
            status["data_summary"] = self.data_processor.get_data_summary(self.config.input_parquet_path)
        
        return status 