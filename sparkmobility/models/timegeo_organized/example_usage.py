#!/usr/bin/env python3
"""
Example usage of the organized TimeGeo workflow package.

This script demonstrates how to use the production-ready TimeGeo workflow
with configurable parameters and proper error handling.
"""

import os
import sys
try:
    from config import TimeGeoConfig
    from workflow import TimeGeoWorkflow
except ImportError:
    # Fallback for package import
    from organized.config import TimeGeoConfig
    from organized.workflow import TimeGeoWorkflow


def main():
    """Example usage of the TimeGeo workflow."""
    
    # Example input path - replace with your actual data path
    input_parquet_path = "/data_1/albert/package_testing/test_results/StayPointsWithHomeWork/"
    
    # Create configuration with custom parameters
    config = TimeGeoConfig(
        input_parquet_path=input_parquet_path,
        output_base_dir="./results_organized",  # Custom output directory
        num_cpus=8,  # Use 8 CPUs instead of default 16
        num_stays_threshold=20,  # Higher threshold for frequent users
        min_num_stay=3,  # Custom minimum stays
        max_num_stay=2000,  # Custom maximum stays
        work_prob_weekday=0.85,  # Custom work probability
        work_prob_weekend=0.30,  # Custom weekend work probability
        num_days=2,  # Simulate 2 days instead of 1
        sample_fraction=0.03  # Sample 3% for other locations
    )
    
    # Create and run the workflow
    try:
        workflow = TimeGeoWorkflow(config)
        
        # Get workflow status before running
        status = workflow.get_workflow_status()
        print("Workflow status:")
        print(f"  Input path: {status['config']['input_path']}")
        print(f"  Output directory: {status['config']['output_dir']}")
        print(f"  CPUs: {status['config']['num_cpus']}")
        print(f"  Module available: {status['module_availability']['binary_exists']}")
        print(f"  Binary path: {status['module_availability'].get('binary_path', 'Not found')}")
        
        if status['data_summary'] and 'error' not in status['data_summary']:
            print(f"  Data summary: {status['data_summary']['total_records']} records, "
                  f"{status['data_summary']['unique_users']} users")
        
        # Run the complete workflow
        success = workflow.run()
        
        if success:
            print("\n✅ Workflow completed successfully!")
            return 0
        else:
            print("\n❌ Workflow failed!")
            return 1
            
    except Exception as e:
        print(f"\n💥 Error initializing or running workflow: {e}")
        import traceback
        traceback.print_exc()
        return 1


def example_with_default_config():
    """Example using default configuration."""
    
    # Simple example with minimal configuration
    config = TimeGeoConfig(
        input_parquet_path="/path/to/your/data.parquet"
    )
    
    workflow = TimeGeoWorkflow(config)
    return workflow.run()


def example_custom_analysis():
    """Example with custom analysis parameters."""
    
    config = TimeGeoConfig(
        input_parquet_path="/path/to/your/data.parquet",
        output_base_dir="./custom_analysis_results",
        num_cpus=32,  # High-performance setup
        num_stays_threshold=50,  # Only very frequent users
        min_num_stay=5,
        max_num_stay=5000,
        num_days=7,  # Simulate a full week
        work_prob_weekday=0.90,  # High work probability
        work_prob_weekend=0.20,  # Low weekend work probability
        sample_fraction=0.05  # More other locations
    )
    
    workflow = TimeGeoWorkflow(config)
    return workflow.run()


if __name__ == "__main__":
    # Run the main example
    exit_code = main()
    sys.exit(exit_code) 