"""
Data processing class for TimeGeo workflow.
"""

import os
import h3
import pandas as pd
from typing import Set
try:
    from .utils.parquet_utils import ParquetHandler
    from .utils.file_utils import FileHandler
except ImportError:
    # Fallback for direct execution
    from utils.parquet_utils import ParquetHandler
    from utils.file_utils import FileHandler


class DataProcessor:
    """
    Handles data processing and alignment for the TimeGeo workflow.
    
    This class provides methods for converting input parquet data to the
    required format for the TimeGeo pipeline.
    """
    
    def __init__(self, parquet_handler: ParquetHandler):
        """
        Initialize the data processor.
        
        Args:
            parquet_handler: ParquetHandler instance for file operations
        """
        self.parquet_handler = parquet_handler
        self.file_handler = FileHandler()
    
    def align_data(self, input_path: str, output_path: str) -> str:
        """
        Convert input parquet to the required format for the pipeline.
        
        The data_alignment function expects the input Parquet table to include 
        exactly these four columns:
        1. caid - a unique identifier for each user (any hashable type)
        2. stay_start_timestamp - the timestamp when the stay begins
        3. type - a label for the kind of stay (e.g. "home", "work", "other")
        4. h3_id_region - an H3 index at resolution 16
        
        Args:
            input_path: Path to input parquet file
            output_path: Path for aligned output file
            
        Returns:
            str: Path to the aligned parquet file
        """
        print(f"Processing data alignment for: {input_path}")
        
        # Load consolidated parquet file
        df = self.parquet_handler.read_parquet(input_path)
        print(f"Loaded {len(df)} records from {input_path}")
        
        # Validate required columns
        required_columns = ["caid", "stay_start_timestamp", "type", "h3_id_region"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert H3 integer to hex string, get lat/lng
        df["h3_id_region_16"] = (
            df["h3_id_region"].astype(int).apply(lambda x: format(x, "x"))
        )
        lat_lng = list(map(h3.h3_to_geo, df["h3_id_region_16"]))
        df["Latitude"], df["Longitude"] = zip(*lat_lng)
        
        # Convert timestamp to UNIX seconds
        df["timestamp"] = pd.to_datetime(df["stay_start_timestamp"]).astype(int) // 10**9
        
        # Insert constant zero column for compatibility
        df["zero"] = 0
        
        # Keep original user IDs as hex strings for compatibility with parameter generation
        df["caid"] = df["caid"].astype(str)
        
        # Reorder for export
        out_cols = [
            "caid",
            "timestamp",
            "type",
            "zero",
            "h3_id_region",
            "Longitude",
            "Latitude",
        ]
        
        # Save as parquet for optimized processing
        self.parquet_handler.write_parquet(df[out_cols], output_path, index=False)
        
        print(f"Data alignment complete. Saved to {output_path}")
        return output_path
    
    def extract_frequent_users(
        self, 
        input_path: str, 
        output_path: str, 
        num_stays_threshold: int = 15
    ) -> Set[str]:
        """
        Extract frequent users from the aligned data.
        
        Args:
            input_path: Path to aligned parquet file
            output_path: Path for frequent users output file
            num_stays_threshold: Minimum number of stays to be considered frequent
            
        Returns:
            Set[str]: Set of frequent user IDs
        """
        print(f"Extracting frequent users with threshold {num_stays_threshold}...")
        
        # Load aligned data
        df = self.parquet_handler.read_parquet(input_path)
        
        # Count stays per user
        user_stay_counts = df.groupby('caid').size()
        frequent_users = user_stay_counts[user_stay_counts >= num_stays_threshold].index
        
        # Filter data to frequent users only
        frequent_users_df = df[df['caid'].isin(frequent_users)]
        
        # Save frequent users data
        self.parquet_handler.write_parquet(frequent_users_df, output_path, index=False)
        
        print(f"Found {len(frequent_users)} frequent users out of {len(user_stay_counts)} total users")
        return set(frequent_users)
    
    def remove_redundant_stays(self, input_path: str, output_path: str) -> str:
        """
        Remove redundant stays from the data.
        
        Args:
            input_path: Path to input parquet file
            output_path: Path for filtered output file
            
        Returns:
            str: Path to filtered file
        """
        print("Removing redundant stays...")
        
        # Load data
        df = self.parquet_handler.read_parquet(input_path)
        initial_count = len(df)
        
        # Remove duplicates based on user, location, and timestamp
        df_filtered = df.drop_duplicates(
            subset=['caid', 'h3_id_region', 'timestamp'], 
            keep='first'
        )
        
        # Save filtered data
        self.parquet_handler.write_parquet(df_filtered, output_path, index=False)
        
        removed_count = initial_count - len(df_filtered)
        print(f"Removed {removed_count} redundant stays ({len(df_filtered)} remaining)")
        
        return output_path
    
    def get_data_summary(self, file_path: str) -> dict:
        """
        Get a summary of the data in a parquet file.
        
        Args:
            file_path: Path to parquet file
            
        Returns:
            dict: Summary statistics
        """
        if not os.path.exists(file_path):
            return {"error": "File not found"}
        
        try:
            df = self.parquet_handler.read_parquet(file_path)
            
            summary = {
                "total_records": len(df),
                "unique_users": df['caid'].nunique() if 'caid' in df.columns else 0,
                "unique_locations": df['h3_id_region'].nunique() if 'h3_id_region' in df.columns else 0,
                "columns": list(df.columns),
                "file_size": self.file_handler.get_file_size(file_path),
                "memory_usage": df.memory_usage(deep=True).sum()
            }
            
            # Add type distribution if available
            if 'type' in df.columns:
                summary["type_distribution"] = df['type'].value_counts().to_dict()
            
            # Add time range if timestamp is available
            if 'timestamp' in df.columns:
                summary["time_range"] = {
                    "start": pd.to_datetime(df['timestamp'], unit='s').min(),
                    "end": pd.to_datetime(df['timestamp'], unit='s').max()
                }
            
            return summary
            
        except Exception as e:
            return {"error": str(e)}
    
    def validate_input_data(self, file_path: str) -> dict:
        """
        Validate input data format and content.
        
        Args:
            file_path: Path to input parquet file
            
        Returns:
            dict: Validation results
        """
        validation = {
            "valid": False,
            "errors": [],
            "warnings": []
        }
        
        try:
            df = self.parquet_handler.read_parquet(file_path)
            
            # Check required columns
            required_columns = ["caid", "stay_start_timestamp", "type", "h3_id_region"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                validation["errors"].append(f"Missing required columns: {missing_columns}")
            else:
                validation["valid"] = True
            
            # Check data quality
            if len(df) == 0:
                validation["errors"].append("File contains no records")
                validation["valid"] = False
            
            if df['caid'].isna().any():
                validation["warnings"].append("Found null values in user IDs")
            
            if df['h3_id_region'].isna().any():
                validation["warnings"].append("Found null values in location IDs")
            
            if df['stay_start_timestamp'].isna().any():
                validation["warnings"].append("Found null values in timestamps")
            
            # Check for reasonable data ranges
            if 'caid' in df.columns:
                unique_users = df['caid'].nunique()
                if unique_users < 10:
                    validation["warnings"].append(f"Very few unique users: {unique_users}")
            
            if 'h3_id_region' in df.columns:
                unique_locations = df['h3_id_region'].nunique()
                if unique_locations < 5:
                    validation["warnings"].append(f"Very few unique locations: {unique_locations}")
            
            return validation
            
        except Exception as e:
            validation["errors"].append(f"Failed to read file: {str(e)}")
            return validation 