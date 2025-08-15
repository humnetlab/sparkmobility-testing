"""
Configuration class for TimeGeo workflow parameters.
"""

from dataclasses import dataclass
from typing import Optional, List
import os


@dataclass
class TimeGeoConfig:
    """
    This class provides default values for all parameters and can be customized
    by users before running the workflow.
    """
    
    # Input/Output 
    input_parquet_path: str
    output_base_dir: str = "./results"
    
    # Processing parameters
    num_cpus: int = 16
    num_stays_threshold: int = 15  # For frequent user extraction
    
    # C++ Module parameters
    min_num_stay: int = 2
    max_num_stay: int = 3000
    nw_thres: float = 1.0
    slot_interval: int = 600
    rho: float = 0.6
    gamma: float = -0.21
    
    # Simulation parameters
    work_prob_weekday: float = 0.829
    work_prob_weekend: float = 0.354
    num_days: int = 1
    reg_prob: float = 0.846
    gmm_group_index: int = 0
    
    # Parameter decoding arrays
    b1_array: Optional[List[int]] = None
    b2_array: Optional[List[int]] = None
    
    # Other locations sampling
    sample_fraction: float = 0.02
    
    # Parquet engine configuration
    parquet_engine: Optional[str] = None  # Will be auto-detected if None
    
    def __post_init__(self):
        """Set default arrays if not provided."""
        if self.b1_array is None:
            self.b1_array = list(range(1, 21))
        if self.b2_array is None:
            self.b2_array = list(range(1, 21))
    
    def get_output_directories(self) -> dict:
        """Get all required output directory paths."""
        return {
            "parameters": os.path.join(self.output_base_dir, "Parameters"),
            "parameters_commuters": os.path.join(self.output_base_dir, "Parameters", "Commuters"),
            "parameters_noncommuters": os.path.join(self.output_base_dir, "Parameters", "NonCommuters"),
            "srfiltered": os.path.join(self.output_base_dir, "SRFiltered_to_SimInput"),
            "simulation": os.path.join(self.output_base_dir, "Simulation"),
            "simulation_locations": os.path.join(self.output_base_dir, "Simulation", "Locations"),
            "simulation_parameters": os.path.join(self.output_base_dir, "Simulation", "Parameters"),
            "simulation_mapped": os.path.join(self.output_base_dir, "Simulation", "Mapped"),
            "simulation_compressed": os.path.join(self.output_base_dir, "Simulation", "Compressed"),
            "analysis": os.path.join(self.output_base_dir, "Analysis"),
            "figs": os.path.join(self.output_base_dir, "figs"),
            "data_cdr": os.path.join(self.output_base_dir, "Data_CDR_to_SRFiltered"),
        }
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if not os.path.exists(self.input_parquet_path):
            raise FileNotFoundError(f"Input path does not exist: {self.input_parquet_path}")
        
        if self.num_cpus < 1:
            raise ValueError("num_cpus must be at least 1")
        
        if self.num_stays_threshold < 1:
            raise ValueError("num_stays_threshold must be at least 1")
        
        if self.min_num_stay < 1:
            raise ValueError("min_num_stay must be at least 1")
        
        if self.max_num_stay <= self.min_num_stay:
            raise ValueError("max_num_stay must be greater than min_num_stay")
        
        if self.nw_thres <= 0:
            raise ValueError("nw_thres must be positive")
        
        if self.slot_interval <= 0:
            raise ValueError("slot_interval must be positive")
        
        if not (0 < self.rho < 1):
            raise ValueError("rho must be between 0 and 1")
        
        if self.work_prob_weekday < 0 or self.work_prob_weekday > 1:
            raise ValueError("work_prob_weekday must be between 0 and 1")
        
        if self.work_prob_weekend < 0 or self.work_prob_weekend > 1:
            raise ValueError("work_prob_weekend must be between 0 and 1")
        
        if self.num_days < 1:
            raise ValueError("num_days must be at least 1")
        
        if self.reg_prob < 0 or self.reg_prob > 1:
            raise ValueError("reg_prob must be between 0 and 1")
        
        if self.sample_fraction <= 0 or self.sample_fraction > 1:
            raise ValueError("sample_fraction must be between 0 and 1")
        
        return True 