"""
C++ module utilities for parameter generation.
"""

import os
import subprocess
import sys
from typing import Optional, Dict, Any


class CppModuleHandler:
    """
    Handles C++ module operations for parameter generation.
    
    This class provides methods for calling the C++ parameter generation
    module both through Python bindings and direct binary execution.
    """
    
    def __init__(self, module_path: str = None):
        """
        Initialize the C++ module handler.
        
        Args:
            module_path: Path to the C++ binary (auto-detected if None)
        """
        if module_path is None:
            # Try to find the binary in common locations
            possible_paths = [
                "./module_2_3_1",  # Current directory
                "../module_2_3_1",  # Parent directory
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "module_2_3_1"),  # Two levels up
                "/usr/local/bin/module_2_3_1",  # System path
            ]
            
            for path in possible_paths:
                if os.path.exists(path) and os.access(path, os.X_OK):
                    module_path = path
                    print(f"Found C++ binary at: {module_path}")
                    break
            else:
                # Default to current directory if not found
                module_path = "./module_2_3_1"
                print(f"Warning: C++ binary not found in common locations, using: {module_path}")
        
        self.module_path = module_path
        self.python_module = None
        self._load_python_module()
    
    def _load_python_module(self) -> None:
        """Load the Python binding for the C++ module."""
        try:
            import module_2_3_1
            self.python_module = module_2_3_1
            print("Python binding for C++ module loaded successfully")
        except ImportError as e:
            print(f"Python binding not available: {e}")
            self.python_module = None
    
    def run_parameter_generation(
        self,
        input_path: str,
        output_dir: str,
        commuter_mode: bool = False,
        min_num_stay: int = 2,
        max_num_stay: int = 3000,
        nw_thres: float = 1.0,
        slot_interval: int = 600,
        rho: float = 0.6,
        gamma: float = -0.21,
        use_direct_binary: bool = True
    ) -> bool:
        """
        Run parameter generation using the C++ module.
        
        Args:
            input_path: Path to input parquet file
            output_dir: Output directory for parameters
            commuter_mode: Whether to run in commuter mode
            min_num_stay: Minimum number of stays
            max_num_stay: Maximum number of stays
            nw_thres: Network threshold
            slot_interval: Time slot interval
            rho: Rho parameter
            gamma: Gamma parameter
            use_direct_binary: Whether to use direct binary call
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        if use_direct_binary:
            return self._run_direct_binary(
                input_path, output_dir, commuter_mode, min_num_stay,
                max_num_stay, nw_thres, slot_interval, rho, gamma
            )
        else:
            return self._run_python_binding(
                input_path, output_dir, commuter_mode, min_num_stay,
                max_num_stay, nw_thres, slot_interval, rho, gamma
            )
    
    def _run_direct_binary(
        self,
        input_path: str,
        output_dir: str,
        commuter_mode: bool,
        min_num_stay: int,
        max_num_stay: int,
        nw_thres: float,
        slot_interval: int,
        rho: float,
        gamma: float
    ) -> bool:
        """
        Call the C++ module directly using subprocess.
        
        Args:
            input_path: Path to input file
            output_dir: Output directory
            commuter_mode: Whether to run in commuter mode
            min_num_stay: Minimum number of stays
            max_num_stay: Maximum number of stays
            nw_thres: Network threshold
            slot_interval: Time slot interval
            rho: Rho parameter
            gamma: Gamma parameter
            
        Returns:
            bool: True if successful, False otherwise
        """
        print("Using direct C++ binary call to bypass Python binding memory issues...")
        
        if not os.path.exists(self.module_path):
            print(f"C++ binary not found at {self.module_path}")
            return False
        
        # Convert module path to absolute path to avoid issues with working directory changes
        if not os.path.isabs(self.module_path):
            abs_module_path = os.path.abspath(self.module_path)
        else:
            abs_module_path = self.module_path
        
        # Convert input path to absolute path as well
        if not os.path.isabs(input_path):
            abs_input_path = os.path.abspath(input_path)
        else:
            abs_input_path = input_path
        
        # Build command arguments
        cmd = [
            abs_module_path,
            abs_input_path,
            output_dir,
            "1" if commuter_mode else "0",  # Convert boolean to string
            str(min_num_stay),
            str(max_num_stay),
            str(nw_thres),
            str(slot_interval),
            str(rho),
            str(gamma),
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            # Run the C++ binary directly without changing working directory
            # Use absolute paths to avoid directory structure issues
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                print("Direct C++ binary execution completed successfully")
                if result.stdout:
                    print("STDOUT:", result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
                
                # Move any parameter files that might have been created
                self._move_parameter_files_to_output_dir(output_dir)
                
                return True
            else:
                print(f"Direct C++ binary failed with return code: {result.returncode}")
                if result.stdout:
                    print("STDOUT:", result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("C++ binary execution timed out after 1 hour")
            return False
        except Exception as e:
            print(f"Error calling direct C++ binary: {e}")
            return False
    
    def _run_python_binding(
        self,
        input_path: str,
        output_dir: str,
        commuter_mode: bool,
        min_num_stay: int,
        max_num_stay: int,
        nw_thres: float,
        slot_interval: int,
        rho: float,
        gamma: float
    ) -> bool:
        """
        Run parameter generation using Python binding.
        
        Args:
            input_path: Path to input file
            output_dir: Output directory
            commuter_mode: Whether to run in commuter mode
            min_num_stay: Minimum number of stays
            max_num_stay: Maximum number of stays
            nw_thres: Network threshold
            slot_interval: Time slot interval
            rho: Rho parameter
            gamma: Gamma parameter
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.python_module is None:
            print("Python binding not available")
            return False
        
        print("Using Python binding for C++ module...")
        
        try:
            result = self.python_module.run_DT_simulation(
                input_path=input_path,
                output_dir=output_dir,
                commuter_mode=commuter_mode,
                min_num_stay=min_num_stay,
                max_num_stay=max_num_stay,
                nw_thres=nw_thres,
                slot_interval=slot_interval,
                rho=rho,
                gamma=gamma,
            )
            print(f"C++ module completed successfully")
            return True
        except Exception as e:
            print(f"C++ module failed: {e}")
            return False
    
    def check_module_availability(self) -> Dict[str, Any]:
        """
        Check the availability of the C++ module.
        
        Returns:
            dict: Information about module availability
        """
        result = {
            "binary_exists": os.path.exists(self.module_path),
            "binary_path": self.module_path,
            "python_binding_available": self.python_module is not None,
            "binary_executable": False,
            "binary_size": 0
        }
        
        if result["binary_exists"]:
            result["binary_size"] = os.path.getsize(self.module_path)
            result["binary_executable"] = os.access(self.module_path, os.X_OK)
        
        return result
    
    def get_module_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the C++ module.
        
        Returns:
            dict: Detailed module information
        """
        availability = self.check_module_availability()
        
        info = {
            "module_path": self.module_path,
            "availability": availability,
            "python_version": sys.version,
            "platform": sys.platform
        }
        
        if availability["binary_exists"]:
            try:
                # Try to get version info from binary
                result = subprocess.run(
                    [self.module_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    info["version_info"] = result.stdout.strip()
            except:
                pass
        
        return info
    
    def _move_parameter_files_to_output_dir(self, output_dir: str) -> None:
        """
        Move parameter files that might have been created in the current directory
        to the specified output directory.
        
        Args:
            output_dir: Directory where parameter files should be moved
        """
        import shutil
        
        # List of parameter files that might be created
        parameter_files = [
            "Comm_pt_daily.txt",
            "Comm_pt_daily_weekly.txt", 
            "Comm_pt_weekly.txt",
            "NonComm_pt_daily.txt",
            "NonComm_pt_daily_weekly.txt",
            "NonComm_pt_weekly.txt"
        ]
        
        moved_count = 0
        for filename in parameter_files:
            # Check in current directory (where the binary runs from)
            if os.path.exists(filename):
                source_path = filename
                dest_path = os.path.join(output_dir, filename)
                
                try:
                    # Ensure output directory exists
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Move the file
                    shutil.move(source_path, dest_path)
                    print(f"Moved {filename} to {output_dir}")
                    moved_count += 1
                except Exception as e:
                    print(f"Warning: Could not move {filename}: {e}")
        
        if moved_count > 0:
            print(f"Successfully moved {moved_count} parameter files to {output_dir}")
        else:
            print("No parameter files found to move")
    
    def cleanup_parameter_files_from_current_dir(self) -> None:
        """
        Clean up any parameter files that might be left in the current directory.
        This is a safety measure to prevent files from accumulating.
        """
        import shutil
        
        # List of parameter files that might be created
        parameter_files = [
            "Comm_pt_daily.txt",
            "Comm_pt_daily_weekly.txt", 
            "Comm_pt_weekly.txt",
            "NonComm_pt_daily.txt",
            "NonComm_pt_daily_weekly.txt",
            "NonComm_pt_weekly.txt"
        ]
        
        cleaned_count = 0
        for filename in parameter_files:
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    print(f"Cleaned up {filename} from current directory")
                    cleaned_count += 1
                except Exception as e:
                    print(f"Warning: Could not remove {filename}: {e}")
        
        if cleaned_count > 0:
            print(f"Cleaned up {cleaned_count} parameter files from current directory") 