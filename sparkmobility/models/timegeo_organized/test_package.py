#!/usr/bin/env python3
"""
Test script to verify the organized TimeGeo package structure.
"""

import os
import sys
from pathlib import Path


def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing package imports...")
    
    try:
        from config import TimeGeoConfig
        print("✅ TimeGeoConfig imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TimeGeoConfig: {e}")
        return False
    
    try:
        from workflow import TimeGeoWorkflow
        print("✅ TimeGeoWorkflow imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TimeGeoWorkflow: {e}")
        return False
    
    try:
        from data_processor import DataProcessor
        print("✅ DataProcessor imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import DataProcessor: {e}")
        return False
    
    try:
        from utils.parquet_utils import ParquetHandler
        print("✅ ParquetHandler imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ParquetHandler: {e}")
        return False
    
    try:
        from utils.file_utils import FileHandler
        print("✅ FileHandler imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import FileHandler: {e}")
        return False
    
    try:
        from utils.cpp_utils import CppModuleHandler
        print("✅ CppModuleHandler imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import CppModuleHandler: {e}")
        return False
    
    return True


def test_config():
    """Test configuration creation and validation."""
    print("\nTesting configuration...")
    
    try:
        from config import TimeGeoConfig
        
        # Test basic config creation
        config = TimeGeoConfig(input_parquet_path="/test/path/data.parquet")
        print("✅ Basic config created successfully")
        
        # Test custom config
        custom_config = TimeGeoConfig(
            input_parquet_path="/test/path/data.parquet",
            num_cpus=8,
            num_stays_threshold=20,
            output_base_dir="./test_results"
        )
        print("✅ Custom config created successfully")
        
        # Test output directories
        dirs = custom_config.get_output_directories()
        expected_keys = [
            "parameters", "parameters_commuters", "parameters_noncommuters",
            "srfiltered", "simulation", "simulation_locations", "simulation_parameters",
            "simulation_mapped", "simulation_compressed", "analysis", "figs", "data_cdr"
        ]
        
        for key in expected_keys:
            if key not in dirs:
                print(f"❌ Missing output directory key: {key}")
                return False
        
        print("✅ Output directories generated correctly")
        
        # Test validation (should fail for non-existent file)
        try:
            custom_config.validate()
            print("❌ Validation should have failed for non-existent file")
            return False
        except FileNotFoundError:
            print("✅ Validation correctly caught non-existent file")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_utils():
    """Test utility classes."""
    print("\nTesting utility classes...")
    
    try:
        from utils.parquet_utils import ParquetHandler
        from utils.file_utils import FileHandler
        from utils.cpp_utils import CppModuleHandler
        
        # Test ParquetHandler
        parquet_handler = ParquetHandler()
        print("✅ ParquetHandler created successfully")
        
        # Test FileHandler
        FileHandler.ensure_dir("./test_dir")
        if os.path.exists("./test_dir"):
            print("✅ FileHandler.ensure_dir works")
            os.rmdir("./test_dir")
        else:
            print("❌ FileHandler.ensure_dir failed")
            return False
        
        # Test CppModuleHandler
        cpp_handler = CppModuleHandler()
        module_info = cpp_handler.check_module_availability()
        print(f"✅ CppModuleHandler created, module info: {module_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utils test failed: {e}")
        return False


def test_package_structure():
    """Test that all required files exist."""
    print("\nTesting package structure...")
    
    required_files = [
        "__init__.py",
        "config.py",
        "workflow.py",
        "data_processor.py",
        "example_usage.py",
        "README.md",
        "utils/__init__.py",
        "utils/parquet_utils.py",
        "utils/file_utils.py",
        "utils/cpp_utils.py"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING ORGANIZED TIMEGEO PACKAGE")
    print("=" * 60)
    
    tests = [
        ("Package Structure", test_package_structure),
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Utilities", test_utils)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Package is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 