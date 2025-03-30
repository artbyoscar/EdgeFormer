#!/usr/bin/env python
"""Validate and fix EdgeFormer dependencies."""
import sys
import subprocess
import importlib.util
import pkg_resources

# Required packages
REQUIRED_PACKAGES = {
    "torch": ">=1.10.0",
    "numpy": ">=1.19.0",
    "matplotlib": ">=3.3.0",
    "psutil": ">=5.8.0",
    "tqdm": ">=4.62.0",
}

# Optional packages for specific features
OPTIONAL_PACKAGES = {
    "limo_training": ["nltk", "textstat", "scikit-learn", "pandas", "seaborn"],
    "visualization": ["matplotlib>=3.3.0", "seaborn>=0.11.0"],
    "benchmarking": ["psutil>=5.8.0", "pandas>=1.3.0"],
}

# NLTK data requirements
NLTK_DATA = ["punkt", "wordnet", "stopwords", "averaged_perceptron_tagger"]

def check_package(package_name, min_version=None):
    """Check if a package is installed and meets the minimum version."""
    try:
        if min_version:
            pkg_resources.require(f"{package_name}>={min_version}")
        else:
            importlib.util.find_spec(package_name)
        return True
    except (ImportError, pkg_resources.VersionConflict):
        return False

def check_and_install_nltk_data(data_name):
    """Check if NLTK data is downloaded and download if missing."""
    try:
        import nltk
        try:
            nltk.data.find(f"corpora/{data_name}")
            return True
        except LookupError:
            print(f"Downloading NLTK data: {data_name}")
            nltk.download(data_name, quiet=True)
            return True
    except Exception as e:
        print(f"Error checking/installing NLTK data {data_name}: {e}")
        return False

def main():
    """Main entry point for dependency validation."""
    print("EdgeFormer Dependency Validator")
    print("===============================")
    
    # Check required packages
    missing_required = []
    for package, version in REQUIRED_PACKAGES.items():
        if ">" in version:
            name, ver = package, version.replace(">=", "")
        else:
            name, ver = package, None
        
        if not check_package(name, ver):
            missing_required.append(f"{name}>={ver}" if ver else name)
    
    if missing_required:
        print(f"Missing required packages: {', '.join(missing_required)}")
        install = input("Install missing required packages? (y/n): ").lower() == 'y'
        if install:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_required)
            print("Required packages installed successfully.")
        else:
            print("Warning: Missing required packages may cause issues.")
    else:
        print("All required packages are installed.")
    
    # Check NLTK data
    try:
        import nltk
        for data in NLTK_DATA:
            check_and_install_nltk_data(data)
        print("NLTK data validated.")
    except ImportError:
        print("NLTK not installed, skipping data validation.")
    
    print("\nValidation complete.")

if __name__ == "__main__":
    main()