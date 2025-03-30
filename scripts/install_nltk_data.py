#!/usr/bin/env python
"""Install NLTK data packages required for LIMO training."""
import os
import sys
import argparse
import nltk

# Required NLTK data packages for LIMO training
REQUIRED_NLTK_DATA = [
    "punkt",
    "wordnet",
    "stopwords",
    "averaged_perceptron_tagger"
]

def download_nltk_data(data_list, download_dir=None):
    """Download specified NLTK data packages."""
    for data_name in data_list:
        try:
            print(f"Downloading {data_name}...")
            nltk.download(data_name, download_dir=download_dir, quiet=False)
            print(f"Successfully downloaded {data_name}")
        except Exception as e:
            print(f"Error downloading {data_name}: {e}")

def main():
    """Main entry point for NLTK data installation."""
    parser = argparse.ArgumentParser(description="Install NLTK data for EdgeFormer LIMO training")
    parser.add_argument("--download-dir", type=str, help="Custom download directory for NLTK data")
    parser.add_argument("--all", action="store_true", help="Download all required NLTK data")
    parser.add_argument("--list", action="store_true", help="List required NLTK data packages")
    
    args = parser.parse_args()
    
    if args.list:
        print("Required NLTK data packages for LIMO training:")
        for data in REQUIRED_NLTK_DATA:
            print(f"- {data}")
        return
    
    if args.all:
        download_nltk_data(REQUIRED_NLTK_DATA, args.download_dir)
    else:
        # Interactive mode
        print("NLTK Data Installer for EdgeFormer LIMO Training")
        print("==============================================")
        print("Required NLTK data packages:")
        for i, data in enumerate(REQUIRED_NLTK_DATA):
            print(f"{i+1}. {data}")
        
        print("\nOptions:")
        print("1. Install all packages")
        print("2. Install specific packages")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            download_nltk_data(REQUIRED_NLTK_DATA, args.download_dir)
        elif choice == "2":
            indices = input("Enter package numbers to install (comma-separated): ").split(",")
            try:
                selected = [REQUIRED_NLTK_DATA[int(idx.strip())-1] for idx in indices]
                download_nltk_data(selected, args.download_dir)
            except (ValueError, IndexError):
                print("Invalid selection")
        else:
            print("Exiting without installation")

if __name__ == "__main__":
    main()