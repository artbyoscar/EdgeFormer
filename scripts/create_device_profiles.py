#!/usr/bin/env python
# create_device_profiles.py

import argparse
import os
import json
import platform
import psutil
import torch
import sys

def get_device_info():
    """Gather current device information."""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "cpu": platform.processor(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    
    return info

def main():
    parser = argparse.ArgumentParser(description="Create device profiles for benchmarking")
    parser.add_argument("--devices", type=str, required=True, help="Comma-separated list of device names")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save device profiles")
    args = parser.parse_args()
    
    devices = args.devices.split(',')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # For the current device
    current_device_info = get_device_info()
    
    # Determine which device in the list matches the current one
    current_device_name = None
    for device in devices:
        response = input(f"Is this the {device} device? (y/n): ")
        if response.lower() == 'y':
            current_device_name = device
            break
    
    if current_device_name:
        # Save the profile
        profile_path = os.path.join(args.output_dir, f"{current_device_name}.json")
        with open(profile_path, 'w') as f:
            json.dump(current_device_info, f, indent=2)
        print(f"Created profile for {current_device_name} at {profile_path}")
    else:
        print("No device was identified. No profile created.")

if __name__ == "__main__":
    main()