#!/usr/bin/env python
"""Generate detailed device profiles for optimization."""
import os
import json
import platform
import psutil
import argparse
import logging
import datetime
import subprocess
import cpuinfo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('device_profiler')

def get_cpu_info():
    """Get detailed CPU information."""
    info = {}
    
    # Use py-cpuinfo for detailed CPU information
    try:
        cpu_info = cpuinfo.get_cpu_info()
        info.update({
            'brand': cpu_info.get('brand_raw', 'Unknown'),
            'vendor': cpu_info.get('vendor_id_raw', 'Unknown'),
            'arch': cpu_info.get('arch', platform.machine()),
            'bits': cpu_info.get('bits', 64),
            'count': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
            'frequency': cpu_info.get('hz_advertised_raw', [0, ''])[0] / 1000000000,
            'features': cpu_info.get('flags', []),
        })
    except Exception as e:
        logger.warning(f"Error getting detailed CPU info: {e}")
        # Fallback to platform module
        info.update({
            'brand': platform.processor(),
            'count': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
        })
    
    return info

def get_memory_info():
    """Get memory information."""
    vm = psutil.virtual_memory()
    return {
        'total': vm.total / (1024**3),  # GB
        'available': vm.available / (1024**3),  # GB
        'used_percent': vm.percent,
    }

def get_gpu_info():
    """Get GPU information if available."""
    try:
        # Try to detect NVIDIA GPUs
        nvidia_smi = subprocess.check_output('nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,noheader', shell=True)
        lines = nvidia_smi.decode('utf-8').strip().split('\n')
        
        gpus = []
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                gpus.append({
                    'index': i,
                    'name': parts[0],
                    'memory_total': parts[1],
                    'memory_free': parts[2],
                    'memory_used': parts[3],
                    'vendor': 'nvidia',
                })
        
        if gpus:
            return gpus
    except Exception:
        pass
    
    # If no NVIDIA GPUs, try AMD
    try:
        rocm_smi = subprocess.check_output('rocm-smi --showproductname --showmeminfo vram', shell=True)
        if 'GPU' in rocm_smi.decode('utf-8'):
            # Parse output (simplified)
            return [{'vendor': 'amd', 'detected': True}]
    except Exception:
        pass
    
    # If no dedicated GPU detected
    return [{'vendor': 'none', 'integrated_graphics': platform.machine() != 'armv7l'}]

def get_disk_info():
    """Get disk information."""
    disk = psutil.disk_usage('/')
    io_counters = psutil.disk_io_counters()
    
    return {
        'total': disk.total / (1024**3),  # GB
        'free': disk.free / (1024**3),  # GB
        'used_percent': disk.percent,
        'read_bytes': io_counters.read_bytes / (1024**3) if io_counters else 0,  # GB
        'write_bytes': io_counters.write_bytes / (1024**3) if io_counters else 0,  # GB
    }

def generate_device_profile(args):
    """Generate a complete device profile."""
    logger.info("Gathering device information...")
    
    profile = {
        'device_name': args.device_name or platform.node(),
        'system': {
            'os': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
        },
        'cpu': get_cpu_info(),
        'memory': get_memory_info(),
        'gpu': get_gpu_info(),
        'disk': get_disk_info(),
        'generated': datetime.datetime.now().isoformat(),
    }
    
    # Add optimization recommendations
    profile['optimization'] = generate_optimization_recommendations(profile)
    
    # Save to file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{profile['device_name']}.json")
    
    with open(output_file, 'w') as f:
        json.dump(profile, f, indent=2)
    
    logger.info(f"Device profile saved to {output_file}")
    return profile

def generate_optimization_recommendations(profile):
    """Generate optimization recommendations based on device profile."""
    recommendations = {}
    
    # CPU-based recommendations
    cpu = profile.get('cpu', {})
    if 'amd' in cpu.get('vendor', '').lower() or 'amd' in cpu.get('brand', '').lower():
        recommendations['attention_type'] = 'sliding_window'
    elif 'intel' in cpu.get('vendor', '').lower() or 'intel' in cpu.get('brand', '').lower():
        recommendations['attention_type'] = 'hybrid'
    else:
        recommendations['attention_type'] = 'standard'
    
    # Memory-based recommendations
    memory = profile.get('memory', {})
    if memory.get('total', 0) < 4:
        recommendations['max_sequence_length'] = 1024
        recommendations['quantization'] = 'int8'
        recommendations['kv_offload'] = True
    elif memory.get('total', 0) < 8:
        recommendations['max_sequence_length'] = 2048
        recommendations['quantization'] = 'int8'
        recommendations['kv_offload'] = True
    else:
        recommendations['max_sequence_length'] = 4096
        recommendations['quantization'] = 'fp16'
        recommendations['kv_offload'] = False
    
    # GPU-based recommendations
    gpu = profile.get('gpu', [{}])[0]
    if gpu.get('vendor') == 'nvidia':
        recommendations['device'] = 'cuda'
    elif gpu.get('vendor') == 'amd':
        recommendations['device'] = 'rocm'
    else:
        recommendations['device'] = 'cpu'
    
    return recommendations

def main():
    """Main entry point for device profiling."""
    parser = argparse.ArgumentParser(description="Generate EdgeFormer device profile")
    parser.add_argument('--device-name', type=str, help='Custom device name')
    parser.add_argument('--output-dir', type=str, default='profiles',
                        help='Directory to save device profiles')
    args = parser.parse_args()
    
    profile = generate_device_profile(args)
    
    # Print summary
    print("\nDevice Profile Summary")
    print("=====================")
    print(f"Device: {profile['device_name']}")
    print(f"CPU: {profile['cpu'].get('brand', 'Unknown')}")
    print(f"Cores: {profile['cpu'].get('count', 0)} (Physical) / {profile['cpu'].get('threads', 0)} (Logical)")
    print(f"Memory: {profile['memory'].get('total', 0):.1f} GB")
    print(f"GPU: {profile['gpu'][0].get('name', 'None') if profile['gpu'][0].get('vendor') != 'none' else 'None'}")
    
    print("\nRecommended Optimization Settings:")
    for key, value in profile['optimization'].items():
        print(f"- {key}: {value}")

if __name__ == "__main__":
    main()