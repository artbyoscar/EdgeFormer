# src/models/advanced_optimizer.py
import logging
import torch
import platform
import psutil
import os

try:
    from src.utils.logging_utils import get_logger
    logger = get_logger('advanced_optimizer')
except ImportError:
    import logging
    logger = logging.getLogger('advanced_optimizer')
    logger.setLevel(logging.INFO)

class Phase2Optimizer:
    """Advanced optimization techniques for Phase 2 hardware support"""
    
    def __init__(self, device_info=None):
        self.device_info = device_info or self._detect_device()
        logger.info(f"Initialized Phase2Optimizer for device: {self.device_info['device_name']}")
    
    def _detect_device(self):
        """Detect device capabilities and characteristics"""
        device_info = {
            'device_name': platform.node(),
            'system': platform.system(),
            'processor': platform.processor(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        }
        
        # Get CPU info
        device_info['cpu_count'] = psutil.cpu_count(logical=False)
        device_info['cpu_threads'] = psutil.cpu_count(logical=True)
        
        # Get GPU info if available
        if device_info['cuda_available'] and device_info['cuda_device_count'] > 0:
            device_info['gpu_name'] = torch.cuda.get_device_name(0)
            device_info['cuda_device_capability'] = torch.cuda.get_device_capability(0)
            device_info['cuda_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        
        # Detect if running on battery (Windows)
        if device_info['system'] == 'Windows':
            try:
                import ctypes
                from ctypes import wintypes
                
                class SYSTEM_POWER_STATUS(ctypes.Structure):
                    _fields_ = [
                        ('ACLineStatus', wintypes.BYTE),
                        ('BatteryFlag', wintypes.BYTE),
                        ('BatteryLifePercent', wintypes.BYTE),
                        ('Reserved1', wintypes.BYTE),
                        ('BatteryLifeTime', wintypes.DWORD),
                        ('BatteryFullLifeTime', wintypes.DWORD),
                    ]
                
                SYSTEM_POWER_STATUS_P = ctypes.POINTER(SYSTEM_POWER_STATUS)
                GetSystemPowerStatus = ctypes.windll.kernel32.GetSystemPowerStatus
                GetSystemPowerStatus.argtypes = [SYSTEM_POWER_STATUS_P]
                GetSystemPowerStatus.restype = wintypes.BOOL
                
                status = SYSTEM_POWER_STATUS()
                if GetSystemPowerStatus(ctypes.pointer(status)):
                    # ACLineStatus: 1 = online, 0 = offline (on battery)
                    device_info['on_battery'] = status.ACLineStatus == 0
                else:
                    device_info['on_battery'] = False
            except Exception:
                device_info['on_battery'] = False
        else:
            # For non-Windows systems (simplified)
            device_info['on_battery'] = False
            
        # Detect specific hardware vendors
        if 'AMD' in device_info['processor'] or 'Ryzen' in device_info['processor']:
            device_info['vendor'] = 'AMD'
        elif 'Intel' in device_info['processor']:
            device_info['vendor'] = 'Intel'
        elif 'ARM' in device_info['architecture'] or 'aarch64' in device_info['architecture']:
            device_info['vendor'] = 'ARM'
        else:
            device_info['vendor'] = 'Unknown'
            
        return device_info
    
    def optimize_model(self, model, config):
        """Apply Phase 2 optimizations to the model"""
        logger.info("Applying Phase 2 optimizations")
        
        # Apply vendor-specific optimizations
        if self.device_info['vendor'] == 'AMD':
            model = self._optimize_for_amd(model, config)
        elif self.device_info['vendor'] == 'Intel':
            model = self._optimize_for_intel(model, config)
        elif self.device_info['vendor'] == 'ARM':
            model = self._optimize_for_arm(model, config)
        
        # Apply general optimizations based on available memory
        model = self._optimize_for_memory(model, config)
        
        # Apply power-aware optimizations
        model = self._apply_power_optimizations(model, config)
        
        return model
    
    def _optimize_for_amd(self, model, config):
        """Apply AMD-specific optimizations"""
        logger.info("Applying AMD-specific optimizations")
        
        # AMD-specific optimizations here
        # Adjust thread count, memory layout, etc.
        if hasattr(torch, 'set_num_threads'):
            cores = self.device_info.get('cpu_count', 4)
            threads = self.device_info.get('cpu_threads', 8)
            torch.set_num_threads(threads)
            
        # Set AMD-optimized configs if available
        if hasattr(config, 'attention_implementation'):
            config.attention_implementation = 'amd_optimized'
            
        return model
    
    def _optimize_for_intel(self, model, config):
        """Apply Intel-specific optimizations"""
        logger.info("Applying Intel-specific optimizations")
        
        # Intel-specific optimizations here
        if hasattr(torch, 'set_num_threads'):
            cores = self.device_info.get('cpu_count', 4)
            threads = self.device_info.get('cpu_threads', 8)
            optimal_threads = max(1, threads - 2)  # Leave 2 threads for OS
            torch.set_num_threads(optimal_threads)
            
        # Try to use Intel MKL if available
        try:
            import mkl
            mkl.set_num_threads(optimal_threads)
        except ImportError:
            pass
            
        return model
    
    def _optimize_for_arm(self, model, config):
        """Apply ARM-specific optimizations"""
        logger.info("Applying ARM-specific optimizations")
        
        # ARM-specific optimizations here
        # Example: Enable NEON instructions, mobile optimizations
        if hasattr(config, 'use_neon_acceleration'):
            config.use_neon_acceleration = True
            
        return model
    
    def _optimize_for_memory(self, model, config):
        """Apply memory-based optimizations"""
        available_ram_gb = self.device_info['ram_gb']
        logger.info(f"Optimizing for system with {available_ram_gb}GB RAM")
        
        # Adjust KV cache strategies based on available memory
        if available_ram_gb < 4:
            # Ultra low memory mode
            if hasattr(config, 'kv_cache_strategy'):
                config.kv_cache_strategy = 'minimal'
            if hasattr(config, 'max_sequence_length'):
                config.max_sequence_length = min(config.max_sequence_length, 1024)
        elif available_ram_gb < 8:
            # Low memory mode
            if hasattr(config, 'kv_cache_strategy'):
                config.kv_cache_strategy = 'efficient'
            if hasattr(config, 'max_sequence_length'):
                config.max_sequence_length = min(config.max_sequence_length, 2048)
        
        return model
    
    def _apply_power_optimizations(self, model, config):
        """Apply power-aware optimizations"""
        logger.info("Applying power-aware optimizations")
        
        # Check if device is running on battery
        on_battery = self.device_info.get('on_battery', False)
        
        if on_battery and hasattr(config, 'power_mode'):
            config.power_mode = 'efficient'
            
            # Adjust batch size for power efficiency
            if hasattr(config, 'max_batch_size'):
                config.max_batch_size = min(config.max_batch_size, 4)
                
            # Enable compute budget forcing
            if hasattr(config, 'enforce_compute_budget'):
                config.enforce_compute_budget = True
                config.compute_budget_level = 'conservative'
        
        return model
    
    def get_hardware_profile(self):
        """Generate a hardware profile for benchmarking"""
        return self.device_info

def apply_phase2_optimizations(model, config):
    """Convenience function to apply Phase 2 optimizations"""
    optimizer = Phase2Optimizer()
    return optimizer.optimize_model(model, config)
