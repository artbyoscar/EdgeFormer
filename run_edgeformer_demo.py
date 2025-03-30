#!/usr/bin/env python
"""Main entry point for EdgeFormer demos."""
import os
import sys
import argparse
import logging
import importlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('edgeformer')

AVAILABLE_DEMOS = {
    "memory": "examples.htps_associative_memory_demo",
    "online_training": "examples.simplified_online_training_demo",
    "benchmark": "scripts.cross_device_benchmark"
}

def run_demo(demo_name, args):
    """Run the specified demo."""
    if demo_name not in AVAILABLE_DEMOS:
        logger.error(f"Unknown demo: {demo_name}")
        print(f"Available demos: {', '.join(AVAILABLE_DEMOS.keys())}")
        return False
    
    # Import and run the demo
    try:
        module_name = AVAILABLE_DEMOS[demo_name]
        module = importlib.import_module(module_name)
        
        # If module has a main function, call it
        if hasattr(module, 'main'):
            # Convert args to sys.argv format for the module
            old_argv = sys.argv
            sys.argv = [module_name] + args
            
            # Run the demo
            module.main()
            
            # Restore original argv
            sys.argv = old_argv
            return True
        else:
            logger.error(f"Demo module {module_name} does not have a main function")
            return False
    except ImportError as e:
        logger.error(f"Error importing demo {demo_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error running demo {demo_name}: {e}")
        return False

def main():
    """Main entry point for the EdgeFormer demo runner."""
    parser = argparse.ArgumentParser(description="EdgeFormer Demo Runner")
    parser.add_argument("demo", choices=list(AVAILABLE_DEMOS.keys()) + ["list"],
                       help="Demo to run or 'list' to see available demos")
    parser.add_argument("args", nargs="*", help="Arguments to pass to the demo")
    
    args = parser.parse_args()
    
    if args.demo == "list":
        print("Available EdgeFormer Demos:")
        for name, module in AVAILABLE_DEMOS.items():
            print(f"- {name}: {module}")
        return
    
    run_demo(args.demo, args.args)

if __name__ == "__main__":
    main()