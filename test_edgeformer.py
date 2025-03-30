#!/usr/bin/env python
"""Test runner for EdgeFormer components."""
import os
import sys
import argparse
import unittest
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_runner')

def run_tests(component=None, verbose=False):
    """Run tests for specified component or all tests."""
    # Configure test discovery
    test_dir = 'tests'
    pattern = f'test_{component}*.py' if component else 'test_*.py'
    verbosity = 2 if verbose else 1
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Discover and run tests
    suite = loader.discover(test_dir, pattern=pattern)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Run tests for EdgeFormer components")
    parser.add_argument("--component", type=str, help="Specific component to test (e.g., 'memory', 'attention')")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    logger.info(f"Running tests for EdgeFormer{'s ' + args.component if args.component else ''}")
    
    success = run_tests(args.component, args.verbose)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()