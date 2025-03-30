#!/usr/bin/env python
# run_tests.py
import argparse
import unittest
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("test_runner")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run EdgeFormer tests")
    
    parser.add_argument(
        "--component",
        type=str,
        default=None,
        choices=["model", "memory", "optimization", "all"],
        help="Component to test",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display verbose output",
    )
    
    parser.add_argument(
        "--test-pattern",
        type=str,
        default="test_*.py",
        help="Pattern for test files",
    )
    
    return parser.parse_args()

def discover_tests(component=None, pattern="test_*.py"):
    """Discover tests based on component."""
    if component == "model":
        start_dir = "tests/model"
    elif component == "memory":
        start_dir = "tests/memory"
    elif component == "optimization":
        start_dir = "tests/optimization"
    else:  # All tests
        start_dir = "tests"
    
    # Create directory if it doesn't exist
    os.makedirs(start_dir, exist_ok=True)
    
    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir, pattern=pattern)
    
    return suite

def run_selected_tests(component=None, verbose=False, pattern="test_*.py"):
    """Run selected tests."""
    # Get test suite
    suite = discover_tests(component, pattern)
    
    # Set verbosity
    verbosity = 2 if verbose else 1
    
    # Create test runner
    runner = unittest.TextTestRunner(verbosity=verbosity)
    
    # Run tests
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()

def main():
    """Main function."""
    args = parse_args()
    
    # Log test configuration
    component_str = args.component if args.component else "all"
    logger.info(f"Running tests for component: {component_str}")
    
    # Run tests
    success = run_selected_tests(
        component=args.component,
        verbose=args.verbose,
        pattern=args.test_pattern,
    )
    
    # Set exit code based on success
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()