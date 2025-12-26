#!/usr/bin/env python
"""
Test runner for Python-EMA-Viewer unit tests

Usage:
    python test/run_tests.py              # Run all tests
    python test/run_tests.py -v           # Verbose output
    python test/run_tests.py unit         # Run only unit tests
    python test/run_tests.py integration  # Run only integration tests
"""
import sys
import os
import unittest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_unit_tests(verbosity=1):
    """Run unit tests only"""
    loader = unittest.TestLoader()
    suite = loader.discover('test/unit', pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_integration_tests(verbosity=1):
    """Run integration tests only"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('test.unit.test_integration')
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_all_tests(verbosity=1):
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = loader.discover('test/unit', pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result.wasSuccessful()


def main():
    """Main entry point"""
    # Parse arguments
    verbosity = 2 if '-v' in sys.argv or '--verbose' in sys.argv else 1
    test_type = 'all'

    for arg in sys.argv[1:]:
        if arg in ['unit', 'integration', 'all']:
            test_type = arg

    print(f"Running {test_type} tests...")
    print("-" * 70)

    # Run tests
    if test_type == 'unit':
        success = run_unit_tests(verbosity)
    elif test_type == 'integration':
        success = run_integration_tests(verbosity)
    else:
        success = run_all_tests(verbosity)

    print("-" * 70)
    if success:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
