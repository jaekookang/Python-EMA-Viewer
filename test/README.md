# Testing Guide for Python-EMA-Viewer

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Run all tests
python test/run_tests.py

# Run with verbose output
python test/run_tests.py -v
```

## Test Structure

```
test/
├── README.md             # This file
├── run_tests.py          # Main test runner script
├── test_ieee.ipynb       # Jupyter notebook for manual testing
└── unit/
    ├── README.md         # Detailed unit test documentation
    ├── __init__.py
    ├── test_utils.py     # Tests for utility functions
    ├── test_viewer.py    # Tests for Viewer class
    └── test_integration.py  # Integration tests with real data
```

## Running Tests

### All tests (recommended)
```bash
python test/run_tests.py
```

### Unit tests only
```bash
python test/run_tests.py unit
```

### Integration tests only
```bash
python test/run_tests.py integration
```

### Specific test file
```bash
python -m unittest test.unit.test_utils
python -m unittest test.unit.test_viewer
python -m unittest test.unit.test_integration
```

### Specific test class
```bash
python -m unittest test.unit.test_utils.TestLoadPkl
python -m unittest test.unit.test_viewer.TestViewerInit
```

### Specific test method
```bash
python -m unittest test.unit.test_utils.TestLoadPkl.test_load_valid_pkl
```

## Test Coverage Summary

### Unit Tests (test_utils.py)
- ✅ `load_pkl()` - Pickle file loading
- ✅ `check_dictionary()` - Dictionary validation
- Total: 8 tests

### Unit Tests (test_viewer.py)
- ✅ `Viewer.__init__()` - Initialization
- ✅ `Viewer.load()` - File loading
- ✅ `Viewer.mat2py()` - MAT to Python conversion
- ✅ `Viewer.update_audio()` - Audio updating
- Total: 7 tests

### Integration Tests (test_integration.py)
- ✅ Example file loading (.pkl, .mat)
- ✅ Full workflow testing
- ✅ Data structure validation
- Total: 4 tests

**Total: 21 tests** (20 passing, 1 skipped)

## Test Results

Expected output when all tests pass:
```
Running all tests...
----------------------------------------------------------------------
..................s..
----------------------------------------------------------------------
Ran 21 tests in 0.005s

OK (skipped=1)
----------------------------------------------------------------------
All tests passed!
```

## What's Tested

### ✅ Covered
- File loading (`.pkl` files)
- Dictionary structure validation
- Viewer initialization
- File existence and type checking
- Exception handling
- Integration with example data

### ⚠️ Partially Covered
- MAT file conversion (requires complex mocking)
- Plot generation (requires display/matplotlib backend)
- Animation generation (requires ffmpeg)

### ❌ Not Covered
- TextGrid parsing (requires .TextGrid test files)
- Audio updates (requires .wav test files)
- Full MAT to Python workflow (complex to mock)

## Adding New Tests

1. Create test file in `test/unit/` with `test_` prefix
2. Import required modules and code to test
3. Create test class inheriting from `unittest.TestCase`
4. Add test methods with `test_` prefix
5. Run tests with test runner

Example:
```python
import unittest
from mviewer import your_function

class TestYourFunction(unittest.TestCase):
    def setUp(self):
        """Setup before each test"""
        self.test_data = create_test_data()

    def tearDown(self):
        """Cleanup after each test"""
        cleanup_test_files()

    def test_basic_functionality(self):
        """Test description"""
        result = your_function(self.test_data)
        self.assertEqual(result, expected_value)
```

## Best Practices

1. **Isolation**: Each test should be independent
2. **Cleanup**: Use `tearDown()` to remove temporary files
3. **Clear Names**: Test names should describe what they test
4. **Skip When Needed**: Use `@unittest.skipIf` for conditional tests
5. **Use Assertions**: Choose the right assertion method
   - `assertEqual(a, b)` - Check equality
   - `assertTrue(x)` / `assertFalse(x)` - Check boolean
   - `assertIn(a, b)` - Check membership
   - `assertRaises(Exception)` - Check exceptions
   - `assertIsNone(x)` / `assertIsNotNone(x)` - Check None

## Continuous Integration

To integrate with CI/CD:

```yaml
# Example for GitHub Actions
- name: Run tests
  run: |
    source .venv/bin/activate
    python test/run_tests.py
```

## Troubleshooting

### Tests fail with import errors
```bash
# Make sure you're in the project root and environment is activated
source .venv/bin/activate
python test/run_tests.py
```

### Integration tests are skipped
```
# Integration tests require example files
# Make sure example/ directory contains:
# - F01_B01_S01_R01_N.pkl
# - F01_B01_S01_R01_N.mat
```

### Permission errors on temporary files
```bash
# Check temp directory permissions
# Tests create files in system temp directory
python -c "import tempfile; print(tempfile.gettempdir())"
```

## Manual Testing

For manual testing and experimentation, use the Jupyter notebook:
```bash
source .venv/bin/activate
jupyter notebook test/test_ieee.ipynb
```

## Further Documentation

See `test/unit/README.md` for detailed information about individual test cases and test data structures.
