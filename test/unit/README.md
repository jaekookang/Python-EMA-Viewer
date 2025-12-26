# Unit Tests for Python-EMA-Viewer

## Test Structure

```
test/
├── run_tests.py          # Main test runner
└── unit/
    ├── __init__.py       # Package init
    ├── test_utils.py     # Tests for utility functions
    ├── test_viewer.py    # Tests for Viewer class
    └── test_integration.py  # Integration tests with example data
```

## Running Tests

### Run all tests
```bash
source .venv/bin/activate
python test/run_tests.py
```

### Run with verbose output
```bash
python test/run_tests.py -v
```

### Run specific test categories
```bash
python test/run_tests.py unit          # Unit tests only
python test/run_tests.py integration   # Integration tests only
```

### Run individual test files
```bash
python -m unittest test.unit.test_utils
python -m unittest test.unit.test_viewer
python -m unittest test.unit.test_integration
```

### Run specific test class
```bash
python -m unittest test.unit.test_utils.TestLoadPkl
```

### Run specific test method
```bash
python -m unittest test.unit.test_utils.TestLoadPkl.test_load_valid_pkl
```

## Test Coverage

### test_utils.py
- `TestLoadPkl`: Tests for pickle file loading
  - Valid pickle files
  - Invalid data types
  - Empty dictionaries
  - Non-existent files

- `TestCheckDictionary`: Tests for dictionary structure validation
  - Valid dictionary structure
  - Missing channels
  - Extra channels
  - Missing fields

### test_viewer.py
- `TestViewerInit`: Tests for Viewer initialization
  - Default parameters
  - Custom parameters

- `TestViewerLoad`: Tests for file loading
  - Pickle files
  - Non-existent files
  - Directories (not files)
  - Invalid file extensions

- `TestViewerMat2Py`: Tests for mat2py conversion
  - Without loading data first
  - Saving to file

- `TestViewerUpdateMethods`: Tests for update methods
  - update_audio with invalid files

### test_integration.py
- Integration tests with actual example data
  - Loading example pickle files
  - Loading example mat files
  - Complete workflow testing

## Adding New Tests

1. Create a new test file in `test/unit/` with prefix `test_`
2. Import unittest and the code to test
3. Create test classes inheriting from `unittest.TestCase`
4. Write test methods with prefix `test_`
5. Run tests using the test runner

Example:
```python
import unittest
from mviewer import your_function

class TestYourFunction(unittest.TestCase):
    def test_basic_case(self):
        result = your_function(input)
        self.assertEqual(result, expected)
```

## Notes

- Tests use Python's built-in `unittest` framework
- Temporary files are created in `tempfile.mkdtemp()` and cleaned up in `tearDown()`
- Integration tests are skipped if example files are not available
- All tests should be independent and not rely on order of execution
