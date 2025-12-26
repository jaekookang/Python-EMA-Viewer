# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python-EMA-Viewer is a specialized tool for working with Electromagnetic Articulography (EMA) data in speech research. It converts between MATLAB `.mat` files (native format from EMA machines) and Python `.pkl` files, visualizes articulatory trajectories with audio waveforms, and generates synchronized animations.

## Development Commands

### Running the Main Module
```bash
python mviewer.py
# Runs demo workflow: load → convert → plot → animate
```

### Running Test/Demo Script
```bash
python test/test_demo.py
# Demonstrates loading, plotting, and animation generation
# Includes clear progress messages and error handling
```

### Testing with Interactive Python
```python
from mviewer import Viewer

# Default usage (backward compatible)
mm = Viewer()
mm.load('example/F01_B01_S01_R01_N.mat')
mm.mat2py(save_file='example/test.pkl')
mm.plot(channel_list=['AUDIO','TR','TB','TT'], show=True)
mm.animate('result/test.mov', channel_list=['AUDIO','TR','TB','TT'])

# New usage with explicit dataset type
mm = Viewer(dataset_type='haskins')
mm.load('example/F01_B01_S01_R01_N.mat')
mm.mat2py(save_file='example/test.pkl')

# Custom sampling rates
mm = Viewer(wav_sr=48000, ema_sr=200)
```

## Environment Setup with uv

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### Activate Virtual Environment
```bash
source .venv/bin/activate
```

### Install Dependencies
```bash
# Install packages using uv
uv pip install <package-name>

# Install from requirements.txt
uv pip install -r requirements.txt
```

### Update requirements.txt
After installing new packages, always update requirements.txt:
```bash
uv pip freeze > requirements.txt
```

### Running Python with uv Environment
```bash
# Activate first, then run
source .venv/bin/activate && python mviewer.py

# Or run demo script
source .venv/bin/activate && python test/test_demo.py

# Or run unit tests
source .venv/bin/activate && python test/run_tests.py
```

## Architecture

### Core Module: mviewer.py

The module uses an **Adapter Pattern** to support multiple EMA datasets. The architecture consists of:

**1. Utility Functions (Lines 40-146):**
- Data loading functions that handle `.mat` and `.pkl` formats
- `get_struct()`: The critical conversion engine that transforms MATLAB struct arrays into Python dictionaries
- TextGrid parsing for linguistic annotation integration

**2. Dataset Adapter Pattern (Lines 149-254):**
- **DatasetAdapter (ABC)**: Abstract base class defining the interface for dataset loaders
  - `load_file()`: Load raw dataset file
  - `parse_to_standard_format()`: Convert to standardized internal format
  - Properties: `channel_names`, `field_names`, `default_ema_srate`, `default_audio_srate`

- **HaskinsAdapter**: Implementation for Haskins IEEE Rate Comparison Dataset
  - Supports `.mat` files with Haskins-specific struct format
  - Channels: TR, TB, TT, UL, LL, JAW (tongue, lips, jaw sensors)
  - Audio: 44.1 kHz, EMA: 100 Hz

**3. Viewer Class (Lines 257-568):**
- Central API for all operations
- Uses adapters internally for dataset-specific operations
- Stateful design: stores loaded data in `self.data_orig` and processed data in `self.data`
- Methods are designed to be chained: `load()` → `mat2py()` → `plot()/animate()`

**Backward Compatibility:** The Viewer class maintains full backward compatibility with legacy code. Old initialization without parameters still works.

### Data Flow Architecture

```
MATLAB .mat file (from EMA machine)
    ↓ load_mat()
Raw MATLAB struct array (self.data_orig)
    ↓ get_struct() via mat2py()
Python dictionary (self.data)
    ↓ Optional: pickle.dump()
.pkl file for Python processing
```

**Reverse flow:**
```
Python dictionary
    ↓ py2mat()
Restructured MATLAB-compatible format
    ↓ scipy.io.savemat()
.mat file compatible with MATLAB mview
```

### Critical Data Structure

The Python dictionary uses channel names as top-level keys (`AUDIO`, `TR`, `TB`, `TT`, `UL`, `LL`, `JAW`, `JAWL`). Each channel contains identical field structure:

- **NAME**: Channel identifier string
- **SRATE**: Sampling rate (float)
- **SIGNAL**: Time-series data (numpy array, 3D coordinates for EMA channels)
- **SOURCE**: Original file identifier
- **SENTENCE**: Utterance text
- **WORDS**: Dict with `LABEL` (list) and `OFFS` (numpy array)
- **PHONES**: Dict with `LABEL` (list) and `OFFS` (numpy array)
- **LABELS**: Dict with `NAME`, `OFFSET`, `VALUE`, `HOOK` arrays

**Important:** Only the `AUDIO` channel contains linguistic metadata (WORDS, PHONES, LABELS). Other channels have empty lists for these fields.

### Meta Information Configuration

The `Viewer` class constructor accepts customizable metadata:

```python
# New recommended usage
Viewer(
    dataset_type='haskins',  # Explicit dataset type
    ignore_meta=False,       # Set True if uncertain about field/channel names
    wav_sr=None,            # None = use dataset defaults (44100 for Haskins)
    ema_sr=None             # None = use dataset defaults (100 for Haskins)
)

# Legacy usage (deprecated but still supported)
Viewer(
    field_names=['NAME', 'SRATE', 'SIGNAL', ...],  # Deprecated
    channel_names=['TR', 'TB', 'TT', 'UL', 'LL', 'JAW'],  # Deprecated
    audio_channel='AUDIO',  # Deprecated
    wav_sr=44100,
    ema_sr=100
)
```

**Key design decision:** If `ignore_meta=True`, the adapter extracts field/channel names directly from loaded data instead of validating against expected names. This allows flexibility when working with different EMA datasets.

**Adapter Pattern Benefits:**
- Easily extensible to new datasets (mngu0, MOCHA-TIMIT, XRMB)
- Dataset-specific logic is isolated in adapters
- Backward compatible with existing code

### Dataset Compatibility

**Currently Supported:**
- **Haskins IEEE Rate Comparison Dataset** (via HaskinsAdapter)
  - Channel naming: TR, TB, TT, UL, LL, JAW (tongue, lips, jaw sensors)
  - Field structure: NAME, SRATE, SIGNAL, SOURCE, SENTENCE, WORDS, PHONES, LABELS
  - Sampling rates: 44100 Hz audio, 100 Hz EMA
  - File format: MATLAB `.mat` files

**Future Extensions (Framework Ready):**
To add support for new datasets (mngu0, MOCHA-TIMIT, XRMB):
1. Create a new adapter class inheriting from `DatasetAdapter`
2. Implement required methods: `load_file()`, `parse_to_standard_format()`, properties
3. Register in `Viewer._get_adapter()` factory method

Example stub:
```python
class Mngu0Adapter(DatasetAdapter):
    DATASET_NAME = "mngu0"
    SUPPORTED_EXTENSIONS = ['.ema']

    @property
    def channel_names(self) -> list:
        return ['T1', 'T2', 'T3', 'T4', 'UL', 'LL', 'LI']

    def load_file(self, file_path: str) -> dict:
        # TODO: implement mngu0 binary reader
        pass

    def parse_to_standard_format(self, raw_data) -> dict:
        # TODO: convert mngu0 format to standard
        pass
```

## Animation Generation

The `animate()` method (lines 359-411) has specific requirements:

- **Requires ffmpeg** installed and available in PATH
- Creates temporary files (`tmp.wav`, `tmp.mp4`) in output directory
- Uses matplotlib.animation with ffmpeg writer
- Final step combines video and audio using ffmpeg command-line tool
- Only supports `.mov` output format

**Frame rate calculation:** `fps = sr_wav / div` where `div=1000` is hardcoded. This creates ~44 fps for 44100 Hz audio.

## Linguistic Annotation Updates

Two static methods handle post-processing:

**`update_meta()`:** Updates WORDS/PHONES/LABELS from corrected TextGrid files. Use when phonetic annotations have been manually corrected in Praat or similar tools.

**`update_audio()`:** Replaces audio signal when preprocessing has been applied (noise reduction, normalization). Validates that sample length and sampling rate match original.

Both methods use `check_dictionary()` to ensure data structure integrity before modifications.

## Dependencies

**Current dependencies** are maintained in `requirements.txt`. Use `uv pip install` (not standard `pip install`) to manage packages.

**Legacy README requirements** (original versions from 2020):
```
numpy==1.18.5
scipy==1.4.1
matplotlib==3.3.3
seaborn==0.11.0
tgt==1.4.4
```

**Current versions** in requirements.txt (as of 2025):
```
numpy==2.4.0
scipy==1.16.3
matplotlib==3.10.8
seaborn==0.13.2
tgt (install with: uv pip install tgt)
```

**External tool:** ffmpeg (required for animation generation only)

## File Locations

- **Example data:** `example/` directory (contains `.mat` and `.pkl` samples)
- **Output directory:** `result/` (for plots `.png` and animations `.mov`)
- **Test directory:** `test/` directory
  - `test/run_tests.py` - Main test runner
  - `test/test_demo.py` - Demo script with visualization
  - `test/test_ieee.ipynb` - Jupyter notebook for manual testing
  - `test/unit/` - Unit test modules (21 tests total)
- **Documentation:** `docs/` directory
  - `docs/mviewer_guide.md` - Complete user guide with API reference
- **Session logs:** `session_logs/` (project documentation, summaries)

## History logs
- `./session_logs/session_{LATESTDATE}.md` (e.g., session_251226.md)
- Don't create md files whenever a fix was created. Just update the current date's session log only