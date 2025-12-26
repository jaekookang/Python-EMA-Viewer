# mviewer.py - Complete User Guide

**Version:** 2.0 (with Adapter Pattern)
**Last Updated:** 2025-12-27
**Author:** Python-EMA-Viewer Project

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Architecture Overview](#architecture-overview)
4. [Installation](#installation)
5. [Basic Usage](#basic-usage)
6. [Advanced Usage](#advanced-usage)
7. [API Reference](#api-reference)
8. [Extending for New Datasets](#extending-for-new-datasets)
9. [Troubleshooting](#troubleshooting)
10. [Examples](#examples)

---

## Introduction

`mviewer.py` is a specialized Python module for working with Electromagnetic Articulography (EMA) data in speech research. It provides tools for:

- **Format Conversion**: MATLAB `.mat` ↔ Python `.pkl`
- **Data Visualization**: Plot articulatory trajectories with synchronized audio
- **Animation Generation**: Create synchronized video animations with audio
- **Dataset Support**: Extensible architecture for multiple EMA datasets

### What is EMA?

Electromagnetic Articulography (EMA) is a technique used in speech research to track the movements of articulators (tongue, lips, jaw) during speech production. Small sensors are attached to articulators, and their 3D positions are tracked over time.

### Key Features

- ✅ Load and convert MATLAB `.mat` files from EMA systems
- ✅ Save as Python-friendly `.pkl` files
- ✅ Visualize multiple channels simultaneously
- ✅ Generate synchronized animations with audio
- ✅ Update metadata from TextGrid annotations
- ✅ Extensible adapter pattern for multiple datasets

---

## Quick Start

```python
from mviewer import Viewer

# 1. Create viewer instance
viewer = Viewer()

# 2. Load .mat file from EMA system
viewer.load('example/F01_B01_S01_R01_N.mat')

# 3. Convert to Python dictionary
viewer.mat2py(save_file='output.pkl')

# 4. Visualize the data
viewer.plot(channel_list=['AUDIO', 'TR', 'TB', 'TT'], show=True)

# 5. Create animation (requires ffmpeg)
viewer.animate('output.mov', channel_list=['AUDIO', 'TR', 'TB'])
```

---

## Architecture Overview

### Design Pattern: Adapter Pattern

The module uses the **Adapter Pattern** to support multiple EMA datasets while maintaining a unified API.

```
┌─────────────────────────────────────────────────────────────┐
│                        Viewer (API)                         │
│  User-facing interface for all operations                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  DatasetAdapter (ABC)                       │
│  Abstract interface for dataset loaders                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┬──────────────┐
        ▼              ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌──────────┐ ┌──────────┐
│  Haskins    │ │   Mngu0     │ │  MOCHA   │ │   XRMB   │
│  Adapter    │ │  Adapter    │ │ Adapter  │ │ Adapter  │
│ (Available) │ │  (Future)   │ │ (Future) │ │ (Future) │
└─────────────┘ └─────────────┘ └──────────┘ └──────────┘
```

### Core Components

**1. Utility Functions (Lines 40-146)**
- `load_mat()`: Load MATLAB `.mat` files
- `load_pkl()`: Load Python pickle files
- `load_textgrid()`: Parse Praat TextGrid annotations
- `check_dictionary()`: Validate data structure
- `get_struct()`: Convert MATLAB structs to Python dicts

**2. DatasetAdapter (Lines 149-204)**
- Abstract base class defining the interface
- Required methods: `load_file()`, `parse_to_standard_format()`
- Required properties: `channel_names`, `field_names`
- Optional properties: `default_ema_srate`, `default_audio_srate`

**3. HaskinsAdapter (Lines 207-254)**
- Concrete implementation for Haskins IEEE dataset
- Supports `.mat` files
- Channels: TR, TB, TT, UL, LL, JAW
- Sampling rates: 44100 Hz (audio), 100 Hz (EMA)

**4. Viewer Class (Lines 257-641)**
- Main user-facing API
- Methods: `load()`, `mat2py()`, `py2mat()`, `plot()`, `animate()`
- Static methods: `update_meta()`, `update_audio()`

### Data Flow

```
Input: MATLAB .mat file
    ↓
load() → adapter.load_file()
    ↓
Raw MATLAB struct (self.data_orig)
    ↓
mat2py() → adapter.parse_to_standard_format()
    ↓
Python dictionary (self.data)
    ↓
plot() / animate() / save to .pkl
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- uv package manager (recommended) or pip

### Dependencies

```bash
# Core dependencies
numpy>=2.0
scipy>=1.16
matplotlib>=3.10
seaborn>=0.13
tgt>=1.5

# External tool (for animation only)
ffmpeg
```

### Setup with uv (Recommended)

```bash
# 1. Clone or navigate to project
cd Python-EMA-Viewer

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Install ffmpeg (for animations)
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg
```

### Setup with pip

```bash
pip install numpy scipy matplotlib seaborn tgt
```

---

## Basic Usage

### 1. Loading Data

#### Load MATLAB .mat File

```python
from mviewer import Viewer

viewer = Viewer()
viewer.load('path/to/file.mat')
```

#### Load Python .pkl File

```python
viewer = Viewer()
viewer.load('path/to/file.pkl')
# Data is immediately available in viewer.data
```

### 2. Converting Data

#### MAT to PKL (MATLAB to Python)

```python
viewer = Viewer()
viewer.load('data.mat')

# Option 1: Convert and save
viewer.mat2py(save_file='output.pkl')

# Option 2: Convert and get dictionary
data = viewer.mat2py()
print(data.keys())  # ['AUDIO', 'TR', 'TB', 'TT', 'UL', 'LL', 'JAW']
```

#### PKL to MAT (Python to MATLAB)

```python
viewer = Viewer()
viewer.load('data.pkl')

# Convert back to .mat format
viewer.py2mat('output.mat', viewer.data, struct_var_name='F01_B01_S01_R01_N')
```

### 3. Visualizing Data

#### Basic Plot

```python
viewer = Viewer()
viewer.load('data.mat')
viewer.mat2py()

# Plot all channels
viewer.plot(show=True)

# Plot specific channels
viewer.plot(
    channel_list=['AUDIO', 'TR', 'TB', 'TT'],
    coordinates=['x', 'z'],
    show=True,
    file_name='plot.png'
)
```

#### Creating Animations

```python
viewer = Viewer()
viewer.load('data.mat')
viewer.mat2py()

# Create synchronized animation with audio
viewer.animate(
    'animation.mov',
    channel_list=['AUDIO', 'TR', 'TB', 'TT']
)
```

---

## Advanced Usage

### 1. Custom Initialization

#### Explicit Dataset Type

```python
# Recommended for new code
viewer = Viewer(dataset_type='haskins')
```

#### Custom Sampling Rates

```python
viewer = Viewer(
    dataset_type='haskins',
    wav_sr=48000,  # Custom audio sampling rate
    ema_sr=200     # Custom EMA sampling rate
)
```

#### Auto-detect Metadata

```python
# Use this when you're unsure about field/channel names
viewer = Viewer(ignore_meta=True)
viewer.load('data.mat')
viewer.mat2py()

# Check what was detected
print(viewer.field_names)
print(viewer.channel_names)
```

#### Legacy Parameter Support

```python
# Old style (deprecated but still works)
viewer = Viewer(
    field_names=['NAME', 'SRATE', 'SIGNAL'],
    channel_names=['TR', 'TB'],
    audio_channel='AUDIO'
)
```

### 2. Working with Annotations

#### Update from TextGrid

```python
from mviewer import Viewer

# Load data
viewer = Viewer()
viewer.load('data.pkl')

# Update annotations from corrected TextGrid
viewer.data = Viewer.update_meta(
    viewer.data,
    'corrected.TextGrid',
    phn_tier='phone',
    wrd_tier='word'
)

# Save updated data
import pickle
with open('updated.pkl', 'wb') as f:
    pickle.dump(viewer.data, f)
```

#### Update Audio Signal

```python
# Load data
viewer = Viewer()
viewer.load('data.pkl')

# Update audio from processed WAV file
viewer.data = Viewer.update_audio(
    viewer.data,
    'processed_audio.wav'
)
```

### 3. Accessing Data Structure

#### Understanding the Data Dictionary

```python
viewer = Viewer()
viewer.load('data.mat')
data = viewer.mat2py()

# Top-level keys are channel names
print(data.keys())
# Output: dict_keys(['AUDIO', 'TR', 'TB', 'TT', 'UL', 'LL', 'JAW'])

# Each channel has the same field structure
channel = data['AUDIO']
print(channel.keys())
# Output: dict_keys(['NAME', 'SRATE', 'SIGNAL', 'SOURCE',
#                    'SENTENCE', 'WORDS', 'PHONES', 'LABELS'])

# Access specific data
audio_signal = data['AUDIO']['SIGNAL']  # numpy array
audio_sr = data['AUDIO']['SRATE']       # float (44100.0)
sentence = data['AUDIO']['SENTENCE']    # string
tr_signal = data['TR']['SIGNAL']        # numpy array (N, 3) - x,y,z coordinates
```

#### Data Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `NAME` | str | Channel identifier |
| `SRATE` | float | Sampling rate in Hz |
| `SIGNAL` | ndarray | Time-series data (1D for audio, Nx3 for EMA) |
| `SOURCE` | str | Original file identifier |
| `SENTENCE` | str | Utterance text (AUDIO channel only) |
| `WORDS` | dict | Word labels and offsets (AUDIO channel only) |
| `PHONES` | dict | Phone labels and offsets (AUDIO channel only) |
| `LABELS` | dict | Generic labels with timing (AUDIO channel only) |

### 4. Customizing Visualizations

#### Multi-panel Plots

```python
viewer = Viewer()
viewer.load('data.mat')
viewer.mat2py()

# Create custom plot
fig, axs = viewer.plot(
    channel_list=['AUDIO', 'TR', 'TB', 'TT', 'JAW', 'UL', 'LL'],
    coordinates=['x', 'z'],  # Show x and z dimensions
    show=False
)

# Customize further
fig.suptitle('My Custom Title', fontsize=14)
axs[0].set_ylabel('Amplitude')
plt.savefig('custom_plot.pdf', dpi=300)
```

#### Animation Parameters

```python
# The animation uses these default parameters:
# - fps = audio_sr / 1000 (≈44 fps for 44100 Hz audio)
# - Video codec: copy (ffmpeg)
# - Audio codec: copy (ffmpeg)
# - Output format: .mov

viewer.animate(
    'output.mov',
    channel_list=['AUDIO', 'TR', 'TB', 'TT'],
    coordinates=['x', 'z']
)
```

---

## API Reference

### Viewer Class

#### Constructor

```python
Viewer(
    dataset_type='haskins',
    field_names=None,      # Deprecated
    channel_names=None,    # Deprecated
    audio_channel=None,    # Deprecated
    ignore_meta=False,
    wav_sr=None,
    ema_sr=None
)
```

**Parameters:**
- `dataset_type` (str): Dataset type ('haskins', 'mngu0', 'mocha', 'xrmb')
- `ignore_meta` (bool): Auto-detect metadata if True
- `wav_sr` (float|None): Audio sampling rate (None = use dataset default)
- `ema_sr` (float|None): EMA sampling rate (None = use dataset default)

**Legacy Parameters (Deprecated):**
- `field_names` (list|None): Override default field names
- `channel_names` (list|None): Override default channel names
- `audio_channel` (str|None): Override default audio channel name

#### Methods

##### load(file_name)

Load data from `.mat` or `.pkl` file.

```python
viewer.load('path/to/file.mat')
```

**Parameters:**
- `file_name` (str): Path to file

**Raises:**
- `AssertionError`: If file doesn't exist
- `ValueError`: If file extension not supported

---

##### mat2py(data=None, save_file=None)

Convert MATLAB format to Python dictionary.

```python
# Convert and return
data = viewer.mat2py()

# Convert and save
viewer.mat2py(save_file='output.pkl')
```

**Parameters:**
- `data` (dict|None): Custom data to convert (None = use loaded data)
- `save_file` (str|None): Path to save as .pkl

**Returns:**
- `dict` or `None`: Converted data (None if save_file specified)

**Raises:**
- `AssertionError`: If no .mat file loaded

---

##### py2mat(file_name, data, struct_var_name='data')

Convert Python dictionary to MATLAB `.mat` file.

```python
viewer.py2mat('output.mat', viewer.data, struct_var_name='F01_B01_S01_R01_N')
```

**Parameters:**
- `file_name` (str): Output file path
- `data` (dict): Data dictionary to convert
- `struct_var_name` (str): MATLAB struct variable name

**Returns:**
- MATLAB struct object

---

##### plot(channel_list, coordinates=['x','z'], show=True, file_name=None)

Visualize signals.

```python
fig, axs = viewer.plot(
    channel_list=['AUDIO', 'TR', 'TB'],
    coordinates=['x', 'z'],
    show=True,
    file_name='plot.png'
)
```

**Parameters:**
- `channel_list` (list): Channels to plot
- `coordinates` (list): Coordinates to show for EMA channels ('x', 'y', 'z')
- `show` (bool): Display plot immediately
- `file_name` (str|None): Save plot to file

**Returns:**
- `(fig, axs)`: Matplotlib figure and axes objects

**Requires:**
- Must call `load()` and `mat2py()` first

---

##### animate(file_name, channel_list, coordinates=['x','z'])

Create synchronized animation with audio.

```python
viewer.animate(
    'animation.mov',
    channel_list=['AUDIO', 'TR', 'TB', 'TT']
)
```

**Parameters:**
- `file_name` (str): Output file path (.mov)
- `channel_list` (list): Channels to include
- `coordinates` (list): Coordinates to show ('x', 'y', 'z')

**Requires:**
- ffmpeg installed and in PATH
- Must call `load()` and `mat2py()` first

**Raises:**
- `RuntimeError`: If ffmpeg not found

---

##### update_meta(dictionary, tgd_file, phn_tier='phone', wrd_tier='word', ...)

Static method to update annotations from TextGrid.

```python
updated_data = Viewer.update_meta(
    data_dict,
    'annotations.TextGrid',
    phn_tier='phone',
    wrd_tier='word'
)
```

**Parameters:**
- `dictionary` (dict): Data dictionary to update
- `tgd_file` (str): Path to TextGrid file
- `phn_tier` (str): Phone tier name
- `wrd_tier` (str): Word tier name
- `field_names` (list): Field names for validation
- `channel_names` (list): Channel names for validation
- `audio_channel` (str): Audio channel name

**Returns:**
- `dict`: Updated dictionary

---

##### update_audio(dictionary, file_name, ...)

Static method to update audio signal from WAV file.

```python
updated_data = Viewer.update_audio(
    data_dict,
    'new_audio.wav'
)
```

**Parameters:**
- `dictionary` (dict): Data dictionary to update
- `file_name` (str): Path to WAV file
- `field_names` (list): Field names for validation
- `channel_names` (list): Channel names for validation
- `audio_channel` (str): Audio channel name

**Returns:**
- `dict`: Updated dictionary

**Raises:**
- `AssertionError`: If audio length or sampling rate doesn't match

---

## Extending for New Datasets

### Creating a New Adapter

To add support for a new EMA dataset (e.g., mngu0):

#### Step 1: Define Adapter Class

```python
class Mngu0Adapter(DatasetAdapter):
    """Adapter for Edinburgh mngu0 dataset"""

    DATASET_NAME = "mngu0"
    SUPPORTED_EXTENSIONS = ['.ema']

    def __init__(self, ignore_meta=False):
        self.ignore_meta = ignore_meta
        self._field_names = ['NAME', 'SRATE', 'SIGNAL']
        self._channel_names = ['T1', 'T2', 'T3', 'T4', 'UL', 'LL', 'LI']
        self._audio_channel = 'AUDIO'

    @property
    def channel_names(self) -> list:
        return self._channel_names

    @property
    def field_names(self) -> list:
        return self._field_names

    @property
    def audio_channel(self) -> str:
        return self._audio_channel

    @property
    def default_ema_srate(self) -> float:
        return 200.0  # mngu0 uses 200 Hz

    @property
    def default_audio_srate(self) -> float:
        return 16000.0  # mngu0 uses 16 kHz audio

    def load_file(self, file_path: str) -> dict:
        """Load mngu0 .ema file"""
        # TODO: Implement binary .ema file reader
        # Read header, channel info, data blocks
        import struct
        with open(file_path, 'rb') as f:
            # Parse mngu0 format...
            pass

    def parse_to_standard_format(self, raw_data) -> dict:
        """Convert mngu0 format to standard dictionary"""
        # TODO: Convert to standardized format
        # Return dict with AUDIO, T1, T2, etc. as keys
        pass
```

#### Step 2: Register Adapter

Edit `Viewer._get_adapter()` method:

```python
def _get_adapter(self, dataset_type: str, ignore_meta: bool):
    adapters = {
        'haskins': HaskinsAdapter,
        'mngu0': Mngu0Adapter,     # Add this line
        # Future: 'mocha': MochaAdapter,
        # Future: 'xrmb': XrmbAdapter,
    }
    # ... rest of method
```

#### Step 3: Use New Adapter

```python
viewer = Viewer(dataset_type='mngu0')
viewer.load('mngu0_data.ema')
viewer.mat2py(save_file='mngu0_data.pkl')
```

### Adapter Interface Requirements

All adapters must implement:

1. **Class Attributes:**
   - `DATASET_NAME` (str): Human-readable name
   - `SUPPORTED_EXTENSIONS` (list): File extensions

2. **Methods:**
   - `load_file(file_path)`: Load raw file
   - `parse_to_standard_format(raw_data)`: Convert to standard dict

3. **Properties:**
   - `channel_names`: List of EMA channel names
   - `field_names`: List of field names
   - `default_ema_srate`: Default EMA sampling rate (optional, default 100.0)
   - `default_audio_srate`: Default audio sampling rate (optional, default 44100.0)

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:** `ImportError: No module named 'mviewer'`

**Solution:**
```bash
# Make sure you're in the project root
cd Python-EMA-Viewer

# Activate environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

---

#### 2. MATLAB File Loading Fails

**Problem:** `AssertionError: loaded .mat file is empty`

**Solution:**
- Ensure the .mat file contains a struct variable with the same name as the filename
- Try using `ignore_meta=True` to auto-detect structure

```python
viewer = Viewer(ignore_meta=True)
viewer.load('data.mat')
```

---

#### 3. Animation Fails

**Problem:** `RuntimeError: ffmpeg is not installed`

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Verify installation
which ffmpeg
```

---

#### 4. Pickle Load Error

**Problem:** `AssertionError: pickle data is not python-dictionary`

**Solution:**
- The .pkl file may be corrupted or in wrong format
- Regenerate from .mat file:

```python
viewer = Viewer()
viewer.load('original.mat')
viewer.mat2py(save_file='regenerated.pkl')
```

---

#### 5. Channel Name Mismatch

**Problem:** `AssertionError: channel_names are not matching`

**Solution:**
```python
# Option 1: Use ignore_meta to auto-detect
viewer = Viewer(ignore_meta=True)

# Option 2: Specify custom channel names
viewer = Viewer(
    channel_names=['TR', 'TB', 'TT'],  # Your actual channels
    audio_channel='AUDIO'
)
```

---

## Examples

### Example 1: Basic Workflow

```python
from mviewer import Viewer

# Load and convert
viewer = Viewer()
viewer.load('example/F01_B01_S01_R01_N.mat')
viewer.mat2py(save_file='output.pkl')

# Visualize
viewer.plot(
    channel_list=['AUDIO', 'TR', 'TB', 'TT'],
    show=True,
    file_name='visualization.png'
)

print("Sentence:", viewer.data['AUDIO']['SENTENCE'])
```

---

### Example 2: Batch Processing

```python
from mviewer import Viewer
import glob
import os

# Process all .mat files in a directory
mat_files = glob.glob('data/*.mat')

for mat_file in mat_files:
    print(f"Processing {mat_file}...")

    viewer = Viewer()
    viewer.load(mat_file)

    # Save as .pkl
    base_name = os.path.splitext(mat_file)[0]
    viewer.mat2py(save_file=f'{base_name}.pkl')

    # Create plot
    viewer.plot(
        channel_list=['AUDIO', 'TR', 'TB'],
        show=False,
        file_name=f'{base_name}.png'
    )

    print(f"  Saved: {base_name}.pkl and {base_name}.png")
```

---

### Example 3: Extracting Specific Data

```python
from mviewer import Viewer
import numpy as np

viewer = Viewer()
viewer.load('data.pkl')

# Extract audio segment
audio_signal = viewer.data['AUDIO']['SIGNAL']
audio_sr = viewer.data['AUDIO']['SRATE']

# Get first 1 second
one_second = int(audio_sr)
audio_segment = audio_signal[:one_second]

# Extract tongue tip (TT) trajectory
tt_signal = viewer.data['TT']['SIGNAL']  # Shape: (N, 3)
tt_x = tt_signal[:, 0]  # X coordinate
tt_y = tt_signal[:, 1]  # Y coordinate
tt_z = tt_signal[:, 2]  # Z coordinate

print(f"Audio length: {len(audio_signal) / audio_sr:.2f} seconds")
print(f"TT trajectory points: {len(tt_signal)}")
```

---

### Example 4: Updating Annotations

```python
from mviewer import Viewer

# Load data
viewer = Viewer()
viewer.load('original.pkl')

# Update from corrected TextGrid
viewer.data = Viewer.update_meta(
    viewer.data,
    'corrected_annotations.TextGrid',
    phn_tier='phone',
    wrd_tier='word'
)

# Check updated annotations
words = viewer.data['AUDIO']['WORDS']
print("Word labels:", words['LABEL'])
print("Word times:", words['OFFS'])

# Save updated data
import pickle
with open('updated.pkl', 'wb') as f:
    pickle.dump(viewer.data, f)
```

---

### Example 5: Custom Dataset Adapter (Template)

```python
from mviewer import DatasetAdapter, Viewer

class CustomAdapter(DatasetAdapter):
    """Adapter for custom EMA dataset"""

    DATASET_NAME = "Custom Dataset"
    SUPPORTED_EXTENSIONS = ['.custom']

    def __init__(self, ignore_meta=False):
        self.ignore_meta = ignore_meta
        self._field_names = ['NAME', 'SRATE', 'SIGNAL']
        self._channel_names = ['CH1', 'CH2', 'CH3']
        self._audio_channel = 'AUDIO'

    @property
    def channel_names(self):
        return self._channel_names

    @property
    def field_names(self):
        return self._field_names

    def load_file(self, file_path):
        # TODO: Implement custom file loader
        pass

    def parse_to_standard_format(self, raw_data):
        # TODO: Convert to standard format
        # Must return dict with structure:
        # {
        #   'AUDIO': {'NAME': ..., 'SRATE': ..., 'SIGNAL': ...},
        #   'CH1': {'NAME': ..., 'SRATE': ..., 'SIGNAL': ...},
        #   ...
        # }
        pass

# Register and use
# (Add to Viewer._get_adapter() first)
viewer = Viewer(dataset_type='custom')
viewer.load('data.custom')
```

---

## Appendix

### A. Data Structure Specification

```python
{
    'AUDIO': {
        'NAME': str,              # 'audio'
        'SRATE': float,           # 44100.0
        'SIGNAL': ndarray,        # (N,) audio samples
        'SOURCE': str,            # File identifier
        'SENTENCE': str,          # Utterance text
        'WORDS': {
            'LABEL': list,        # ['the', 'cat', ...]
            'OFFS': ndarray       # [[0.0, 0.3], [0.3, 0.6], ...]
        },
        'PHONES': {
            'LABEL': list,        # ['dh', 'ax', 'k', ...]
            'OFFS': ndarray       # [[0.0, 0.1], [0.1, 0.2], ...]
        },
        'LABELS': {
            'NAME': list,         # Label names
            'OFFSET': ndarray,    # Start times (ms)
            'VALUE': ndarray,     # Durations (ms)
            'HOOK': ndarray       # Additional data
        }
    },
    'TR': {  # Tongue Right
        'NAME': str,              # 'tr'
        'SRATE': float,           # 100.0
        'SIGNAL': ndarray,        # (N, 3) - x, y, z coordinates
        'SOURCE': str or [],
        'SENTENCE': str or [],
        'WORDS': list,            # Empty for non-audio channels
        'PHONES': list,           # Empty for non-audio channels
        'LABELS': list            # Empty for non-audio channels
    },
    # ... similar structure for TB, TT, UL, LL, JAW
}
```

### B. Supported Datasets

| Dataset | Status | Adapter | Channels | Audio SR | EMA SR |
|---------|--------|---------|----------|----------|--------|
| Haskins IEEE | ✅ Available | HaskinsAdapter | TR,TB,TT,UL,LL,JAW | 44100 Hz | 100 Hz |
| mngu0 | 🔜 Planned | - | T1-T4,UL,LL,LI | 16000 Hz | 200 Hz |
| MOCHA-TIMIT | 🔜 Planned | - | Various | 16000 Hz | 500 Hz |
| XRMB | 🔜 Planned | - | Pellets | 22050 Hz | 145 Hz |

### C. Channel Naming Conventions

**Haskins IEEE:**
- `TR`: Tongue Right
- `TB`: Tongue Back
- `TT`: Tongue Tip
- `UL`: Upper Lip
- `LL`: Lower Lip
- `JAW`: Jaw
- `AUDIO`: Audio channel

### D. Performance Notes

- **Loading**: `.pkl` files load ~10x faster than `.mat` files
- **Memory**: Keep both `data_orig` and `data` in memory (planned optimization)
- **Animation**: Generation time ≈ 2x playback duration
- **File Sizes**: `.pkl` typically larger than `.mat` (Python overhead)

---

## Support

For issues, questions, or contributions:
- **Documentation**: See CLAUDE.md and README.md
- **Test Suite**: Run `python test/run_tests.py`
- **Examples**: Check `example/` directory
- **Issues**: Create issues on project repository

---

**Last Updated:** 2025-12-27
**Version:** 2.0 with Adapter Pattern
**License:** See project LICENSE file
