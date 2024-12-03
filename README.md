# Python-EMA-Viewer (`MVIEWER`)

This repository includes a Python-based procedure for the following tasks:
- [x] 1) Convert Matlab MVIEW-compatible EMA (electromagnetic articulography) data (`.mat`) into pickle `.pkl` format
- [x] 2) Convert from `.pkl` back to MVIEW-compatible `.mat`.
- [x] 3) Visualize articulatory trajectories with the corresponding waveform.

<br>

Note that this procedure is based on Haskins IEEE rate comparison dataset ([Link](https://yale.app.box.com/s/cfn8hj2puveo65fq54rp1ml2mk7moj3h)). The compatibility with other EMA data format from different machines (AG500, AG501) or OS (Windows) has not been test yet. You may need to configure `mviewer.py` for your own.

<br>

## How to
- (1) Convert `.mat` to `.pkl`
```python
from mviewer import Viewer
mm = Viewer()
mm.load('example/F01_B01_S01_F01_N.mat') # load .mat file
mm.mat2py(save_file='example/F01_B01_S01_F01_N.pkl') # convert .mat to .pkl & save as .mat file

pprint(mm.data)

	{'AUDIO': {...},
	 'JAW': {...},
	 'JAWL': {...},
	 'LL': {...},
	 'ML': {...},
	 'TB': {...},
	 'TR': {...},
	 'TT': {...},
	 'UL': {...}}
```

- (2) Convert `.pkl` to `.mat`
```python
f = open('example/F01_B01_S01_F01_N.pkl', 'rb') # load .pkl file
data = pickle.load(f); f.close()
from mviewer import Viewer
mm = Viewer()
mm.py2mat('example/F01_B01_S01_F01_N_new.mat', data, save=True) # convert .pkl to .mat & save as .pkl file
```

- (3) Visualize
```python
from mviewer import Viewer
mm = Viewer()
mm.load('example/F01_B01_S01_F01_N.mat') # load .mat file
mm.mat2py(save_file='example/F01_B01_S01_F01_N.pkl') # convert .mat to .pkl & save as .mat file
mm.plot(channel_list=['AUDIO','TR', 'TB', 'TT'], show=True)
```

<img alt="plot" width="600px" src="https://raw.githubusercontent.com/jaekookang/Python-EMA-Viewer/master/result/F01_B01_S01_R01_N.png" />
<br><br>

- (4) Update meta information (phone/word labels and time info in TextGrids)
```python
from mviewer import Viewer
mm = Viewer()
mm.load('example/F01_B01_S01_F01_N.mat') # load .mat file
dictionary = mm.mat2py() # outputs dictionary instead of saving it
dictionary = mm.update_meta(dictionary, 'example/F01_B01_S01_F01_N.TextGrid')
# => label information updated!
```

- (5) Update audio information (when you modified your audio signal; eg. amplitude normalization)
```python
from mviewer import Viewer
mm = Viewer()
mm.load('example/F01_B01_S01_F01_N.mat') # load .mat file
dictionary = mm.mat2py() # outputs dictionary instead of saving it
dictionary = mm.update_audio(dictionary, 'example/F01_B01_S01_F01_N.wav')
# => AUDIO information updated!
```

- If you want to specify meta information (channel names or field names in struct), you can do so (See `class Viewer` in `mviwer.py`)
- If you are now sure about the meta information, you can specify `ignore_meta=True` when intiaiting `class Viewer`.


## Data structure
The structure of the re-organized python dictionary including the EMA and AUDIO data looks like following:

```python
{'AUDIO': {
	'NAME': {...},
	'SRATE': {...},
	'SIGNAL': {...},
	'SOURCE': {...},   # eg. S07_sen01_HS01_B01_R01_0004_01
	'SENTENCE': {...}, # eg. The birch canoe slid on the smooth planks.
	'WORDS': {...},    # eg. {'LABEL': ..., 'OFFS': ... }
	'PHONES': {...},   # eg. {'LABEL': ..., 'OFFS': ... }
	'LABELS': {...},   # eg. {'NAME': ..., 'OFFSET': ..., 'VALUE': ..., 'HOOK': ... }
},
 'JAW': {
 	'NAME': {...},
	'SRATE': {...},
	'SIGNAL': {...},
	'SOURCE': [],
	'SENTENCE': [],
	'WORDS': [],
	'PHONES': [],
	'LABELS': [],
 },
 'JAWL': {
 	'NAME': {...},
	'SRATE': {...},
	'SIGNAL': {...},
	'SOURCE': [],
	'SENTENCE': [],
	'WORDS': [],
	'PHONES': [],
	'LABELS': [],
 },
 'LL': {
 	'NAME': {...},
	'SRATE': {...},
	'SIGNAL': {...},
	'SOURCE': [],
	'SENTENCE': [],
	'WORDS': [],
	'PHONES': [],
	'LABELS': [],
 },
 'TB': {
 	'NAME': {...},
	'SRATE': {...},
	'SIGNAL': {...},
	'SOURCE': [],
	'SENTENCE': [],
	'WORDS': [],
	'PHONES': [],
	'LABELS': [],
 },
 'TR': {
 	'NAME': {...},
	'SRATE': {...},
	'SIGNAL': {...},
	'SOURCE': [],
	'SENTENCE': [],
	'WORDS': [],
	'PHONES': [],
	'LABELS': [],
 },
 'TT': {
 	'NAME': {...},
	'SRATE': {...},
	'SIGNAL': {...},
	'SOURCE': [],
	'SENTENCE': [],
	'WORDS': [],
	'PHONES': [],
	'LABELS': [],
 },
 'UL': {
 	'NAME': {...},
	'SRATE': {...},
	'SIGNAL': {...},
	'SOURCE': [],
	'SENTENCE': [],
	'WORDS': [],
	'PHONES': [],
	'LABELS': [],
 }}
```



## Requirements
- This procedure was tested on macOS (xx) as of 2020-12-31
```bash
# Data
Haskins IEEE dataset

# Python
python==3.7.4
numpy==1.18.5
scipy==1.4.1
matplotlib==3.3.3
seaborn==0.11.0
tgt==1.4.4

# Matlab
(optional) mview (developed by Mark Tiede @ Haskins Labs)

# Misc
(optional) ffmpeg
```

## TODOs
- [x] Support for IEEE
- [x] Fix `.animate()` method
- [ ] Support for XRMB
- [ ] Support for mngu0
- [ ] Support for MOCHA-TIMIT
- [ ] Provide support for data compression (bz2)

## Acknowledgements
- The example files in `example` folder (i.e., `F01_B01_S01_R01_N.mat` and `M01_B01_S01_R01_N.mat`) were retrieved from the original data repository ([Link](https://yale.app.box.com/s/cfn8hj2puveo65fq54rp1ml2mk7moj3h)) without any modifications only for demonstration (version 3 of the GNU General Public License).

## History
- 2020-12-31: first created
- 2024-12-03: dev branch created