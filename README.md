# Python-EMA-Viewer (`MVIEWER`)

This repository includes a Python-based procedure for the following tasks:
- [x] 1) Convert Matlab MVIEW-compatible EMA (electromagnetic articulography) data (`.mat`) into pickle `.pkl` format
- [x] 2) Convert from `.pkl` back to MVIEW-compatible `.mat`.
- [x] 3) Visualize articulatory trajectories with the corresponding waveform.

Note that this procedure is based on Haskins IEEE rate comparison dataset ([Link](https://yale.app.box.com/s/cfn8hj2puveo65fq54rp1ml2mk7moj3h)). The compatibility with other EMA data format from different machines (AG500, AG501) or OS (Windows) yet. You may need to configure xx for your own.

## How to
- Convert `.mat` to `.pkl`
```python
>> from mviewer import Viewer
>> mm = Viewer()
>> mm.load('example/F01_B01_S01_F01_N.mat') # load .mat file
>> mm.mat2py(save_file='example/F01_B01_S01_F01_N.pkl') # convert .mat to .pkl & save as .mat file

```
TODO: Add a screenshot of mview --> python

- Convert `.pkl` to `.mat`
```python
>> f = open('example/F01_B01_S01_F01_N.pkl', 'rb') # load .pkl file
>> data = pickle.load(f); f.close()
>> from mviewer import Viewer
>> mm = Viewer()
>> mm.py2mat('example/F01_B01_S01_F01_N_new.mat', data, save=True) # convert .pkl to .mat & save as .pkl file
```

- Visualize
```python
>> mm.plot(channel_list=['AUDIO','TR', 'TB', 'TT'], show=True)
```
[<img align="left" alt="plot" width="600px" src="https://raw.githubusercontent.com/jaekookang/Python-EMA-Viewer/master/png/test.png" />]
<br />

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

# Matlab
(optional) mview

# Misc
(optional) ffmpeg
```

## TODOs
- [x] Support for IEEE
- [ ] Fix `.animate()` method
- [ ] Support for XRMB
- [ ] Support for mngu0
- [ ] Support for MOCHA-TIMIT

## Acknowledgements
- The example files in `example` folder (i.e., `F01_B01_S01_R01_N.mat` and `M01_B01_S01_R01_N.mat`) were retrieved from the original data repository ([Link](https://yale.app.box.com/s/cfn8hj2puveo65fq54rp1ml2mk7moj3h)) without any modifications only for demonstration (version 3 of the GNU General Public License).

## History
- 2020-12-31: first created