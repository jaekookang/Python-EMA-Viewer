'''
mviewer - Multi-dataset EMA Viewer

Supports multiple EMA datasets through adapter pattern:
- Haskins IEEE Rate Comparison (default)
- mngu0 (future)
- MOCHA-TIMIT (future)
- XRMB (future)

Architecture:
- DatasetAdapter (ABC): Interface for dataset loaders
- HaskinsAdapter: Implementation for Haskins dataset
- Viewer: Main API (uses adapters internally)

Usage:
    # Default (Haskins)
    viewer = Viewer()
    viewer.load('file.mat')

    # Explicit dataset type
    viewer = Viewer(dataset_type='haskins')

    # Future: other datasets
    viewer = Viewer(dataset_type='mngu0')

2020-12-31 first created
2025-12-27 refactored with adapter pattern
'''
import os
import subprocess
import tgt
import pickle
import numpy as np
from abc import ABC, abstractmethod
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile

def load_mat(file_name):
    '''Load .mat file'''
    fname, ext = os.path.splitext(file_name)
    fid = os.path.basename(fname)
    mat = loadmat(file_name)[fid][0]
    assert len(mat) > 0, 'loaded .mat file is empty'
    return mat


def load_pkl(file_name):
    '''Load .pkl file'''
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    assert isinstance(data, dict), 'pickle data is not python-dictionary'
    assert len(data.keys()) > 0, 'there is no valid keys in data'
    return data


def load_textgrid(file_name, tier_name='phone'):
    '''Load textgrid & return times and labels'''
    tg = tgt.read_textgrid(file_name)
    tier = tg.get_tier_by_name(tier_name)

    times = []
    labels = []
    for t in tier:
        times.append([round(t.start_time,4), round(t.end_time,4)])
        labels.append(t.text)
    assert len(times) > 0, f'"times" is empty: len={len(times)}'
    assert len(labels) > 0, f'"{tier_name}" is empty: len={len(labels)}'
    return np.array(times, dtype='float32'), labels    


def check_dictionary(dt, field_names, channel_names, audio_channel):
    '''Check dictionary structure'''
    channels = [audio_channel]+channel_names
    assert set(dt.keys()) == set([audio_channel]+channel_names), \
        f'channel_names are not matching\n   Target{channels} <==>\nProvided:{dt.keys()}'
    for key in dt.keys():
        keys = dt[key].keys()
        assert set(keys) == set(field_names), \
            f'field_names are not matching at key={key}\n   Target{field_names} <==>\nProvided:{keys}'


def get_struct(mat, field_names=None, channel_names=None, audio_channel=None, ignore_meta=False):
    '''Get the structure from the loaded mat file
    ** From .mat to .pkl **
    - dictionary will be created with keys based on channel_names
      subkeys will be the field_names
    '''
    # Check names
    _field_names = list(mat.dtype.fields.keys())
    _channel_names = [m[0][0] for m in mat]
    if ignore_meta:
        field_names = _field_names
        channel_names = _channel_names
    else:
        assert set(field_names) & set(_field_names) == set(field_names), \
             f'Given field_names does not match with data\nGiven:{field_names} <==>\n Data:{_field_names}'
        assert set([audio_channel]+channel_names) & set(_channel_names) == set([audio_channel]+channel_names), \
             f'Given channel_names does not match with data\nGiven:{[audio_channel]+channel_names} <==>\n Data:{_channel_names}'
        channel_names = [audio_channel] + channel_names
    
    # Reorganize mat based on channel_names
    # to prevent unwanted sensors to added (eg. ML)
    mat = [m for m in mat if m[0][0] in channel_names]
    assert len(mat) == len(channel_names), f'no. of channel names do not match with the data'

    # Build dictionary
    mat_dict = {ch:'' for ch in channel_names}
    for ch, m in zip(channel_names, mat):
        fields = {}
        for fd in field_names:
            i = _field_names.index(fd)
            data = m[i]
            if fd in ['SRATE']:
                data = float(data[0][0])
            elif fd in ['SIGNAL']:
                data = data.astype('float32')
            elif fd in ['NAME', 'SENTENCE', 'SOURCE']:
                data = str(data[0]) if isinstance(data[0], np.str_) else []
            elif fd in ['WORDS','PHONES']:
                if len(data[0]) > 0:
                    LABEL = [d[0][0] for d in data[0]]
                    OFFS = np.array([d[1][0] for d in data[0]], dtype='float32')
                else:
                    LABEL, OFFS = [], []
                data = {'LABEL': LABEL, 'OFFS': OFFS}
            elif fd in ['LABELS']:
                if len(data[0]) > 0:
                    NAME = [d[0][0] for d in data[0]]
                    OFFSET = np.array([d[1][0] for d in data[0]], dtype='float32')
                    VALUE = np.array([d[2][0] for d in data[0]], dtype='float32')
                    HOOK = np.array([d[3] for d in data[0]])
                else:
                    NAME = []
                    OFFSET = []
                    VALUE = []
                    HOOK = []
                data = {'NAME': NAME, 'OFFSET': OFFSET, 'VALUE':VALUE, 'HOOK':HOOK}
            else:
                data = []
            fields[fd] = data
        # Update
        mat_dict.update({ch: fields})

    return mat_dict


class DatasetAdapter(ABC):
    """Abstract base class for EMA dataset adapters

    This defines the interface that all dataset adapters must implement.
    Each adapter encapsulates the logic for loading and parsing a specific
    EMA dataset format (Haskins, mngu0, MOCHA-TIMIT, XRMB, etc.).
    """

    DATASET_NAME: str = "Unknown"
    SUPPORTED_EXTENSIONS: list = []

    @abstractmethod
    def load_file(self, file_path: str) -> dict:
        """Load raw file and return dataset-specific structure

        Args:
            file_path: Path to the file to load

        Returns:
            Raw data structure from the file
        """
        pass

    @abstractmethod
    def parse_to_standard_format(self, raw_data) -> dict:
        """Convert dataset format to standardized internal format

        Args:
            raw_data: Raw data from load_file()

        Returns:
            Standardized dictionary with channel_names as keys
        """
        pass

    @property
    @abstractmethod
    def channel_names(self) -> list:
        """Return list of EMA channel names for this dataset"""
        pass

    @property
    @abstractmethod
    def field_names(self) -> list:
        """Return list of field names in data structure"""
        pass

    @property
    def default_ema_srate(self) -> float:
        """Default EMA sampling rate (can be overridden)"""
        return 100.0

    @property
    def default_audio_srate(self) -> float:
        """Default audio sampling rate (can be overridden)"""
        return 44100.0


class HaskinsAdapter(DatasetAdapter):
    """Adapter for Haskins IEEE Rate Comparison Dataset

    This dataset uses MATLAB .mat files with specific struct format.
    Channels: TR, TB, TT, UL, LL, JAW (tongue, lips, jaw sensors)
    Audio: 44.1 kHz, EMA: 100 Hz
    """

    DATASET_NAME = "Haskins IEEE"
    SUPPORTED_EXTENSIONS = ['.mat', '.MAT']

    def __init__(self, ignore_meta=False):
        """Initialize Haskins adapter

        Args:
            ignore_meta: If True, auto-detect field/channel names from file
        """
        self.ignore_meta = ignore_meta
        self._field_names = ['NAME', 'SRATE', 'SIGNAL', 'SOURCE',
                             'SENTENCE', 'WORDS', 'PHONES', 'LABELS']
        self._channel_names = ['TR', 'TB', 'TT', 'UL', 'LL', 'JAW']
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

    def load_file(self, file_path: str) -> dict:
        """Load Haskins .mat file"""
        return load_mat(file_path)

    def parse_to_standard_format(self, raw_data) -> dict:
        """Convert Haskins MATLAB format to standard dictionary"""
        return get_struct(
            raw_data,
            field_names=self._field_names,
            channel_names=self._channel_names,
            audio_channel=self._audio_channel,
            ignore_meta=self.ignore_meta
        )


class Viewer:
    def __init__(self,
                 dataset_type='haskins',
                 field_names=None,
                 channel_names=None,
                 audio_channel=None,
                 ignore_meta=False,
                 wav_sr=None,
                 ema_sr=None):
        '''
        Initialize Viewer with dataset adapter

        Parameters:
        -----------
        dataset_type: str
            Dataset type: 'haskins', 'mngu0', 'mocha', 'xrmb', or 'auto'
            Default: 'haskins'
        ignore_meta: bool
            If True, auto-detect metadata from files
        wav_sr, ema_sr: float or None
            Sampling rates (None = use dataset defaults)

        Backward Compatibility (Legacy Parameters):
        ------------------------------------------
        field_names: list or None (deprecated)
            Name of the fields defined in the `.mat` file
            If provided, assumes Haskins dataset and overrides defaults
        channel_names: list or None (deprecated)
            Name of the EMA sensors on the articulators except for AUDIO
            If provided, assumes Haskins dataset and overrides defaults
        audio_channel: str or None (deprecated)
            Name of the audio channel name
            If provided, assumes Haskins dataset and overrides defaults

        Note: Legacy parameters are kept for backward compatibility.
        New code should use dataset_type parameter instead.
        '''
        # Handle legacy parameters - if any legacy param is provided, use Haskins
        if field_names is not None or channel_names is not None or audio_channel is not None:
            dataset_type = 'haskins'

        # Initialize adapter
        if dataset_type == 'haskins':
            self.adapter = HaskinsAdapter(ignore_meta=ignore_meta)

            # Override with legacy parameters if provided
            if field_names is not None:
                self.adapter._field_names = field_names
            if channel_names is not None:
                self.adapter._channel_names = channel_names
            if audio_channel is not None:
                self.adapter._audio_channel = audio_channel
        else:
            # Future: auto-detect or other datasets
            self.adapter = self._get_adapter(dataset_type, ignore_meta)

        # Use adapter defaults or override with user values
        self.wav_sr = wav_sr if wav_sr is not None else self.adapter.default_audio_srate
        self.ema_sr = ema_sr if ema_sr is not None else self.adapter.default_ema_srate
        self.ignore_meta = ignore_meta

        # State variables
        self.data_orig = None
        self.data = None
        self.loaded_data_type = None
        self.file_name = None

        # Legacy properties for backward compatibility
        self.field_names = self.adapter.field_names
        self.channel_names = self.adapter.channel_names
        if hasattr(self.adapter, 'audio_channel'):
            self.audio_channel = self.adapter.audio_channel

    def _get_adapter(self, dataset_type: str, ignore_meta: bool):
        """Factory method for dataset adapter selection

        Args:
            dataset_type: Type of dataset ('haskins', 'mngu0', 'mocha', 'xrmb')
            ignore_meta: Whether to auto-detect metadata

        Returns:
            DatasetAdapter instance for the specified dataset type

        Raises:
            ValueError: If dataset_type is not supported
        """
        adapters = {
            'haskins': HaskinsAdapter,
            # Future: 'mngu0': Mngu0Adapter,
            # Future: 'mocha': MochaAdapter,
            # Future: 'xrmb': XrmbAdapter,
        }

        if dataset_type not in adapters:
            raise ValueError(
                f"Unknown dataset type: {dataset_type}. "
                f"Supported: {list(adapters.keys())}"
            )

        return adapters[dataset_type](ignore_meta=ignore_meta)

    def load(self, file_name=None):
        '''Load files using dataset adapter

        Supports .mat files (via adapter) and .pkl files (direct load).
        For .mat files, the adapter handles dataset-specific parsing.

        Args:
            file_name: Path to file (.mat or .pkl)

        Raises:
            AssertionError: If file doesn't exist or is not a file
            ValueError: If file extension not supported
        '''
        self.file_name = file_name
        assert os.path.isfile(file_name), f'File does not exist or is not a file: {file_name}'

        fname, ext = os.path.splitext(file_name)

        # Check if adapter supports this extension for .mat files
        if ext in ['.mat', '.MAT']:
            if ext not in self.adapter.SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f'{ext} not supported by {self.adapter.DATASET_NAME} adapter. '
                    f'Supported: {self.adapter.SUPPORTED_EXTENSIONS}'
                )
            self.loaded_data_type = 'mat'
            data = self.adapter.load_file(file_name)
            self.data_orig = data
            # Note: Don't parse yet - keep raw for potential mat2py() call

        elif ext in ['.pkl', '.PKL', '.pickle', '.pckl']:
            self.loaded_data_type = 'pkl'
            data = load_pkl(file_name)
            self.data_orig = data
            self.data = data

        else:
            raise ValueError('Check the file extensions. Choose either .mat or .pkl')

    @staticmethod
    def update_meta(dictionary, tgd_file, phn_tier='phone', wrd_tier='word',
                    field_names=['NAME', 'SRATE', 'SIGNAL', 'SOURCE', 'SENTENCE', 'WORDS', 'PHONES', 'LABELS'], 
                    channel_names=['TR', 'TB', 'TT','UL', 'LL', 'JAW'], 
                    audio_channel='AUDIO'):
        '''Update WORDS PHONES LABELS from the updated textgrid file
        - If you have corrected the TextGrid file, you need to run this 
          to update the dictionary
        '''
        check_dictionary(dictionary, field_names, channel_names, audio_channel)

        phn_times, phns = load_textgrid(tgd_file, tier_name='phone')
        wrd_times, wrds = load_textgrid(tgd_file, tier_name='word')

        dictionary[audio_channel]['WORDS']['LABEL'] = wrds
        dictionary[audio_channel]['WORDS']['OFFS'] = wrd_times
        dictionary[audio_channel]['PHONES']['LABEL'] = phns
        dictionary[audio_channel]['PHONES']['OFFS'] = phn_times

        dictionary[audio_channel]['LABELS']['NAME'] = wrds
        dictionary[audio_channel]['LABELS']['OFFSET'] = wrd_times[:,[0]] * 1000 # sec => msec
        dictionary[audio_channel]['LABELS']['VALUE'] = wrd_times[:,[1]] * 1000 - wrd_times[:,[0]] * 1000
        return dictionary

    @staticmethod
    def update_audio(dictionary, file_name, 
                     field_names=['NAME', 'SRATE', 'SIGNAL', 'SOURCE', 'SENTENCE', 'WORDS', 'PHONES', 'LABELS'], 
                     channel_names=['TR', 'TB', 'TT', 'UL', 'LL', 'JAW'], 
                     audio_channel='AUDIO'):
        '''Update AUDIO in the dictionary 
        - if the recorded speech signal was updated,
          for example, noise reduction or amplitude normalization might have applied,
          you need to run this
        '''
        assert os.path.exists(file_name), f'wav file {file_name} does not exist'
        check_dictionary(dictionary, field_names, channel_names, audio_channel)
        
        sr, sig = wavfile.read(file_name)
        assert len(dictionary['AUDIO']['SIGNAL']) == len(sig), \
            f'Sample length differs {len(dictionary["AUDIO"]["SIGNAL"])} != {len(sig)}'
        assert dictionary['AUDIO']['SRATE'] == sr, \
            f'Sampling rate differs {dictionary["AUDIO"]["SRATE"]} != {sr}'
        dictionary['AUDIO']['SIGNAL'] = sig

        return dictionary

    def mat2py(self, data: dict=None, save_file: str=None):
        '''Convert matlab mat to python dict using dataset adapter

        The output `data` is a dictionary with channel names as keys
        and field names as subkeys. Uses the adapter's parse method
        to handle dataset-specific conversion.

        Parameters:
        -----------
        data (opt): dict
            Dictionary object for the EMA data. If None, uses loaded data.
        save_file (opt): str
            New file name for saving as pkl

        Returns:
        --------
        dict or None
            Parsed data dictionary, or None if save_file is specified
        '''
        if data is None:
            assert self.loaded_data_type == 'mat', 'Load mat file first'
            assert self.data_orig is not None, 'load mat file first'
            # Use adapter to parse the raw data
            data = self.adapter.parse_to_standard_format(self.data_orig)
        self.data = data  # updated

        if save_file is not None:
            fname, ext = os.path.splitext(self.file_name)
            fpath = os.path.dirname(fname)
            fid = os.path.basename(fname)
            with open(save_file, 'wb') as pkl:
                pickle.dump(data, pkl)
            return None
        else:
            return data

    def py2mat(self, file_name, data, struct_var_name='data'):
        '''Convert python dict to matlab mat file
        - Given `data` (python dictionary), create a new `.mat` file

        Parameters:
        ---
        file_name: str. 
            new file name for saving as mat
        data: dict. 
            dictionary object for the EMA data
        struct_var_name: str. 
            name of the Matlab struct variable; e.g., 'F01_B01_S01_R01_N' 
            this encapsulates the struct variable 
            default name is 'data'
        
        '''
        check_dictionary(data, self.field_names, self.channel_names, self.audio_channel)
        channel_names = [self.audio_channel] + self.channel_names

        # Restructure
        rows = []
        for fd in self.field_names:
            row = []
            for ch in channel_names:
                irow = data[ch][fd]
                if fd in ['WORDS','PHONES']:
                    if len(irow['LABEL']) == 0:
                        # Empty cells
                        irow = []
                    else: 
                        # Non-empty cells
                        LABEL = [np.array([i]) for i in irow['LABEL']]
                        OFFS = [i.reshape(-1,2).astype('float') for i in irow['OFFS']]
                        irow = np.array([(l,o) for l, o in zip(LABEL, OFFS)], dtype=[('LABEL', 'O'), ('OFFS', 'O')])

                elif fd in ['LABELS']:
                    if len(irow['NAME']) == 0:
                        # Empty cells
                        irow = []
                    else:    
                        # Non-empty cells
                        NAME = [np.array([i]) for i in irow['NAME']]
                        OFFSET = [i.astype('float32') for i in irow['OFFSET']]
                        VALUE = [i.astype('float32') for i in irow['VALUE']]
                        HOOK = [i for i in irow['HOOK']]
                        irow = np.array([(name, offset, value, hook) 
                                        for name, offset, value, hook in zip(NAME, OFFSET, VALUE, HOOK)], 
                                        dtype=[('NAME', 'O'), ('OFFSET', 'float'), ('VALUE', 'float'), ('HOOK', 'O')])
                elif fd in ['SRATE']:
                    irow = float(irow)
                row += [irow]
            rows.append(row)
        mat = np.core.records.fromarrays(np.array(rows), names=self.field_names)

        # Save
        savemat(file_name, mdict={struct_var_name: mat})
        return mat

    def plot(self, channel_list=['AUDIO','TR','TB','TT','JAW','UL','LL'], 
             coordinates=['x','z'], show=True, file_name=None):
        '''Visualize signal
        Make sure to run .load() and .mat2py() first
        '''
        assert self.file_name is not None, 'Load a mat file first: .load()'
        assert self.data is not None, 'Run .mat2py()'
        assert len(channel_list) > 0, 'Specify at least one channel'
        assert set(channel_list) & set([self.audio_channel]+self.channel_names) == set(channel_list), \
            f'Some of the channel names does not exist: Total:{set([self.audio_channel]+self.channel_names)} <==> Specified:{set(channel_list)}'
        coords = {'x':0, 'y':1, 'z':2}
        n_chs = len(channel_list)
        h_fig = 1
        w_fig = 8
        with sns.plotting_context('paper', font_scale=1.2):
            fig, axs = plt.subplots(n_chs, 1, figsize=(w_fig, n_chs*h_fig), facecolor='white')
            axs = [axs] if len(channel_list) == 1 else axs
            for ch, ax in zip(channel_list, axs):
                srate = self.data[ch]['SRATE']
                # Plot
                if ch == 'AUDIO':
                    data = self.data[ch]['SIGNAL']
                    ax.plot(data,'b-', label='audio')
                else:
                    data = self.data[ch]['SIGNAL'][:, [coords[c] for c in coordinates]]
                    ax.plot(data[:,0], 'b-', label='x')
                    ax.plot(data[:,1], 'r-', label='y')
                # Prettify
                ax.set_title(ch, fontsize=10, fontweight='bold', loc='left')
                ax.set_xlim([0, len(data)])
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ticks = np.linspace(0, len(data), 10)
                ax.xaxis.set_ticks(ticks)
                if ch == channel_list[-1]:
                    ax.set_xticklabels([f'{t/srate:.2f}' for t in ticks])
                else:
                    ax.xaxis.set_ticklabels('')
        
        # Finalize
        if self.data['AUDIO']['SENTENCE'] is not None:
            fig.suptitle(self.data['AUDIO']['SENTENCE'], fontsize=12)
        fig.tight_layout()
        if show:
            plt.show()
        if file_name is not None:
            fig.savefig(file_name)
        return fig, axs

    def animate(self, file_name,
                channel_list=['AUDIO','TR','TB','TT','JAW','UL','LL'],
                coordinates=['x','z']):
        '''Make animation
        - This requires ffmpeg installed in the system. If not, it won't run.
        - File extension has to be 'mov'.
        '''
        from shutil import which
        import matplotlib.animation as animation
        if which('ffmpeg') is None:
            raise RuntimeError('ffmpeg is not installed. Install ffmpeg to generate an animation.')
        sr_wav = self.data['AUDIO']['SRATE']
        sr_ema = self.data['TT']['SRATE']
        sig = self.data['AUDIO']['SIGNAL']
        n_sig = len(sig)
        div = 1000
        fps = sr_wav/div

        tmp_wav = os.path.join(os.path.dirname(file_name), 'tmp.wav')
        tmp_mp4 = os.path.join(os.path.dirname(file_name), 'tmp.mp4')

        try:
            wavfile.write(tmp_wav, int(sr_wav), sig)

            fig, axs= self.plot(channel_list=channel_list, coordinates=coordinates, show=False)
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps)

            lines = {}
            for ch, ax in zip(channel_list, axs):
                line, = ax.plot([], [], lw=2)
                lines.update({ch: line})

            def update(i, data, lines):
                for key in lines.keys():
                    if key == 'AUDIO':
                        lines[key].set_data([i*div, i*div], [-1, 1])
                    else:
                        t = (i*div)/sr_wav
                        s = round(sr_ema * t)
                        # pad = int(div/fps * sr_ema/div)
                        # lines[key].set_data([i*pad, i*pad], [-1, 1])
                        lines[key].set_data([s, s], [-60, 60])
                return list(lines.values())

            anim = animation.FuncAnimation(fig, update, frames=int(n_sig/div)+1, fargs=([], lines), blit=True)
            # Write animation
            anim.save(tmp_mp4, writer=writer)
            # Combine with audio
            subprocess.run(['ffmpeg', '-i', tmp_mp4, '-i', tmp_wav, '-c:v', 'copy', '-c:a', 'copy', file_name, '-y'], check=True)
        finally:
            # Clean up temp files if they exist
            if os.path.exists(tmp_mp4):
                os.remove(tmp_mp4)
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)


if __name__ == '__main__':
    file_name = 'example/F01_B01_S01_R01_N.mat'
    
    # Initiate
    mm = Viewer()
    # Load .mat file
    mm.load(file_name)
    # Convert to python dictionary
    mm.mat2py(save_file='example/test.pkl')
    # Convert back to mat file
    mm.py2mat('example/test.mat', mm.data)
    # Visualize
    mm.plot(channel_list=['AUDIO','TR', 'TB', 'TT','JAW','UL','LL'], show=False, file_name='result/test.png')
    # Animate
    mm.animate('result/test.mov', channel_list=['AUDIO','TR', 'TB', 'TT'])
    print('done')
