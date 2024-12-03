'''
mviewer

2020-12-31 first created
'''
import os
import tgt
import pickle
import numpy as np
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
    with open(file_name, 'rb') as pkl:
        pkl = pickle.load(pkl)
    assert isinstance(pkl, dict), 'pickle data is not python-dictionary'
    assert len(pkl.keys()) > 0, 'there is no valid keys in data'
    return pkl


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
        f'channel_names are not matching\n   Target{channels} <==>\nProvided:{dt.keyes()}'
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


class Viewer:
    def __init__(self,
                 field_names=['NAME', 'SRATE', 'SIGNAL', 'SOURCE', 'SENTENCE', 'WORDS', 'PHONES', 'LABELS'],
                 channel_names=['TR', 'TB', 'TT', 'UL', 'LL', 'JAW'],
                 audio_channel='AUDIO',
                 ignore_meta=False,
                 wav_sr=44100,
                 ema_sr=100):
        '''
        Specify the meta-information stored in the `.mat` file
        or you can change the names if necessary. It will throw an error
        if names are not matching.

        field_names: name of the fields defined in the `.mat` file
        channel_names: name of the EMA sensors on the articulators except for AUDIO
        audio_channel: name of the audio channel name
        wav_sr: sampling rate of the audio signal
        ema_sr: sampling rate of the EMA data

        ignore_meta: if True, meta information (`field_names`, `channel_names`,
        `audio_channel`, `wav_sr`, `ema_sr` will be ignored and retrieved from
        data directly). If you are not sure the field/channel names, 
        specify `ignore_meta=True` and then check the meta information loaded from data.
        '''
        self.field_names = field_names
        self.channel_names = channel_names
        self.audio_channel = audio_channel
        self.ignore_meta = ignore_meta
        self.wav_sr = wav_sr
        self.ema_sr = ema_sr
        self.data_orig = None
        self.data = None
        self.loaded_data_type = None
        self.file_name = None

    def load(self, file_name=None):
        '''Load files either .mat or .pkl'''
        self.file_name = file_name
        assert os.path.exists(file_name), 'File does not exist'

        fname, ext = os.path.splitext(file_name)
        if ext in ['.mat', '.MAT']:
            self.loaded_data_type = 'mat'
            data = load_mat(file_name)
            self.data_orig = data
        elif ext in ['.pkl','.PKL','.pickle','.pckl']:
            self.loaded_data_type = 'pkl'
            data = load_pkl(file_name)
            self.data_orig = data
            self.data = data
        else:
            raise 'Check the file extensions. Choose either .mat or .pkl'

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
            f'Sample length differs {len(dictionary["AUDIO"]["SIGNAL"])} =\= {len(sig)}'
        assert dictionary['AUDIO']['SRATE'] == sr, \
            f'Sampling rate differs {dictionary["AUDIO"]["SRATE"]} =\= {sr}'
        dictionary['AUDIO']['SIGNAL'] = sig

        return dictionary

    def mat2py(self, data: dict=None, save_file: str=None):
        '''Convert matlab mat to python dict
        - The output `data` is a dictionary with channel names as keys
          and field names as subkeys
        - Option to save as .mat file or not

        Parameters:
        ---
        data (opt): dict. 
            dictionary object for the EMA data
        save_file (opt): str. 
            new file name for saving as pkl
        '''
        if data is None:
            assert self.loaded_data_type == 'mat', 'Load mat file first'
            assert self.data_orig is not None, 'load mat file first'
            data = get_struct(self.data_orig, field_names=self.field_names, 
                channel_names=self.channel_names, audio_channel=self.audio_channel, ignore_meta=self.ignore_meta)
        self.data = data # updated

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
        from tqdm.auto import tqdm
        import matplotlib.animation as animation
        if which('ffmpeg') is None:
            raise 'ffmpeg is not installed. Install ffmpeg to generate an animation.'
        sr_wav = self.data['AUDIO']['SRATE']
        sr_ema = self.data['TT']['SRATE']
        sig = self.data['AUDIO']['SIGNAL']
        n_sig = len(sig)
        div = 1000
        fps = sr_wav/div

        tmp_wav = os.path.join(os.path.dirname(file_name), 'tmp.wav')
        tmp_mp4 = os.path.join(os.path.dirname(file_name), 'tmp.mp4')
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
        os.system(f'ffmpeg -i {tmp_mp4} -i {tmp_wav} -c:v copy -c:a copy {file_name} -y')
        # Clean up
        os.remove(tmp_mp4)
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
