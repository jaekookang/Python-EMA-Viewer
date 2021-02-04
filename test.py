'''Make animation

2021-02-04
'''

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mviewer import Viewer

if __name__=='__main__':
    # Initiate
    file_name = 'example/F01_B01_S01_R01_N.pkl'
    mm = Viewer()
    # Load .mat file
    mm.load(file_name)
    # Check plot
    mm.plot(channel_list=['AUDIO','TR', 'TB', 'TT','JAW','UL','LL'], 
            show=True, file_name='result/F01_B01_S01_R01_N.png')
    # Animate
    mm.animate('result/F01_B01_S01_R01_N.mov', channel_list=['AUDIO','TR','TB','TT','JAW','UL','LL'])