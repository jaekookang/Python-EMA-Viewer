#!/usr/bin/env python
'''
Demo script for Python-EMA-Viewer

Demonstrates:
- Loading EMA data from .pkl file
- Plotting articulatory trajectories
- Creating animations

Usage:
    python test/test_demo.py

2021-02-04 first created
2025-12-27 moved to test/ directory and improved
'''

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mviewer import Viewer


def main():
    """Main demo function"""

    # Configuration
    data_file = 'example/F01_B01_S01_R01_N.pkl'
    plot_output = 'result/F01_B01_S01_R01_N.png'
    animation_output = 'result/F01_B01_S01_R01_N.mov'
    channels = ['AUDIO', 'TR', 'TB', 'TT', 'JAW', 'UL', 'LL']

    print('=' * 70)
    print('Python-EMA-Viewer Demo')
    print('=' * 70)

    # Check if example file exists
    if not os.path.exists(data_file):
        print(f'\n❌ Error: Example file not found: {data_file}')
        print('Please ensure example data is available.')
        sys.exit(1)

    # Ensure result directory exists
    os.makedirs('result', exist_ok=True)

    # Initialize viewer
    print('\n[1] Initializing Viewer...')
    viewer = Viewer()
    print(f'    ✓ Viewer initialized (Adapter: {viewer.adapter.DATASET_NAME})')

    # Load data
    print(f'\n[2] Loading data from: {data_file}')
    viewer.load(data_file)
    print(f'    ✓ Loaded {viewer.loaded_data_type.upper()} file')
    print(f'    ✓ Channels: {list(viewer.data.keys())}')
    if 'SENTENCE' in viewer.data['AUDIO']:
        sentence = viewer.data['AUDIO']['SENTENCE']
        print(f'    ✓ Sentence: "{sentence[:50]}..."')

    # Create plot
    print(f'\n[3] Creating plot...')
    print(f'    Channels: {channels}')
    fig, axs = viewer.plot(
        channel_list=channels,
        show=False,  # Don't display, just save
        file_name=plot_output
    )
    print(f'    ✓ Plot saved to: {plot_output}')

    # Create animation
    print(f'\n[4] Creating animation...')
    print(f'    Channels: {channels}')
    print(f'    This may take a moment...')

    try:
        viewer.animate(
            animation_output,
            channel_list=channels
        )
        print(f'    ✓ Animation saved to: {animation_output}')
    except RuntimeError as e:
        if 'ffmpeg' in str(e):
            print(f'    ⚠ Warning: ffmpeg not found - skipping animation')
            print(f'    Install ffmpeg to enable animation generation')
        else:
            raise

    print('\n' + '=' * 70)
    print('✅ Demo completed successfully!')
    print('=' * 70)
    print(f'\nOutput files:')
    print(f'  - Plot: {plot_output}')
    if os.path.exists(animation_output):
        print(f'  - Animation: {animation_output}')
    print()


if __name__ == '__main__':
    main()
