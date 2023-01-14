import noisereduce as nr
from scipy.io import wavfile
import sys
import librosa
import numpy as np
import wave
import glob
import os
import matplotlib.pyplot as plt
import re 
import math
from pathlib import Path 

file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

dir_name=sys.argv[1]

fig, axs = plt.subplots(6,2,figsize=(25,15),dpi=100,constrained_layout=True)
fig.suptitle(dir_name)
i=0
for filename in sorted(glob.glob(dir_name+'/receive*.wav'),key=get_order):
#    print(filename)
    rate,wavdata=wavfile.read(filename)
    axs[int(i%6)][int(i/6)].plot(wavdata)
    axs[int(i%6)][int(i/6)].set_title(os.path.basename(filename))
    i=i+1


#plt.subplots_adjust(left=0.05, bottom=0.05, right=0.9, top=1, wspace=0.2, hspace=0.6)
#plt.show()
plt.savefig(dir_name+'/mic_plot.png')

