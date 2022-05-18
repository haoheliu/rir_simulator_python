import roomsimove_single
import ipdb
import soundfile as sf
import olafilt
import numpy as np
import os
from glob import glob

rt60 =0.001
# rt60 = 0.15 # in seconds
room_dim = [3.6, 5.925, 2.80] # in meters
mic_pos1 = [1.8, 2.92, 1.4] # in  meters
mic_pos2 = [1.8, 3.08, 1.4] # in  meters
sampling_rate = 48000
seg_len = 400
mic_positions = [mic_pos1, mic_pos2]

lookup = {}

files = glob("/Users/liuhaohe/Downloads/fordsp/testset/*/*48k.wav")

def find(x, y):
    x_round = round(x, 2)
    y_round = round(y, 2)
    key = "%.2f_%.2f" % (x_round, y_round)
    if(not key in lookup.keys()):
        source_pos = [1.8+x, 3.0+y, 1.4] # in meters
        rir = roomsimove_single.do_everything(room_dim, mic_positions, source_pos, rt60)
        lookup[key] = rir
    return lookup[key]     
   
def get_pos(pos, index):
    position = pos[index,:]
    return position[0], position[1]

def apply_filter(x, y, data):
    rir = find(x, y)
    data[:,0] = olafilt.olafilt(rir[:,0], data[:,0])
    data[:,1] = olafilt.olafilt(rir[:,1], data[:,1])
    return data

def proc(file):
    target_file = file.replace("48k.wav","48k_rev.wav")
    dir = os.path.dirname(file)
    [data, fs] = sf.read(file,always_2d=True)
    binarual_pos = np.load(os.path.join(dir, "binaural_pos.npy"))
    length = binarual_pos.shape[0]
    i = 0
    while(i < length):
        print(i)
        x,y = get_pos(binarual_pos, i)
        data[i:i+seg_len] = apply_filter(x, y, data[i:i+seg_len])
        i += seg_len
    maxval = np.max(np.abs(data))
    data = (data/maxval) * 0.99
    sf.write(target_file, data, sampling_rate)
    
for file in files:
    proc(file)









