import os
from glob import glob
import librosa
import soundfile as sf

files = glob("/Users/liuhaohe/Downloads/fordsp/*/*/*44k.wav") + glob("/Users/liuhaohe/Downloads/fordsp/*/*/*pinna.wav")

for file in files:
    if("44k.wav" in file): target_file = file.replace("44k.wav","48k.wav")
    if("pinna.wav" in file): target_file = file.replace("pinna.wav","pinna_48k.wav")
    if(os.path.exists(target_file)): continue
    target,_ = librosa.load(os.path.join(os.path.dirname(file), "mono.wav"), sr=None, mono=False)
    x,sr = librosa.load(file, sr=None, mono=False)
    x = librosa.resample(x, sr, 48000)
    target = target[None,...]
    
    print(file, x.shape, target.shape)
    
    minlen = min(x.shape[1], target.shape[1])
    x = x[:, :minlen]
    target = target[:, :minlen]
    
    assert x.shape[1] == target.shape[1]
    
    sf.write(target_file, x.T, samplerate=48000)
    