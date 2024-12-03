import numpy as np
import librosa 
import soundfile as sf
import os
from util import *
from tqdm import tqdm

# Align both siganl
# Downsampling and delay EMG signal 

def get2ends(marker,threshold):
    for start in range(marker.size):
        if marker[start+1]-marker[start]>threshold:
            start = start + 1
            break
    for end in range(marker.size-1,0,-1):
        if marker[end-1]-marker[end]>threshold:
            end = end-1
            break
    #print("Start and end:", start, end)
    #print ("total length:", end - start)
    return start, end

# get all wav file in directory
#block_paths = ['Spk1_Block1-Initial','Spk4_Block1-Initial','Spk5_Block1-Initial','Spk6_Block1-Initial']
#block_paths = ['Spk4_Block1-Initial']
block_paths = ['Spk5_Block1-Initial']
out_path ='./data_1spk5'
Corpus_path = '../../EMG-master/main/CSL-EMG_Array/'
delay = 100 # n/1000 * 2000 = 50ms

check_path(out_path)
check_path(out_path+'/train/clean')
check_path(out_path+'/test/clean')
check_path(out_path+'/val/clean')

count = 0

for i in range(len(block_paths)):
    block_path = block_paths[i]
    file_path = os.path.join(Corpus_path,block_path)
    
    for f in tqdm(os.listdir(file_path)):
        if f[0]!='0':
            continue
        audio_file = block_path+'_'+f+'_audio.wav'
        audio,fs_clean = librosa.load(os.path.join(file_path,f,audio_file),sr=16000,mono=False)
        audio = audio.T
        marker = audio[:,-1]
        start,end = get2ends(marker,0.5)
        audio = audio[start:,0]
        #print("audio shape:",audio.shape)
        #print("audio time:",audio.shape[0]/16000)

        emg_file = block_path+'_'+f+'_emg.npy'
        emg = np.load(os.path.join(file_path,f,emg_file))
        marker = emg[:,-1]
        start,end = get2ends(marker,10)
        #emg = emg[start-delay:-delay,:-1]
        emg = emg[start:,:-1]
        emg = down_sampling(emg)
        fs_emg = 2000
        fs_ratio = 16000/2000

        #print("emg shape:",emg.shape)
        #print("emg time:",emg.shape[0]/2000)
        """
        if int(f)<281:
            save_path = out_path+'/train/clean'
        elif int(f)<301:
            save_path = out_path+'/val/clean'
        else:
            save_path = out_path+'/test/clean'
        """
        
        if int(f)<41:
            save_path = out_path+'/test/clean'
        elif int(f)<61:
            save_path = out_path+'/val/clean'
        else:
            save_path = out_path+'/train/clean'
        
        sf.write(os.path.join(save_path,audio_file.replace('_audio','')), audio, fs_clean)
        np.save(os.path.join(save_path,emg_file.replace('_emg','')),emg)
        count = count + 1
      
    print('total number of utterances:', count)