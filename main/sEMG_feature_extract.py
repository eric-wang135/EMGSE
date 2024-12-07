"""
TDF feature extraction
"""
import numpy as np
import os, argparse
from scipy import signal
from util import *
from tqdm import tqdm

def normalize(data, mode='std'):
    if mode == 'std':
        for i in range(data.shape[1]):
                data[:,i] = np.subtract(data[:,i],np.mean(data[:,i])) # zero mean
                data[:,i] = data[:,i]/ np.std(data[:,i]) # unit std
    else:
        # Minmax_norm
        for i in range(data.shape[1]):
                min = np.min(data[:,i])
                max = np.max(data[:,i])
                data[:,i] = np.divide(np.subtract(data[:,i],min),max-min)
    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speaker_idx', type=str, default='5')
    parser.add_argument('--channel_num', type=int, default=35) # 28 or 35
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args() 
    
    file_dir_list = ["./data_1spk"+args.speaker_idx]       # ['./data_1spk1','./data_1spk2','./data_1spk5']
    out_dir_list = ["./data_1spk"+args.speaker_idx+"_tdf"] # ['./data_1spk1_28ch_tdf','./data_1spk2_28ch_tdf','./data_1spk5_28ch_tdf']

    sub_path = ['/train/clean/','/val/clean/','/test/clean/']
    file_dirs, out_dirs= [] , []

    for i in range(len(file_dir_list)):
        for s in sub_path:
            file_dirs.append(file_dir_list[i]+s)
            out_dirs.append(out_dir_list[i]+s)

    lowpass = signal.butter(3, 134, 'low',output='sos',fs=2000)
    highpass = signal.butter(3, 134, 'high',output='sos',fs=2000)

    win_size = 64   # 32ms
    step = 16       # 8ms
    window = np.blackman(win_size)

    feature_num = 5
    if args.channel_num == 35:
        channel = [0,1,2,3,4,5,6,8,9,10,11,12,13,14,16,17,18,19,20,21,22,24,25,26,27,28,29,30,32,33,34,35,36,37,38]  # 35 channels
    else:
        channel = [0,1,2,3,4,5,6,8,9,10,11,12,13,14,16,17,18,19,20,21,22,24,25,26,27,28,29,30]  # 28 channels

    for i in range(len(file_dirs)):
        file_dir = file_dirs[i]
        out_dir = out_dirs[i]
        check_path(out_dir)
        os.system("cp "+file_dir+"*.wav "+out_dir)
        for f in tqdm(os.listdir(file_dir)):
            if f[-1]!='y':
                continue
            emg = np.load(os.path.join(file_dir,f))
            emg = emg.astype('float32')
            #print(emg.shape)
            #emg = np.pad(emg,((win_size//2,win_size//2),(0,0)),mode='reflect') #
            win_num = (emg.shape[0]-win_size)//step + 1 #  n/16 - 64/16 + 1

            #emg = np.pad(emg,((win_size//2,win_size),(0,0)),mode='edge') #
            #emg = np.pad(emg,((win_size//2,win_size//2),(0,0)),mode='edge') #
            #print(emg.shape)

            data = np.zeros([win_num, feature_num * len(channel)])

            for i in range(len(channel)):
                    ch = channel[i]
                    emg_lowband = signal.sosfilt(lowpass,emg[:,ch])
                    emg_highband = signal.sosfilt(highpass,emg[:,ch])
                    for j in range(win_num): 
                        emg_low = emg_lowband[step*j:step*j+win_size] * window
                        emg_high = emg_highband[step*j:step*j+win_size] * window
                        data[j,i*feature_num] = np.mean(emg_low)                              # low_band mean
                        data[j,i*feature_num+1] = np.mean(np.power(emg_low,2))                 # low_band power
                        data[j,i*feature_num+2] = np.mean(np.abs(emg_high))                   # high_band absolute mean
                        data[j,i*feature_num+3] = np.mean(np.power(emg_high,2))                # high_band power
                        data[j,i*feature_num+4] = sum((emg_high[0:-1])*(emg_high[1:])<=0)/win_size    # high_band zero-crossing rate

                        #print(data[j,i])
                        #data[j,i*2+1] = np.log1p( np.sum(np.power(emg_high,2)) )# high_band power
                    #data[:,i]=normalized(data[:,i])

            np.save(os.path.join(out_dir,f),data)


