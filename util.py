import librosa,os,pdb
import numpy as np
from pypesq import pesq
from pystoi.stoi import stoi
import scipy, sklearn
import torch.nn.functional as F
import torch
import torch.nn as nn
import yaml,torch,librosa,numpy as np,sys
sys.path.append("..") 
from parallel_wavegan.utils import read_hdf5
from parallel_wavegan.models.parallel_wavegan import ParallelWaveGANGenerator
from sklearn.preprocessing import StandardScaler

def down_sampling(signal):
  # Down sampling to 2khz
  delete_idx = [0]
  cursor = 0
  segment = signal.shape[0]//128
  for i in range (segment):
    for count in range(1,4):
      if count%3 == 0:
        delete_idx.append(delete_idx[-1]+42)
      else:
        delete_idx.append(delete_idx[-1]+43)
    
  for count in range(2):
    if delete_idx[-1] +42 < signal.shape[0]:
      delete_idx.append(delete_idx[-1]+42)
    else:
        break
  delete_idx.remove(0)
  delete_idx[:] = [idx - 1 for idx in delete_idx]
  # print(delete_idx)

  return  np.delete(signal,delete_idx,axis=0)

def get_ends(marker,threshold):
    for start in range(marker.shape[0]):
        if marker[start+1]-marker[start]>threshold:
            start = start + 1
            break
    for end in range(marker.shape[0],0,-1):
        if marker[end-1]-marker[end]>threshold:
            end = end-1
            break
    print("Start and end:", start, end)
    print ("total length:", end - start)
    return start, end

def normalize(data,mode='std'):
    if len(list(data.shape)) == 1:
        data = data.unsqueeze(1)
    if mode == 'std':
        for i in range(data.shape[1]):
                data[:,i] = np.subtract(data[:,i],np.mean(data[:,i])) # zero mean
                #data[:,i] = data[:,i]/ np.max(np.abs(a[:,i])) # max scaling
                data[:,i] = data[:,i]/ np.std(data[:,i]) # unit std
    elif mode=='minmax':
        for i in range(data.shape[1]):
                min = np.min(data[:,i])
                max = np.max(data[:,i])
                data[:,i] = np.divide(np.subtract(data[:,i],min),max-min)
    else:
        min = np.min(data)
        max = np.max(data)
        data = np.divide(data-min ,max-min)
    return data.squeeze()

epsilon = np.finfo(float).eps
def read_emg(emg_path,step=1,mode='denoise',log_scale=False,downsample=False,norm=False):
    emg = np.load(emg_path)
    #emg = np.asarray(m) # get the value(1-d)
    emg = emg.astype('float32')
    if downsample:
        emg = down_sampling(emg)
    #emg = align_signals(emg)
    """ 
    # normalize each channel
    emg = sklearn.preprocessing.normalize(emg, norm="max",axis=0) 
    #rectified
    for i in range(emg.shape[1]):
        emg[:,i] = emg[:,i] - np.mean(emg[:,i])
        #for j in range(emg.shape[0]):
        #    emg[j,i] = 0 if emg[j,i] < 0 else emg[j,i]
        #emg[:,i] = emg[:,i]/ np.max(np.abs(emg[:,i]))
    # absolute value
     emg = torch.abs(emg)
    """

    #print("emg time:",emg.shape[0]/2000)
    #print("emg shape:", emg.shape)
    # log scale
    if log_scale:
        for i in range(emg.shape[0]):
            for j in range(emg.shape[1]):
                emg[i,j] = np.log1p(emg[i,j]) if emg[i,j]>0 else -np.log1p(-emg[i,j])
    
    #print("emg original shape:", emg.shape)
    emg = torch.from_numpy(emg)  # Get the data of 2d array nx18 (with 18 channels)
    
    # Context
    if step == 16:
        return emg
    if mode=='denoise':
        # hop_length = 16, kernel size = 16
        x = emg.unsqueeze(0) 
        _,_,d = x.shape  # d is the number of channels
        x = x.unsqueeze(1) # add a dimension at axis 1 , become 4d tensor 1x1xnx18
        #pad_size = (0,0,32,16-emg.shape[0]%16)  
        #x = F.pad(x, pad=pad_size, mode='replicate') # Padding emg data for extracting the marginal data 1x1x(n+4)x18
        #x = F.unfold(x, (64, d), stride=(16,d) )  # Chop emg data into 16 x d segments (output shape 1x16dxL)
        x = F.unfold(x, (544, d), stride=(16,d) )  # Chop emg data into (64+30x16) x d segments (output shape 1x16dxL)
        emg = x.transpose(1,2).squeeze()
    
    if mode=='synthesis':
        # unfold function is only supported for 4-D input tensors, so we need to expand origianl data into 4d array.
        # hop_length = 8, kernel size = 9
        x = emg.unsqueeze(0) # add a dimension at axis 0, become 3 dimension tensor 1xnx18
        _,_,d = x.shape  # d is the number of channels
        x = x.unsqueeze(1) # add a dimension at axis 1 , become 4d tensor 1x1xnx18
        pad_size = (0,0,4,4)   
        x = F.pad(x, pad=pad_size, mode='replicate') # Padding emg data for extracting the marginal data 1x1x(n+4)x18
        x = F.unfold(x, (4*2+1, d), stride=(8,d) )  # Chop emg data into 5 x d segments (shape 1x5dxn)
        emg = x.transpose(1,2).squeeze()  # Remove all dimension with size 1, become 2d array n x 5d 
     #print("emg_train_data shape:",emg.shape)
    return emg

def read_enc_emg(emg_path,mode='denoise',norm='',context=None):
    emg = np.load(emg_path)
    #emg = np.asarray(m) # get the value(1-d)
    #print("emg original shape:", emg.shape)
    emg = emg.astype('float32').squeeze()
    #emg = sklearn.preprocessing.normalize(emg, norm="max",axis=0) #normalize each feature
    #emg = align_signals(emg)
    """
    # log scale
    for i in range(emg.shape[0]):
        for j in range(emg.shape[1]):
            emg[i,j] = np.log1p(emg[i,j]) if emg[i,j]>0 else -np.log1p(-emg[i,j])
    """
    if norm!='':
        emg = normalize(emg,norm)
    
    
    emg = torch.from_numpy(emg) # Get the data of 2d array nx18 (with 18 channels)
    #print("emg shape:", emg.shape)
    if context!= None:     
        # TD15 feature
        x = emg.unsqueeze(0) 
        _,_,d = x.shape  # d is the number of channels
        x = x.unsqueeze(1) # add a dimension at axis 1 , become 4d tensor 1x1xnx18
        pad_size = (0,0,context,context)  
        #x = F.pad(x, pad=pad_size, mode='replicate') # Padding emg data for extracting the marginal data 1x1x(n+4)x18
        x = F.pad(x, pad=pad_size) # Padding emg data for extracting the marginal data 1x1x(n+4)x18
        x = F.unfold(x, (context*2+1, d), stride=(1,d) )  # Chop emg data into 16 x d segments (shape 1x16dxL)
        emg = x.transpose(1,2).squeeze()

    return emg

def check_path(path):
    # Check if path directory exists. If not, create a file directory
    if not os.path.isdir(path): 
        os.makedirs(path)
        
def check_folder(path):
    # Check if the folder of path exists
    path_n = '/'.join(path.split('/')[:-1])
    check_path(path_n)

def cal_score(clean,enhanced):
    # Calculte scores of enhanced speech
    clean = clean/abs(clean).max()
    enhanced = enhanced/abs(enhanced).max()
#     pdb.set_trace()
    s_stoi = stoi(clean, enhanced, 16000)
    s_pesq = pesq(clean, enhanced, 16000)
    
    return round(s_pesq,5), round(s_stoi,5)


def get_filepaths(directory,ftype='.wav'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

    return sorted(file_paths)

def pad_data(spec, mat):
    # Make spectrum and emma data have same number of data in time sequence by padding emma or remove 
    # pad in the front to create the delay effect
    
    #pad_size = (0,0,spec.shape[0]-mat.shape[0]//2,spec.shape[0]-mat.shape[0]//2)
    if len(list(mat.shape)) == 1:
        mat = mat.unsqueeze(1)
    #mat = F.pad(mat.unsqueeze(0).unsqueeze(0), pad=pad_size,mode='replicate').squeeze(0).squeeze(0)
    if spec.shape[0]!=mat.shape[0]:
        pad_size = (0,0,0,spec.shape[0]-mat.shape[0])
        mat = F.pad(mat.unsqueeze(0).unsqueeze(0), pad=pad_size,mode='replicate').squeeze(0).squeeze(0)
    # concatenate them into a big tensor
    return torch.cat((spec,mat),dim=-1)

# get spectrum of input file (filename or y)
#feature type logmag v.s. lps
def make_spectrum(filename=None, y=None, is_slice=False, feature_type='logmag', mode=None, FRAMELENGTH=None,
                 SHIFT=None, _max=None, _min=None):
    if y is not None:
        y = y
    else:
        y, sr = librosa.load(filename, sr=16000)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        if y.dtype == 'int16':
            y = np.float32(y/32767.)
        elif y.dtype !='float32':
            y = np.float32(y)

    ### Normalize waveform
    # y = y / np.max(abs(y)) / 2.

    #D = librosa.stft(y,center=False, n_fft=512, hop_length=128,win_length=512,window=scipy.signal.hamming)
    D = librosa.stft(y,center=False, n_fft=512, hop_length=128,win_length=512,window=scipy.signal.blackman)
    #D = librosa.stft(y,center=False, n_fft=512, hop_length=160,win_length=512,window=scipy.signal.blackman)
    utt_len = D.shape[-1]
    phase = np.exp(1j * np.angle(D))
    D = np.abs(D)

    ### Feature type
    if feature_type == 'logmag':
        Sxx = np.log1p(D)
    elif feature_type == 'lps':
        Sxx = np.log10(D**2)
    else:
        Sxx = D
    ### normalizaiton mode
    if mode == 'mean_std':
        mean = np.mean(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))
        std = np.std(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))+1e-12
        Sxx = (Sxx-mean)/std  
    elif mode == 'minmax':
        _min = np.max(Sxx)
        _max = np.min(Sxx)
        #Sxx = 2 * (Sxx - _min)/(_max - _min) - 1
        Sxx = (Sxx - _min)/(_max - _min)

    return Sxx, phase, len(y)

def recons_spec_phase(Sxx_r, phase, length_wav, feature_type='logmag'):
    if feature_type == 'logmag':
        Sxx_r = np.expm1(Sxx_r)
        if np.min(Sxx_r) < 0:
            print("Expm1 < 0 !!")
        Sxx_r = np.clip(Sxx_r, a_min=0., a_max=None)
    elif feature_type == 'lps':
        Sxx_r = np.sqrt(10**Sxx_r)

    R = np.multiply(Sxx_r , phase)
    result = librosa.istft(R,
                     #center=True,
                     center=False,
                     hop_length=128,#hop_length=160,
                     win_length=512,
                     #window=scipy.signal.hamming,
                     window=scipy.signal.blackman,
                     length=length_wav,
                     )
    return result


def get_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config['trim_silence'] = False
    return config

def get_stats(hdf5_path):
    scaler = StandardScaler()
    scaler.mean_ = read_hdf5(hdf5_path, "mean")
    scaler.scale_ = read_hdf5(hdf5_path, "scale")
    scaler.n_features_in_ = scaler.mean_.shape[0]
    return scaler

def get_model(model_path,config):
    model = ParallelWaveGANGenerator(**config["generator_params"])
    model.load_state_dict(torch.load(model_path, map_location="cpu")["model"]["generator"])
    model = model.eval().cuda()
    return model

class get_output(nn.Module):
    def __init__(self,config,model,device):
        super().__init__()
        self.model  = model.to(device)
        self.device = device
        self.hop_size = config["hop_size"]
    def forward(self,c):

        z = torch.randn(c.shape[0], 1, c.shape[1] * self.hop_size).to(self.device)
        c = F.pad(c.permute(0,2,1),(2,2),'reflect')
        y = self.model(z,c)
        return y

def specn(audio,mel_basis,fft_size, hop_size, window):
    # return the spectorgram (2 types: spc1p and mel)
    x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size, window=window, pad_mode="reflect")
    phase = np.exp(1j * np.angle(x_stft))
    spc = np.abs(x_stft).T
    spc1p = torch.log1p(torch.from_numpy(spc))
    #convert spc to tensor
    spc = torch.expm1(spc1p)
    return spc1p,torch.log10(torch.mm(spc, mel_basis))

def get_mel(data,config,mel_basis,mean,scale):
    # get melspectrum of data
    spec1p, mel = specn(data,mel_basis,hop_size=config["hop_size"],
                           fft_size=config["fft_size"],
                           window=config["window"],
                           )
    # padding audio for
    audio = np.pad(data, (0, config["fft_size"]), mode="reflect")
    audio = audio[:len(mel) * config["hop_size"]]
    mel = (mel-mean[None,:])/scale[None,:]
    return spec1p,mel,audio

def progress_bar(epoch, epochs, step, n_step, time, loss, mode):
    line = []
    line = f'\rEpoch {epoch}/ {epochs}'
    loss = loss/step
    if step==n_step:
        progress = '='*30
    else :
        n = int(30*step/n_step)
        progress = '='*n + '>' + '.'*(29-n)
    eta = time*(n_step-step)/step
    line += f'[{progress}] - {step}/{n_step} |Time :{int(time)}s |ETA :{int(eta)}s  '
    if step==n_step:
        line += '\n'
    sys.stdout.write(line)
    sys.stdout.flush()


def make_emgspec(filename=None, y=None, is_slice=False, feature_type='logmag', mode=None, FRAMELENGTH=None,
                 SHIFT=None, _max=None, _min=None):
    y = np.float32(y)
    ### Normalize waveform
    # y = y / np.max(abs(y)) / 2.
    D = librosa.stft(y,center=False, n_fft=64, hop_length=16,win_length=64,window=scipy.signal.hamming)
    utt_len = D.shape[-1]
    D = np.abs(D)
    ### Feature type
    if feature_type == 'logmag':
        Sxx = np.log1p(D)
    elif feature_type == 'lps':
        Sxx = np.log10(D**2)
    else:
        Sxx = D
    ### normalizaiton mode
    if mode == 'mean_std':
        mean = np.mean(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))
        std = np.std(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))+1e-12
        Sxx = (Sxx-mean)/std  
    elif mode == 'minmax':
        _min = np.max(Sxx)
        _max = np.min(Sxx)
        Sxx = (Sxx - _min)/(_max - _min)
    return Sxx

def smooth(x,window_len=15,window='flat'):
    if x.ndim != 1:
        raise ValueError#, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError#, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError#, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y