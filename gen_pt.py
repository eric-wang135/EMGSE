import argparse,librosa,os,torch,scipy,numpy as np,pdb
from tqdm import tqdm
from util import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_path', type=str, default='data_1spk5/train/noisy')
    parser.add_argument('--clean_path', type=str,default='data_1spk5_tdf/train/clean')
    parser.add_argument('--out_path', type=str, default='./trainpt_1spk5/')
    parser.add_argument('--only_clean', action='store_true', default=False)
    parser.add_argument('--val', action='store_true', default=True)
    parser.add_argument('--encoded', action='store_true', default=False)
    parser.add_argument('--norm_emg', type=str, default='')
    parser.add_argument('--norm_spec', type=str, default=None)
    parser.add_argument('--context', type=int, default=None)
    parser.add_argument('--task', type=str, default='denoise')  # denoise / synthesis 
    parser.add_argument('--pwg_path', type=str, default='./pretrain_model')
    parser.add_argument('--down', action='store_true', default=False)
    parser.add_argument('--save_phase', action='store_true', default=False)
    parser.add_argument('--delay', action='store_true', default=False)
    parser.add_argument('--log1p', action='store_true', default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = get_args()
    train_path = args.noisy_path
    clean_path = args.clean_path
    config_path = f'{args.pwg_path}/config.yml'
    hdf5_path = f'{args.pwg_path}/stats.h5'
    out_path = args.out_path
    
    #load pwg parameter
    if args.task=='synthesis':
        config = get_config(config_path)
        scaler = get_stats(hdf5_path)
        # Create a Mel filter-bank with num_mels=80 filters, so the mel_basis shape would be [fft_size/2+1,80].
        mel_basis = librosa.filters.mel(config["sampling_rate"], config["fft_size"], config["num_mels"], config["fmin"], config["fmax"])
        torch.save(torch.from_numpy(mel_basis.T),f'{args.pwg_path}/mel_basis.pt')
        torch.save(torch.from_numpy(scaler.mean_),f'{args.pwg_path}/mean.pt')
        torch.save(torch.from_numpy(scaler.scale_),f'{args.pwg_path}/scale.pt')
        # convert data to tensor
        mel_basis, mean, scale = torch.load(f'{args.pwg_path}/mel_basis.pt'), torch.load(f'{args.pwg_path}/mean.pt'), torch.load(f'{args.pwg_path}/scale.pt')

    n_frame = 64

    # Down sampling emg
    #step = 4  if args.task=='denoise' else 2
    step = 1
    #generate noisy file
    if args.task=='denoise':
        noisy_files = get_filepaths(train_path)
        for wav_file in tqdm(noisy_files):
            if args.only_clean:
                break
            wav,sr = librosa.load(wav_file,sr=16000)
            wav_name = wav_file.split('/')[-1]
            noise = wav_file.split(os.sep)[-2]
            snr = wav_file.split(os.sep)[-3]
            nout_path = os.path.join(out_path,'Noisy',snr,noise,wav_name.split(".")[0])
            if args.save_phase:
                nphase_path = os.path.join(out_path,'Nphase',snr,noise,wav_name.split(".")[0])
                spec,n_phase,n_len = make_spectrum(y=wav,mode=args.norm_spec)
                spec = torch.from_numpy(spec).t()
            else:
                spec = torch.from_numpy(make_spectrum(y=wav,mode=args.norm_spec)[0]).t()
                for i in np.arange(spec.shape[0]//n_frame):
                    nout_name = nout_path+'_'+str(i)+'.pt'
                    check_folder(nout_name)
                    torch.save( spec[i*n_frame:(i+1)*n_frame] ,nout_name)
                    if args.save_phase:
                        check_folder(nphase_path)
                        torch.save( spec, nphase_path+'.pt')
                        np.save( nphase_path+'_len', n_len)
                        np.save( nphase_path, n_phase)
        if args.val:
            noisy_files = get_filepaths(train_path.replace('train','val'))
            for wav_file in tqdm(noisy_files):
                if args.only_clean:
                    break
                wav,sr = librosa.load(wav_file,sr=16000)
                wav_name = wav_file.split('/')[-1]
                noise = wav_file.split(os.sep)[-2]
                snr = wav_file.split(os.sep)[-3]
                nout_path = os.path.join(out_path,'val',snr,noise,wav_name.split(".")[0])
                if args.save_phase:
                    nphase_path = os.path.join(out_path,'val_Nphase',snr,noise,wav_name.split(".")[0])
                    spec,n_phase,n_len = make_spectrum(y=wav,mode=args.norm_spec)
                    spec = torch.from_numpy(spec).t()
                else:
                    spec = torch.from_numpy(make_spectrum(y=wav,mode=args.norm_spec)[0]).t()
                if args.delay:
                    spec = spec[2:,:]
                for i in np.arange(spec.shape[0]//n_frame):
                    nout_name = nout_path+'_'+str(i)+'.pt'
                    check_folder(nout_name)
                    torch.save( spec[i*n_frame:(i+1)*n_frame] ,nout_name)
                

    #generate clean file
    clean_files = get_filepaths(clean_path) 
    # iterate through all the file names in filepath "clean_files"
    for wav_file in tqdm(clean_files):
        wav_name = wav_file.split('/')[-1]
        c_file = os.path.join(clean_path,wav_name)
        c_wav,sr = librosa.load(c_file,sr=16000)
        c_wav = c_wav.astype('float32')
        #print("audio time:",c_wav.shape[0]/16000)
        #print("audio shape:",c_wav.shape[0])
        cout_path = os.path.join(out_path,'clean_log1p') if args.log1p else os.path.join(out_path,'clean')
        # Transform clean speech data into spectorgram
        #cdata = torch.from_numpy(make_spectrum(y=c_wav,mode=args.norm_spec)[0]).t() if args.task=='denoise' else get_mel(c_wav,config,mel_basis,mean,scale)[0]
        cdata = torch.from_numpy(make_spectrum(y=c_wav)[0]).t() if args.task=='denoise' else get_mel(c_wav,config,mel_basis,mean,scale)[0]
        print("spectrum data shape:",cdata.shape)
        # read in corresponding emma data
        mat = read_enc_emg(c_file.replace('.wav','.npy'),args.task,norm=args.norm_emg,context=args.context) if args.encoded else read_emg(c_file.replace('.wav','.npy'),step,args.task,args.log1p,args.down)
        print("mat shape:",mat.shape)
        # make emma and spectrum data have same size and concatenate them 
        cdata = pad_data(cdata,mat)
        #print("cdata shape:",cdata.shape)
        # delay case
        if args.delay:
            cdata = cdata[2:,:]
        for i in np.arange(cdata.shape[0]//n_frame):
            # save each segment of data(emma+spec) with n_frame by name folder/wav_name_i.pt
            cout_name = os.path.join(cout_path,wav_name.split(".")[0]+'_'+str(i)+'.pt')
            # create a folder with cout_path if not exist
            check_folder(cout_name)
            # save the spectrum and emma data by n_frame
            torch.save(cdata[i*n_frame:(i+1)*n_frame] ,cout_name)
            
    if args.val:
        cval_path = clean_path.replace('train','val')
        clean_files = get_filepaths(cval_path)
        for wav_file in tqdm(clean_files):
            wav_name = wav_file.split('/')[-1]
            c_file = os.path.join(cval_path,wav_name)
            c_wav,sr = librosa.load(c_file,sr=16000)
            
            c_wav = c_wav.astype('float32')
            cout_path = os.path.join(out_path,'clean_log1p') if args.log1p else os.path.join(out_path,'clean')
            # Transform clean speech data into spectorgram
            #cdata = torch.from_numpy(make_spectrum(y=c_wav,mode=args.norm_spec)[0]).t() if args.task=='denoise' else get_mel(c_wav,config,mel_basis,mean,scale)[0]
            cdata = torch.from_numpy(make_spectrum(y=c_wav)[0]).t() if args.task=='denoise' else get_mel(c_wav,config,mel_basis,mean,scale)[0]
            print("spectrum data shape:",cdata.shape)
            # read in corresponding emma data
            mat = read_enc_emg(c_file.replace('.wav','.npy'),args.task,norm=args.norm_emg,context=args.context) if args.encoded else read_emg(c_file.replace('.wav','.npy'),step,args.task,args.log1p)
            print("mat shape:",mat.shape)
            # make emma and spectrum data have same size and concatenate them 
            cdata = pad_data(cdata,mat)
            print("cdata shape:",cdata.shape)
            # delay case
            if args.delay:
                cdata = cdata[2:,:]        
            for i in np.arange(cdata.shape[0]//n_frame):
                # save each segment of data(emma+spec) with n_frame by name folder/wav_name_i.pt
                cout_name = os.path.join(cout_path,wav_name.split(".")[0]+'_'+str(i)+'.pt')
                # create a folder with cout_path if not exist
                check_folder(cout_name)
                # save the spectrum and emma data by n_frame
                torch.save( cdata[i*n_frame:(i+1)*n_frame] ,cout_name)
            

    