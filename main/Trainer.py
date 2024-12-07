import torch
import soundfile as sf
import os, time, numpy as np, librosa, pandas as pd
from tqdm import tqdm
from util import *

class Trainer:
    def __init__(self, model, epochs, epoch, best_loss, optimizer, 
                      criterion, device, loader, writer, model_path, score_path, args):

        self.epoch = epoch
        self.epoch_count = 0
        self.epochs = epochs
        self.best_loss = best_loss
        self.model = model.to(device)
        self.optimizer = optimizer


        self.device = device
        self.loader = loader
        self.criterion = criterion

        self.train_loss = 0
        self.val_loss = 0
        self.writer = writer
        self.model_path = model_path
        self.score_path = score_path
        self.train_clean = args.test_clean.replace('test','train')
        self.task = args.task
        self.phase = args.phase
        self.pesq = 0
        self.stoi = 0
        if self.phase:
            self.nspec_paths = get_filepaths(f'{args.train_path}/Nphase','.pt')


        if args.mode=='train':
            self.train_step = len(loader['train'])
            self.val_step = len(loader['val'])
        self.args = args
       
    def save_checkpoint(self,):
        state_dict = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss
            }
        check_folder(self.model_path)
        torch.save(state_dict, self.model_path)

    def print_score(self,test_file):
        self.model.eval()
        step = 1 
        if self.args.task=='denoise':
            n_data = torch.load(test_file)
            nphase_file = test_file.replace('.pt','.npy') #test_file = xxx.pt (noisy spec)
            n_phase = np.load(nphase_file)
            n_len = np.load(test_file.replace('.pt','_len.npy')).item(0)
            #c_file = os.path.join(self.args.test_clean,wavname.split('_')[0],test_file.split('/')[-1])
            c_file = os.path.join(self.train_clean,test_file.split('/')[-1].replace('.pt','.wav'))
            clean,sr = librosa.load(c_file,sr=16000)
            mat =  read_enc_emg(c_file.replace('.wav','.npy')) if self.args.encoded else read_emg(c_file.replace('.wav','.npy'),step,self.args.task,self.args.log1p)
            
            n_data = pad_data(n_data,mat).to(self.device).unsqueeze(0).type(torch.float32)
            spec   = n_data[:,:,:257]
            emg   = n_data[:,:,257:]
            pred = self.model(spec,emg)
            loss = self.criterion(pred, spec).item()
            pred = pred.cpu().detach().numpy()
            enhanced = recons_spec_phase(pred.squeeze().transpose(),n_phase,n_len)
            
        s_pesq, s_stoi = cal_score(clean,enhanced[:len(clean)])
        self.pesq += s_pesq
        self.stoi += s_stoi

    def _train_epoch(self):
        self.train_loss = 0
        self.pesq, self.stoi = 0 , 0
        self.model.train()
        t_start =  time.time()
        step = 0
        
        self._train_step = getattr(self,f'_train_step_mode_{self.task}')
        for data in self.loader['train']:
            #print(data[1].shape)
            step += 1
            self._train_step(data)
            progress_bar(self.epoch,self.epochs,step,self.train_step,time.time()-t_start,loss=self.train_loss,mode='train')
            #if step>180:
            #    break
            
        self.train_loss /= len(self.loader['train'])
        #self.train_loss /= 180
        print(f'train_loss:{self.train_loss}')
#     @torch.no_grad()
    
        

    def _val_epoch(self):
        self.val_loss = 0
        self.model.eval()
        t_start =  time.time()
        step = 0
        
        self._val_step = getattr(self,f'_val_step_mode_{self.task}')
        for data in self.loader['val']:
            step += 1
            self._val_step(data)
            progress_bar(self.epoch,self.epochs,step,self.val_step,time.time()-t_start,loss=self.val_loss,mode='val')
        self.val_loss /= len(self.loader['val'])
        print(f'val_loss:{self.val_loss}')
        if self.phase:
            if self.epoch % 5 == 0:
                for noisy_file in self.nspec_paths:
                    self.print_score(noisy_file)
                self.pesq /= len(self.loader['train'])
                self.stoi /= len(self.loader['train'])
                print(f'pesq:{self.pesq}')
                print(f'stoi:{self.stoi}')

        if self.best_loss > self.val_loss:
            self.epoch_count = 0
            print(f"Save model to '{self.model_path}'")
            
            self.save_checkpoint()
            self.best_loss = self.val_loss


    def train(self):
        model_name = self.model.__class__.__name__ 
        while self.epoch < self.epochs and self.epoch_count<15:
            self._train_epoch()
            self._val_epoch()
            self.writer.add_scalars(f'{self.args.task}/{model_name}_{self.args.optim}_{self.args.loss_fn}', {'train': self.train_loss},self.epoch)
            self.writer.add_scalars(f'{self.args.task}/{model_name}_{self.args.optim}_{self.args.loss_fn}', {'val': self.val_loss},self.epoch)
            self.epoch += 1
            self.epoch_count += 1
        print("best loss:",self.best_loss)
        self.writer.close()
    
    
    def write_score(self,test_file,test_path,write_wav=False):
        
        self.model.eval()
        step = 1 if self.args.task=='denoise' else 1
        outname  = test_file.replace(f'{test_path}','').replace('/','_')
        if self.args.task=='denoise':
            noisy,sr = librosa.load(test_file,sr=16000)
            wavname = test_file.split('/')[-1].split('.')[0]
            #c_file = os.path.join(self.args.test_clean,wavname.split('_')[0],test_file.split('/')[-1])
            c_file = os.path.join(self.args.test_clean,test_file.split('/')[-1])
            clean,sr = librosa.load(c_file,sr=16000)
            n_data,n_phase,n_len = make_spectrum(y = noisy,mode=self.args.norm_spec)
            n_data = torch.from_numpy(n_data).t()
            c_spec = make_spectrum(y = clean)[0]
            c_spec = torch.from_numpy(c_spec).t().to(self.device).unsqueeze(0).type(torch.float32)
            mat =  read_enc_emg(c_file.replace('.wav','.npy'),norm=self.args.norm_emg,context=self.args.context) if self.args.encoded else read_emg(c_file.replace('.wav','.npy'),step,self.args.task,self.args.log1p)
            n_data = pad_data(n_data,mat).to(self.device).unsqueeze(0).type(torch.float32)
            spec   = n_data[:,:,:257]
            emg   = n_data[:,:,257:]
            
            #emg_only input            
            #spec = torch.zeros(spec.shape).to(self.device).type(torch.float32)
            #spec_only input 
            #emg = torch.zeros(emg.shape).to(self.device).type(torch.float32)
            #print(emg.shape)
            if self.args.feature:
                #EMGSE
                """
                pred,feature,emg_f,spec_f = self.model(spec,emg)
                feature = feature.cpu().detach().numpy()
                emg_f = emg_f.cpu().detach().numpy()
                spec_f = spec_f.cpu().detach().numpy()
                """
                #base
                pred,feature = self.model(spec,emg)
                feature = feature.cpu().detach().numpy()
            else: 
                pred = self.model(spec,emg)[0]
            loss = self.criterion(pred, c_spec.squeeze()).item()
            pred = pred.cpu().detach().numpy()
            
            enhanced = recons_spec_phase(pred.squeeze().transpose(),n_phase,n_len)
            
        elif self.args.task=='evaluate':
            noisy,sr = librosa.load(test_file,sr=16000)
            wavname = test_file.split('/')[-1].split('.')[0]
            #c_file = os.path.join(self.args.test_clean,wavname.split('_')[0],test_file.split('/')[-1])
            c_file = os.path.join(self.args.test_clean,test_file.split('/')[-1])
            clean,sr = librosa.load(c_file,sr=16000)
            enhanced = noisy
            pred = make_spectrum(y = noisy)[0]
        
        
        enhanced = enhanced[512:len(clean)-512]
        s_pesq, s_stoi = cal_score(clean[512:-512],enhanced)
        
        if self.args.task=='evaluate':
            with open(self.score_path, 'a') as f1:
                f1.write(f'{outname},{s_pesq},{s_stoi}\n')
        else:
            with open(self.score_path, 'a') as f1:
                f1.write(f'{outname},{s_pesq},{s_stoi},{loss}\n')
        
        if write_wav:
            method = self.model.__class__.__name__
            wav_path = test_file.replace(f'{test_path}',f'./spk5_Enhanced_wav/{method}/') 
            check_folder(wav_path)
            enhanced = enhanced/abs(enhanced).max()
            sf.write(wav_path,enhanced,sr)

        if self.args.save_spec:
            method = self.model.__class__.__name__
            spec_path = test_file.replace(f'{test_path}',f'./spk5_Enhanced_spec/{method}/').replace('.wav','')
            check_folder(spec_path)
            np.save(spec_path,pred.squeeze())
        

        if self.args.feature:
            #feature_path = test_file.replace(f'{test_path}',f'./spk5_feature_emgin_tdf3_mini/').replace('.wav','_emgin')
            feature_path = test_file.replace(f'{test_path}',f'./spk5_feature_2in_cleanspec_base_mini/').replace('.wav','_specin')
            #feature_path = test_file.replace(f'{test_path}',f'./spk5_feature_2in_tdf3_mini/').replace('.wav','_2in')
            check_folder(feature_path)
            np.save(feature_path,feature.squeeze())
            """
            emg_path = test_file.replace(f'{test_path}',f'./spk5_feature_emg_enc/').replace('.wav','_emgenc')
            spec_path = test_file.replace(f'{test_path}',f'./spk5_feature_spec_enc/').replace('.wav','_specenc')
            check_folder(emg_path)
            check_folder(spec_path)
            np.save(emg_path,emg_f.squeeze())
            np.save(spec_path,spec_f.squeeze())
            """
    
            
    def test(self):
        # load model
        #mkl.set_num_threads(1)
        self.model.eval()
        print("best loss:",self.best_loss)
        if self.args.task != 'evaluate':
            checkpoint = torch.load(self.model_path,weights_only=True)
            self.model.load_state_dict(checkpoint['model'])
            # Get the parameters of model
            """
            print(self.model)
            print(self.model.emg_enc[0].dense[0].weight[:,:].shape)
            torch.save(self.model.emg_enc[0].dense[0].weight[:,:],"./parameters/spk5_EMGSE_parameters.pt")
            input("con?")
            torch.save(self.model.spec_enc[0].dense[0].weight[:,:],"./parameters/spk5_EMGSE_parameters_audio.pt")
            for parameter in self.model.parameters():
                print(len(parameter))
                #print(parameter)
            return
            """
            test_path = self.args.test_noisy if self.args.task=='denoise' else self.args.test_clean
        else:
            test_path = self.args.test_noisy
        test_folders = get_filepaths(test_path)
        
        check_folder(self.score_path)
        if os.path.exists(self.score_path):
            os.remove(self.score_path)
        with open(self.score_path, 'a') as f1:
            f1.write('Filename,PESQ,STOI,Loss\n')
        for test_file in tqdm(test_folders):
            self.write_score(test_file,test_path,write_wav=self.args.output)
        
        data = pd.read_csv(self.score_path)
        pesq_mean = data['PESQ'].to_numpy().astype('float').mean()
        stoi_mean = data['STOI'].to_numpy().astype('float').mean()
        loss_mean = data['Loss'].to_numpy().astype('float').mean()
        with open(self.score_path, 'a') as f:
            f.write(','.join(('Average',str(pesq_mean),str(stoi_mean),str(loss_mean)))+'\n')
        
    def _train_step_mode_denoise(self, data):
        device = self.device
        noisy, clean = data
        noisy, clean = noisy.to(device).type(torch.float32), clean.to(device).type(torch.float32)
        emg = clean[:,:,257:]
        spec = clean[:,:,:257]
        pred = self.model(noisy,emg)
        loss = self.criterion(pred, spec)
        self.train_loss += loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        
    def _val_step_mode_denoise(self, data):
        device = self.device
        noisy, clean = data
        noisy, clean = noisy.to(device).type(torch.float32), clean.to(device).type(torch.float32)
        emg = clean[:,:,257:]
        spec = clean[:,:,:257]
        #spec = clean[:,:,:100]
        pred = self.model(noisy,emg)
        loss = self.criterion(pred, spec)
        self.val_loss += loss.item()
  


    
