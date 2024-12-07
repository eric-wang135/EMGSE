import os, argparse, torch, random
from Trainer import Trainer
from Load_model import Load_model, Load_data
from util import check_folder
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import sys

# fix random
SEED = 999
random.seed(SEED)
torch.manual_seed(SEED)
torch.set_warn_always(False)
cudnn.deterministic = True
# assign gpu
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--train_path', type=str, default='./trainpt_1spk5')
    parser.add_argument('--test_noisy', type=str, default='./data_1spk5/test/noisy')
    parser.add_argument('--test_clean', type=str, default='./data_1spk5_tdf/test/clean')

    parser.add_argument('--writer', type=str, default='./train_log')
    
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)  
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--loss_fn', type=str, default='l1')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--model', type=str, default='EMGSE_all') 
    
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--resume' , action='store_true', default=False, help='resume training')
    parser.add_argument('--checkpoint', type=str, default=None)
    
    parser.add_argument('--norm', action='store_true', default=False)
    parser.add_argument('--phase', action='store_true', default=False)
    parser.add_argument('--val', action='store_true', default=True, help='validation set data exist')
    
    parser.add_argument('--encoded', action='store_true', default=True, help='use emg features')
    parser.add_argument('--context', type=int, default=15, help='number of previous and future frames')
    
    parser.add_argument('--norm_emg', type=str, default='minmax', help='normalization method for encoded emg')
    parser.add_argument('--norm_spec', type=str, default='minmax', help='normalization method for speech spectrogram')
    parser.add_argument('--log1p', action='store_true', default=False, help='log1p for semg spectrogram')
    
    parser.add_argument('--feature', action='store_true', default=False, help='extract and save latent features')
    parser.add_argument('--output', action='store_true', default=True, help='output generated speech wav')
    parser.add_argument('--save_spec', action='store_true', default=False, help='save generated speech spectrogram')
    
    parser.add_argument('--emg_only', action='store_true', default=False, help='only use emg, no speech')
    parser.add_argument('--task', type=str, default='denoise', help='denoise or evaluate noisy data')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # get current path
    cwd = os.path.dirname(os.path.abspath(__file__))
    print(cwd)
    print(SEED)
    
    # get parameter
    args = get_args()

    # declair path
    if args.encoded:
        
        model_path = f'./save_model/{args.task}_{args.model}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr}_enc.pth.tar'
        checkpoint_path = f'./checkpoint/{args.task}_{args.model}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr}_enc.pth.tar'
        score_path = f'./Result/{args.task}_{args.model}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr}_enc.csv'
    else:
        checkpoint_path = f'./checkpoint/{args.task}_{args.model}_epochs{args.epochs}' \
                        f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                        f'lr{args.lr}.pth.tar'
        model_path = f'./save_model/{args.task}_{args.model}_epochs{args.epochs}' \
                        f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                        f'lr{args.lr}.pth.tar'
        
        score_path = f'./Result/{args.task}_{args.model}_epochs{args.epochs}' \
                        f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                        f'lr{args.lr}.csv'
    
    # tensorboard
    writer = SummaryWriter(args.writer)
    # import model from its directory and create a model
    exec (f"from model.{args.model.split('_')[0]} import {args.model} as model")
    model     = model()
    #print(model)
    #for parameter in model.parameters():
    #    print(len(parameter))
#     if args.update=='all':
#         param=model
#     else:
#         exec( f"param = model.{args.update}")
    model, epoch, best_loss, optimizer, criterion, device = Load_model(args,model,checkpoint_path,model)
    

    loader = Load_data(args) if args.mode == 'train' else 0
    print("Establish trainer")
    Trainer = Trainer(model, args.epochs, epoch, best_loss, optimizer, 
                      criterion, device, loader, writer, model_path, score_path,args)
    try:
        if args.mode == 'train':
            print("Training start")
            Trainer.train()
        Trainer.test()
        
    except KeyboardInterrupt:
        state_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss
            }

        check_folder(checkpoint_path)
        torch.save(state_dict, checkpoint_path)
        print('epoch:',epoch)
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
