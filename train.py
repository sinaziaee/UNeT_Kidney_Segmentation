from dataset_utils.makedataset import makeDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils.config as config
import pickle
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.losses import GeneralizedDiceLoss
from torchvision import transforms
from models.eff_unet import EffUNet
from utils.dice_score import DiceScore


# define Transform
tr = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90)
])

dataset_folder_path = 'data_npy2'

trainds_augmented = makeDataset(kind='train', location=dataset_folder_path, transform=tr)
trainds_original = makeDataset(kind='train', location=dataset_folder_path)
trainds = torch.utils.data.ConcatDataset([trainds_augmented, trainds_original])

validds = makeDataset(kind='valid', location=dataset_folder_path)
BATCH_SIZE = 60                                                         
trainLoader = DataLoader(trainds, batch_size=BATCH_SIZE, shuffle=True,
                        pin_memory=config.PIN_MEMORY)
validLoader = DataLoader(validds, batch_size=BATCH_SIZE, shuffle=False,
                        pin_memory=config.PIN_MEMORY)

print(config.DEVICE)
output_folder = 'final_result16'
params = [0.001]
os.makedirs(output_folder, exist_ok=True)
for (lr_) in params:
    # Define Model################################################################################################
    model = EffUNet(1, 5, use_xavier=True, use_batchNorm=True, dropout=0.2, retain_size=True, nbCls=2)
    
    devices = 'cpu'
    device_num = 0
    if torch.cuda.is_available():
        devices = 'gpu'
        device_num = torch.cuda.device_count()
        if device_num > 1:
            model = torch.nn.DataParallel(model)
    model.to(config.DEVICE)
    #############################################################################################################

    # Define History, optimizer, schedular, loss function########################################################
    history = {'train_loss': [], 'valid_loss': [], 'dice_valid_score': []}
    num_train = int(len(trainds) // BATCH_SIZE)
    writer = SummaryWriter(log_dir='./runs/Train')
    opt = torch.optim.NAdam(model.parameters(), lr=lr_)
    schedular = ReduceLROnPlateau(opt, 'min', patience=5, factor=0.25, verbose=True)
    dicelossfunc = GeneralizedDiceLoss(normalization='softmax')
    
    diceScore = DiceScore()
    #############################################################################################################
    NO_EPOCH = 40
    # main train#################################################################################################
    pbar = tqdm(range(NO_EPOCH), leave=False, position=0)
    max_avgvaliddice = 0
    for e in pbar:
        model.train()
        totalloss = 0
        totalvalidloss = 0
        totalvaliddice = 0

        trainstep = 0
        validstep = 0

        inner_pbar = tqdm(range(num_train), leave=False, position=1)
        data_iter = iter(trainLoader)
        for i in inner_pbar:
            (x, y) = next(data_iter)
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

            pred = model(x)
            loss = dicelossfunc(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            totalloss += loss
            trainstep += 1
            inner_pbar.set_postfix({'Train_loss': "{:.4f}".format(loss)})


        with torch.no_grad():
            model.eval()
            for (x, y) in validLoader:
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

                pred = model(x)
                validloss = dicelossfunc(pred.clone(), y.clone())
                totalvalidloss += validloss

                validScore = diceScore(pred, y)

                totalvaliddice += validScore
                validstep += 1

        avgloss = (totalloss / trainstep).cpu().detach().numpy()
        avgvalidloss = (totalvalidloss / validstep).cpu().detach().numpy()
        avgvaliddice = (totalvaliddice / validstep).cpu().detach().numpy()
        
        schedular.step(avgvalidloss)

        history['train_loss'].append(avgloss)
        history['valid_loss'].append(avgvalidloss)
        history['dice_valid_score'].append(avgvaliddice)

        writer.add_scalar('train_loss', avgloss, e)
        writer.add_scalar('validation_loss', avgvalidloss, e)
        writer.add_scalar('validation_dice', avgvaliddice, e)

        writer.add_scalars('loss', {'Train': avgloss, 'Valid': avgvalidloss}, e)

        pbar.set_postfix({'Train_avg_loss': '{:.4f}'.format(avgloss),
                        'Valid_avg_loss': '{:.4f}'.format(avgvalidloss),
                          'Valid_avg_dice': '{:.4f}%'.format(100 * avgvaliddice)})

        torch.save(model.state_dict(), './{}/unet_{}.pt'.format(output_folder, e + 1))
        with open('./{}/history_{}.pkl'.format(output_folder, e + 1), 'wb') as f:
            pickle.dump(history, f)
            
        if avgvaliddice > max_avgvaliddice:
            max_avgvaliddice = avgvaliddice
            torch.save(model.state_dict(), f'./{output_folder}/UNet.pt')
            
    writer.flush()
    writer.close()

    print('Saving model...\n\n')
    torch.save(model.state_dict(), f'./{output_folder}/UNet.pt')

    print('Saving figure...\n\n')
    plt.style.use('ggplot')
    plt.figure(figsize=(15, 10))
    plt.plot(history['train_loss'], label='Train_Dice_Loss')
    plt.plot(history['valid_loss'], label='Validation_Dice_Loss')
    plt.title('Training Dice Score on Dataset')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Dice Loss')
    plt.legend(loc='lower left')
    plt.savefig(f'./{output_folder}/train_result.png')

    print('Saving History...\n\n')
    with open(f'./{output_folder}/history.pkl', 'wb') as f:
        pickle.dump(history, f)

print('***************End of System***************')
        


