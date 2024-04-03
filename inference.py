import json
from tqdm import tqdm
from dataset_utils.makedataset import makeDataset
from models.eff_unet import EffUNet
import torch
import numpy as np
import os
import nibabel as nib
from scipy.ndimage import rotate
from utils.dice_score import DiceScore


# Specify the file path
file_path = "data_npy/depth_lists.json"

# Read the JSON file and convert it to a dictionary
with open(file_path, "r") as file:
    depth_dict = json.load(file)
    

file_names = [int(file_name[7:11]) for file_name in os.listdir('/scratch/student/sinaziaee/datasets/3d_dataset/testing/labels')]
test_dataset = makeDataset(kind='test', location='data_npy')
output_folder = '3d_predictions3'
os.makedirs(output_folder, exist_ok=True)
dice_score = DiceScore()

model = EffUNet(1, 5, use_xavier=True, use_batchNorm=True, dropout=0.2, retain_size=True, nbCls=2)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
folder_path = 'final_result11'
model_name = 'unet_30.pt'
devices = 'cpu'
device_num = 0
if torch.cuda.is_available():
    devices = 'gpu'
    device_num = torch.cuda.device_count()
model = torch.nn.DataParallel(model)
model.to(DEVICE)
if torch.cuda.is_available():
    print('CUDA Available!')
    model.load_state_dict(torch.load(f'./{folder_path}/{model_name}'))
else:
    print('CUDA is unavailable, using CPU instead!')
    print('Warning: using CPU might require several hours')
    model.load_state_dict(torch.load(f'./{folder_path}/{model_name}', map_location=torch.device('cpu')))


start_inx = 0
treshold = 0.5
dice_list = []
for num, depth in enumerate(tqdm(depth_dict['test'])):
    image_dice = 0
    slices = []
    for inx in tqdm(range(start_inx, start_inx+depth)):
        image, mask = test_dataset[inx]
        image, mask = image.unsqueeze(0).to(DEVICE), mask.unsqueeze(0).to(DEVICE)
        pred = model(image)
        pred = pred.cpu()
        dice = dice_score(pred.clone(), mask.cpu())
        image_dice += dice
        pred = torch.sigmoid(pred)
        pred=np.where(pred[:,1,...].cpu().detach().numpy()>treshold,1,0)
        pred = pred.squeeze(0)
        # rotate to be the same as the original image (due to stacking this rotation is necessary)
        pred = rotate(pred, 270)
        slices.append(pred)
    image_3d = np.stack(slices, axis=0)
    image_3d = image_3d.astype(np.float32)
    nifti_path = os.path.join(output_folder, f'pred_{file_names[num]:04d}.nii.gz')
    nifti_img = nib.Nifti1Image(image_3d, np.eye(4))
    nib.save(nifti_img, nifti_path)
    start_inx = start_inx + depth
    avg_image_dice = image_dice / depth
    dice_list.append(avg_image_dice)
    print(avg_image_dice)
avg_dice = sum(dice_list) / len(dice_list)
print(f'Average dice score: {avg_dice}')

print(dice_list)