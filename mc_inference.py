import json
from tqdm import tqdm
from makedataset import makeDataset
from eff_unet import EffUNet
import torch
import torch.nn as nn
import numpy as np
import os
import nibabel as nib
from scipy.ndimage import rotate
from mcdropout import MCDropout2D
from my_dice_score import DiceScore



# Specify the file path
file_path = "kits23/depth_lists.json"

# Read the JSON file and convert it to a dictionary
with open(file_path, "r") as file:
    depth_dict = json.load(file)


file_names = [int(file_name[7:11]) for file_name in sorted(os.listdir('/scratch/student/sinaziaee/datasets/kits23/test'))]
test_dataset = makeDataset(kind='test', location='kits23')
output_folder = '3d_mc_predictions'
os.makedirs(output_folder, exist_ok=True)
# Activate MC Dropout for inference
MCDropout2D.activate()
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
    
def calculate_entropy(prob_maps):
    return -np.sum(prob_maps * np.log(prob_maps + 1e-8), axis=0)

NUM_PASSES = 1
start_inx = 0
treshold = 0.5
dice_list = []
entropy_list = []
variance_list = []
for num, depth in enumerate(tqdm(depth_dict['test'])): # each 3d image
    image_dice = 0
    slices = []
    for inx in tqdm(range(start_inx, start_inx+depth)): # each slice in one 3d image
        image, mask = test_dataset[inx]
        image, mask = image.unsqueeze(0).to(DEVICE), mask.unsqueeze(0).to(DEVICE)
        prob_maps = []
        with torch.no_grad():
            for pass_inx in range(NUM_PASSES): # each pass of in 1 slice of 1 3d image
                pred = model(image)
                pred = pred.cpu()
                dice = dice_score(pred.clone(), mask.cpu())
                image_dice += dice
                pred = torch.sigmoid(pred)
                pred=np.where(pred[:,0,...].cpu().detach().numpy()>treshold,1,0)
                # pred = pred.cpu().detach().numpy()
                pred = pred.squeeze(0)
                pred = rotate(pred, 270)
                prob_maps.append(pred)
        prob_maps = np.stack(prob_maps, axis=0)
        uncertainty_map = calculate_entropy(prob_maps)        
        # infered_mask = np.mean(prob_maps, axis=0)
        # rotate to be the same as the original image (due to stacking this rotation is necessary)
        # infered_mask = rotate(infered_mask, 270)
        slices.append(uncertainty_map)
                
        
        uncertainty_variance = np.var(prob_maps, axis=0)
    start_inx = start_inx + depth

    image_3d = np.stack(slices, axis=-1)
    entropy_list.append(uncertainty_map)
    variance_list.append(uncertainty_variance)
    image_3d = image_3d.astype(np.float32)
    nifti_path = os.path.join(output_folder, f'uncertainty_map_{file_names[num]:04d}.nii.gz')
    nifti_img = nib.Nifti1Image(image_3d, np.eye(4))
    nib.save(nifti_img, nifti_path)
    
    avg_image_dice = image_dice / (depth * NUM_PASSES)
    dice_list.append(avg_image_dice)
    print(avg_image_dice)

avg_dice = sum(dice_list) / len(dice_list)
print(f'Average dice score: {avg_dice}')