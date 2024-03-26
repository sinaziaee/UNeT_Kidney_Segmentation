import torch
import nibabel as nib
import os
import torch.nn.functional as F
import numpy as np
import time
from tqdm.auto import tqdm
from scipy.ndimage import rotate
import json


def get_current_path(path=None):
    if path:
        return path
    else:
        return os.getcwd()

new_folder = os.path.join(get_current_path(), '..', 'kits23')

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

train_image_path = make_path(os.path.join(get_current_path(), new_folder, 'train', 'image'))
train_segment_path = make_path(os.path.join(get_current_path(), new_folder, 'train', 'segmentation'))
valid_image_path = make_path(os.path.join(get_current_path(), new_folder, 'valid', 'image'))
valid_segment_path = make_path(os.path.join(get_current_path(), new_folder, 'valid', 'segmentation'))
test_image_path = make_path(os.path.join(get_current_path(), new_folder, 'test', 'image'))
test_segment_path = make_path(os.path.join(get_current_path(), new_folder, 'test', 'segmentation'))

src_folder_name = os.path.join(get_current_path(), '..', 'kits23_raw')

src_train_path = f'{src_folder_name}/train'
src_valid_path = f'{src_folder_name}/valid'
src_test_path = f'{src_folder_name}/test' 

def perform_image_slicing(root_path, saving_path):
    count = 0
    depth_list = []
    for file_name in tqdm(sorted(os.listdir(root_path))):
        file_path = os.path.join(root_path, file_name, 'imaging.nii.gz')
        img_id = str(file_name.split('_')[1])
        img = nib.load(file_path).get_fdata()
        img = np.transpose(img, (1, 2, 0))
        # normalizing the image between 0 and 1
        img=(img-img.min())/(max((img.max()-img.min()),1e-3))
        depth = img.shape[2]
        depth_list.append(depth)
        for j in range(depth):
            new_path=os.path.join(saving_path, '{:05d}.npy'.format(j+count))
            new_img = torch.tensor(img[:, :, j:j+1].astype(np.float32))
            new_img = np.array(new_img)
            new_img = np.squeeze(new_img, axis=2)
            new_img = np.expand_dims(new_img, axis=0)
            np.save(new_path, new_img)
        count += depth
    return depth_list

train_depth_list = perform_image_slicing(root_path=src_train_path, saving_path=train_image_path)
valid_depth_list = perform_image_slicing(root_path=src_valid_path, saving_path=valid_image_path)
test_depth_list = perform_image_slicing(root_path=src_test_path, saving_path=test_image_path)

depth_lists = {
    'train': train_depth_list,
    'valid': valid_depth_list,
    'test': test_depth_list
}
json_file_path = f'{new_folder}/depth_lists.json'
with open(json_file_path, 'w') as json_file:
    json.dump(depth_lists, json_file)

def perform_segment_slicing(root_path, saving_path, only_healthy=False):
    count = 0
    depth_list = []
    for file_name in tqdm(sorted(os.listdir(root_path))):
        if only_healthy == True:
            folder_path = os.path.join(root_path, file_name)
            kidney1_path = os.path.join(folder_path, 'instances', 'kidney_instance-1_annotation-1.nii.gz')
            kidney2_path = os.path.join(folder_path, 'instances', 'kidney_instance-2_annotation-1.nii.gz')
            try:
                kid1 = nib.load(kidney1_path).get_fdata()
                kid2 = nib.load(kidney2_path).get_fdata()
                merged_kidneys = np.stack([kid1, kid2], axis=0)
                kidney = np.max(merged_kidneys, axis=0)
                seg = np.transpose(kidney, (1, 2, 0))
            except Exception as e:
                kidney = kid1
                seg = np.transpose(kidney, (1, 2, 0))
                print(e)
        else:
            file_path = os.path.join(root_path, file_name, 'segmentation.nii.gz')
            seg = nib.load(file_path).get_fdata()
            seg = np.transpose(seg, (1, 2, 0))
        seg_id = str(file_name.split('_')[1])
        seg_no_cancer=np.where(seg>0,1,0).astype(np.uint8)
        depth = seg_no_cancer.shape[2]
        depth_list.append(depth)
        for j in range(depth):
            new_path=os.path.join(saving_path, '{:05d}.npy'.format(j+count))
            seg_1ch=torch.tensor(seg_no_cancer[:, :, j:j+1],dtype=torch.int64)
            seg_1ch = np.array(seg_1ch)
            seg_1ch = np.squeeze(seg_1ch, axis=2)
            seg_1ch = np.expand_dims(seg_1ch, axis=2)
            seg_1ch = torch.tensor(seg_1ch)
            seg_2ch=F.one_hot(seg_1ch,num_classes=2)
            seg_2ch=torch.squeeze(seg_2ch.permute(3,0,1,2))
            seg_2ch=np.array(seg_2ch,dtype=np.uint8)
            np.save(new_path,seg_2ch)
        count += depth
only_healthy = False
perform_segment_slicing(root_path=src_train_path, saving_path=train_segment_path, only_healthy=only_healthy)
perform_segment_slicing(root_path=src_valid_path, saving_path=valid_segment_path, only_healthy=only_healthy)
perform_segment_slicing(root_path=src_test_path, saving_path=test_segment_path, only_healthy=only_healthy)

