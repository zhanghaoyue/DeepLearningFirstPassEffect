from torchvision import datasets
from base import BaseDataLoader
from torch.utils.data.dataset import Dataset
from data_loader import threeD_Transforms
import numpy as np
import torch
import nibabel as nib
import os
from skimage.transform import resize
from torchvision import transforms


class StrokeDataLoader(BaseDataLoader):
    """
    stroke data loading
    """

    def __init__(self, dataset_input=None, batch_size=10, shuffle=False, validation_split=0.0, num_workers=1):
        self.dataset = dataset_input
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class Nifti_Dataset(Dataset):
    def __init__(self, patient_df, params):
        self.patients = np.array(patient_df['ID'])
        self.labels = np.array(patient_df[params['inputs']['label']].astype(int))
        self.pt_sides = np.array(patient_df['Side'])
        self.total_patients = len(self.patients)
        self.transform = threeD_Transforms.Compose([
            threeD_Transforms.RandomNoise(low=-0.1, high=0.1),
            threeD_Transforms.RandomRotation(20),
            threeD_Transforms.RandomHorizontalFlip(),
            # threeD_Transforms.RandomResize((0.7,1.4)),
            # threeD_Transforms.RandomAffine(15, 0.8, 0.26,(0.14,0.21),0.21),    
            threeD_Transforms.ToTensor()
        ])
        self.modalities = params['inputs']['modalities']
        self.image_input = params['inputs']['image_input']
        self.image_size = tuple(params['inputs']['image_size'])
        self.data_location = params['files']['data_location']
        self.modalities = params['inputs']['modalities']
        self.tensorshape = None

        preprocess_folder = ''
        for m in self.modalities:
            preprocess_folder += str(m)[:-7]
        for i in self.image_size:
            preprocess_folder += str(i) + 'x'
        preprocess_folder += self.image_input

        self.preprocess_folder = '/home/stroke/'+preprocess_folder

    def __getitem__(self, index):
        idx = self.patients[index]
        label = self.labels[index]
        side = self.pt_sides[index]
        # make a preprocessed directory if it doesn't exist
        if not os.path.exists(self.preprocess_folder):
            os.makedirs(self.preprocess_folder)
        idx_file = os.path.join(self.preprocess_folder, str(idx))

        # check if resized already exists
        if not os.path.exists(idx_file + '.npy') or not os.path.exists(idx_file+'_aug.npy'):

            series_list = [nib.load(os.path.join(self.data_location, str(idx), series)).get_data().T for series in
                           self.modalities]
            series_array = np.array(series_list)

            series_array = (series_array - np.min(series_array)) / (np.max(series_array) - np.min(series_array))

            if self.image_input == 'half':
                if side == 'R':
                    series_array = series_array[:, :, :, :int(series_array.shape[3] / 2)]
                elif side == 'L':
                    series_array = series_array[:, :, :, int(series_array.shape[3] / 2):]

            use_raw = False
            if use_raw:
                series_resized = series_array
            else:
                # Resize to specified dimensions
                series_resized = resize(series_array, self.image_size, mode='reflect', anti_aliasing=True)
            series_resized = np.stack((series_resized,) * 3, axis=-1)
            series_resized = series_resized[:,5:20,:,:]
            # Save to new numpy file
            np.save(idx_file, series_resized)
            img_tensor_0 = self.transform(series_resized[0, :, :, :, :])
            img_tensor_1 = self.transform(series_resized[1, :, :, :, :])
            image_tensor = torch.stack((img_tensor_0, img_tensor_1), 0)
            image_tensor = image_tensor.permute(0, 4, 1, 2, 3)
            np.save(idx_file+'_aug', image_tensor)
        else:
            # series_resized = np.load(idx_file + '.npy')
            image_tensor = np.load(idx_file + '_aug.npy')
        self.tensorshape = image_tensor.shape

        return image_tensor, label

    def __len__(self):
        return self.total_patients  # of how many examples(images?) you have

    def __tensorshape__(self):
        return self.tensorshape
        
