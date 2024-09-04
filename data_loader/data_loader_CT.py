from torch.utils.data.dataset import Dataset
from data_loader import threeD_Transforms
import numpy as np
import warnings

warnings.simplefilter("ignore", UserWarning)
import torch
import nibabel as nib
import os
from skimage.transform import resize
from scipy.ndimage import binary_dilation, zoom
import torchio as tio


class Nifti_Dataset(Dataset):
    def __init__(self, patient_df, params):
        self.patients = np.array(patient_df['ID'])
        # self.labels = np.array(patient_df[params['inputs']['label']].astype(int))
        self.labels = np.array(patient_df[params['inputs']['label']])
        self.pt_sides = np.array(patient_df['Side'])
        self.pt_locations = np.array(patient_df['Location'])
        self.total_patients = len(self.patients)
        self.transform_1 = threeD_Transforms.Compose([
            threeD_Transforms.RandomNoise(low=-0.1, high=0.1),
            threeD_Transforms.ToTensor()
        ])
        self.torchio_transform = tio.Compose([
            tio.RandomElasticDeformation(p=0.7),
            tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(1.5, 5), p=0.5),
            tio.RandomFlip(axes=('LR',), flip_probability=0.5),
            tio.RandomAffine(scales=(0.9, 1.2), degrees=20, p=0.5),
            tio.RescaleIntensity(out_min_max=(0, 1)),
        ])
        self.modalities = params['inputs']['modalities']
        self.image_input = params['inputs']['image_input']
        self.image_size = tuple(params['inputs']['image_size'])
        self.data_location = params['files']['data_location']
        self.training_path = params['files']['training_path']
        self.projection = False
        preprocess_folder = ''
        for m in self.modalities:
            preprocess_folder += str(m)[:-7]
        for i in self.image_size:
            preprocess_folder += str(i) + 'x'
        if self.projection:
            preprocess_folder += 'p'
        preprocess_folder += self.image_input

        self.preprocess_folder = os.path.join(self.training_path, preprocess_folder)

    def __getitem__(self, index):
        global image_tensor
        idx = self.patients[index]
        label = self.labels[index]
        side = self.pt_sides[index]
        location = self.pt_locations[index]
        # make a preprocessed directory if it doesn't exist
        if not os.path.exists(self.preprocess_folder):
            os.makedirs(self.preprocess_folder)
        idx_file = os.path.join(self.preprocess_folder, str(idx))

        if self.image_input == 'CT_half_ab':
            if not os.path.exists(idx_file + '.pt'):

                series_list = [nib.load(os.path.join(self.data_location, str(idx)[:-1], series)).get_data().T for series
                               in self.modalities]
                series_array = np.array(series_list)
                series_array.astype(np.float32)
                series_array = (series_array - np.min(series_array)) / np.ptp(series_array)
                series_array_L = series_array[:, :, :, :int(series_array.shape[3] / 2)]
                series_array_R = series_array[:, :, :, int(series_array.shape[3] / 2):]
                series_array_R = np.flip(series_array_R, 3)
                series_resized_L = resize(series_array_L, self.image_size, mode='reflect', anti_aliasing=True)
                series_resized_R = resize(series_array_R, self.image_size, mode='reflect', anti_aliasing=True)
                tensor_L = self.transform_1(series_resized_L)
                tensor_R = self.transform_1(series_resized_R)
                torch.save(tensor_R, idx_file[:-1] + "A.pt")
                torch.save(tensor_L, idx_file[:-1] + "B.pt")

                image_tensor = torch.load(idx_file + '.pt')
            else:
                image_tensor = torch.load(idx_file + '.pt')
        elif self.image_input == 'CT_whole':
            if not os.path.exists(idx_file + '.pt'):
                series_list = [nib.load(os.path.join(self.data_location, str(idx), series)).get_fdata().T for series
                               in self.modalities]
                series_array = np.array(series_list)
                series_array = series_array.astype(np.float32)
                image_tensor = resize(series_array, self.image_size, mode='reflect', anti_aliasing=True)
                torch.save(image_tensor, idx_file + ".pt")
                image_tensor = torch.load(idx_file + '.pt')
                image_tensor = self.transform_1(image_tensor)
                image_tensor = self.torchio_transform(image_tensor)
            else:
                image_tensor = torch.load(idx_file + '.pt')
                image_tensor = self.transform_1(image_tensor)
                image_tensor = self.torchio_transform(image_tensor)

        elif self.image_input == 'CT_region':
            if not os.path.exists(idx_file + '.pt'):
                series_list = [nib.load(os.path.join(self.data_location, str(idx), series)).get_fdata().T for series
                               in self.modalities]
                series_array = np.array(series_list)
                series_array = series_array.astype(np.float32)
                # load the atlas file
                mask_file = nib.load("vasc_terr_r.nii.gz")
                atlas = mask_file.get_fdata() / 2
                label_terr = ['Whole_Brain', 'R_ACA', 'R_MCA', 'R_PCA', 'R_ICA', 'R_PICA', 'L_ACA', 'L_MCA', 'L_PCA',
                              'L_ICA', 'L_PICA', ]
                # generate compound mask
                mask = np.zeros(atlas.T.shape)
                location = location.split('/')
                for l in location:
                    if l.startswith('M'):
                        l = 'MCA'
                    elif l.startswith('PI'):
                        l = 'PICA'
                    elif l.startswith('P'):
                        l = 'PCA'
                    elif l.startswith('I') or l.startswith('C'):
                        l = 'ICA'
                    # index = side + "_" + l
                    index_0 = "L_" + l
                    index_1 = "R_" + l
                    mask += atlas.T == label_terr.index(index_0)
                    mask += atlas.T == label_terr.index(index_1)
                # dilate the mask
                kernel = np.ones((3, 3, 3), np.uint8)
                img_dil = binary_dilation(mask, kernel, iterations=2)
                # multiply the mask
                series_array_masked = series_array * img_dil
                image_tensor = resize(series_array_masked, self.image_size, mode='reflect', anti_aliasing=True)

                torch.save(image_tensor, idx_file + ".pt")
                image_tensor = torch.load(idx_file + '.pt')
                image_tensor = self.torchio_transform(image_tensor)
                image_tensor = self.transform_1(image_tensor)
            else:
                image_tensor = torch.load(idx_file + '.pt')
                image_tensor = self.torchio_transform(image_tensor)
                image_tensor = self.transform_1(image_tensor)

        return idx, image_tensor.float(), label

    def __len__(self):
        return self.total_patients  # of how many examples(images?) you have
