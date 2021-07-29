# coding: utf8

import torch
import pandas as pd
import numpy as np
from os import path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import abc
from .data_utils import FILENAME_TYPE
from tools.Random_augmentations import RandAugment,  RandAugment_23, Random_RandAugment, Two_stage_Random_RandAugment
from tools.autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy


"""
Reference : https://github.com/aramis-lab/AD-DL/tree/master/clinicadl
"""

"""
Dataset_loaders
"""


class MRIDataset(Dataset):
    """Abstract class for all derived MRIDatasets."""

    def __init__(self, caps_directory, data_file,
                 preprocessing, transformations=None):
        self.caps_directory = caps_directory
        self.transformations = transformations
        self.diagnosis_code = {
            'CN': 0,
            'AD': 1,
            'sMCI': 0,
            'pMCI': 1,
            'MCI': 1,
            'unlabeled': -1}
        self.preprocessing = preprocessing

        if not hasattr(self, 'elem_index'):
            raise ValueError(
                "Child class of MRIDataset must set elem_index attribute.")
        if not hasattr(self, 'mode'):
            raise ValueError(
                "Child class of MRIDataset must set mode attribute.")

        # Check the format of the tsv file here
        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, sep='\t')
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
        else:
            raise Exception('The argument data_file is not of correct type.')

        mandatory_col = {"participant_id", "session_id", "diagnosis"}
        if self.elem_index == "mixed":
            mandatory_col.add("%s_id" % self.mode)

        if not mandatory_col.issubset(set(self.df.columns.values)):
            raise Exception("the data file is not in the correct format."
                            "Columns should include %s" % mandatory_col)

        self.elem_per_image = self.num_elem_per_image()

    def __len__(self):
        return len(self.df) * self.elem_per_image

    def _get_path(self, participant, session, mode="image"):

        if self.preprocessing == "t1-linear":
            image_path = path.join(self.caps_directory, 'subjects', participant, session,
                                   'deeplearning_prepare_data', '%s_based' % mode, 't1_linear',
                                   participant + '_' + session
                                   + FILENAME_TYPE['cropped'] + '.pt')
        elif self.preprocessing == "t1-extensive":
            image_path = path.join(self.caps_directory, 'subjects', participant, session,
                                   'deeplearning_prepare_data', '%s_based' % mode, 't1_extensive',
                                   participant + '_' + session
                                   + FILENAME_TYPE['skull_stripped'] + '.pt')
        else:
            raise NotImplementedError(
                "The path to preprocessing %s is not implemented" % self.preprocessing)

        return image_path

    def _get_meta_data(self, idx):
        image_idx = idx // self.elem_per_image
        participant = self.df.loc[image_idx, 'participant_id']
        session = self.df.loc[image_idx, 'session_id']

        if self.elem_index is None:
            elem_idx = idx % self.elem_per_image
        elif self.elem_index == "mixed":
            elem_idx = self.df.loc[image_idx, '%s_id' % self.mode]
        else:
            elem_idx = self.elem_index

        diagnosis = self.df.loc[image_idx, 'diagnosis']
        label = self.diagnosis_code[diagnosis]
        # print(image_idx)
        # print(elem_idx)
        return participant, session, elem_idx, label

    def _get_full_image(self):
        image = [169, 208, 179]

        return image

    @abc.abstractmethod
    def __getitem__(self, idx):
        pass

    @abc.abstractmethod
    def num_elem_per_image(self):
        pass


class MRIDatasetSlice(MRIDataset):

    def __init__(self, caps_directory, data_file, preprocessing="t1-linear",
                 transformations=None, mri_plane=0,
                 discarded_slices=20, mixed=False):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string or DataFrame): Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing (string): Defines the path to the data in CAPS.
            transformations (callable, optional): Optional transform to be applied on a sample.
            mri_plane (int): Defines which mri plane is used for slice extraction.
            discarded_slices (int or list): number of slices discarded at the beginning and the end of the image.
                If one single value is given, the same amount is discarded at the beginning and at the end.
            mixed (bool): If True will look for a 'slice_id' column in the input DataFrame to load each slice
                independently.
        """
        # Rename MRI plane
        self.mri_plane = mri_plane
        self.direction_list = ['sag', 'cor', 'axi']
        if self.mri_plane >= len(self.direction_list):
            raise ValueError(
                "mri_plane value %i > %i" %
                (self.mri_plane, len(
                    self.direction_list)))

        # Manage discarded_slices
        if isinstance(discarded_slices, int):
            discarded_slices = [discarded_slices, discarded_slices]
        if isinstance(discarded_slices, list) and len(discarded_slices) == 1:
            discarded_slices = discarded_slices * 2
        self.discarded_slices = discarded_slices

        if mixed:
            self.elem_index = "mixed"
        else:
            self.elem_index = None

        self.mode = "slice"
        super().__init__(caps_directory, data_file, preprocessing, transformations)

    def __getitem__(self, idx):
        participant, session, slice_idx, label = self._get_meta_data(idx)
        slice_idx = slice_idx + self.discarded_slices[0]
        # print('participant, session, slice_idx, label is ', participant, session, slice_idx, label)
        # read the slices directly
        slice_path = path.join(self._get_path(participant, session, "slice")[0:-7]
                                   + '_axis-%s' % self.direction_list[self.mri_plane]
                                   + '_channel-rgb_slice-%i_T1w.pt' % slice_idx)
        image = torch.load(slice_path)
        if self.transformations:
            image = self.transformations(image)

        sample = {'image': image, 'label': label,
                  'participant_id': participant, 'session_id': session,
                  'slice_id': slice_idx}
        return sample

    def num_elem_per_image(self):
        if self.elem_index == "mixed":
            return 1

        image = self._get_full_image()
        # print('image[self.mri_plane] is ', image[self.mri_plane])
        return image[self.mri_plane] - \
            self.discarded_slices[0] - self.discarded_slices[1]


"""
Transformations
"""


class MinMaxNormalization(object):
    """Normalizes a tensor between 0 and 1"""

    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())


def get_transforms(mode="wen_paper", minmaxnormalization=True, mode2='validation', r_n=1, r_m=5, p=0.9, n_color=3, n_shape=1):
    transformations = None
    if mode == 'wen_paper':
        trg_size = (224, 224)
        if minmaxnormalization:
            transformations = transforms.Compose([MinMaxNormalization(),
                                                  transforms.ToPILImage(),
                                                  transforms.Resize(trg_size),
                                                  transforms.ToTensor()])
        else:
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(trg_size),
                                                  transforms.ToTensor()])

    elif mode == 'plain':
        if mode2 == 'train':
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.ColorJitter(brightness=0.5),
                                                  transforms.ColorJitter(contrast=0.5),
                                                  transforms.Resize(256),
                                                  transforms.RandomCrop(224),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomVerticalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        elif mode2 == 'validation' or 'test':
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    elif mode == 'imagenet':
        if mode2 == 'train':
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(256),
                                                  ImageNetPolicy(),
                                                  transforms.RandomCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        elif mode2 == 'validation' or 'test':
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    elif mode == 'CIFAR10':
        if mode2 == 'train':
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(256),
                                                  CIFAR10Policy(),
                                                  transforms.RandomCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        elif mode2 == 'validation' or 'test':
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    elif mode == 'SVHN':
        if mode2 == 'train':
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(256),
                                                  SVHNPolicy(),
                                                  transforms.RandomCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        elif mode2 == 'validation' or 'test':
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    elif mode == 'rand-augment':
        if mode2 == 'train':
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(256),
                                                  RandAugment(r_n, r_m),
                                                  transforms.RandomCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        elif mode2 == 'validation' or 'test':
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    elif mode == 'rand-augment-23':
        if mode2 == 'train':
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(256),
                                                  RandAugment_23(r_n, r_m),
                                                  transforms.RandomCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        elif mode2 == 'validation' or 'test':
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    elif mode == 'random-rand-augment':
        if mode2 == 'train':
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(256),
                                                  Random_RandAugment(r_n, r_m),
                                                  transforms.RandomCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        elif mode2 == 'validation' or 'test':
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    elif mode == 'two-stage-random-rand-augment':
        if mode2 == 'train':
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(256),
                                                  Two_stage_Random_RandAugment(p, n_color, n_shape),
                                                  transforms.RandomCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        elif mode2 == 'validation' or 'test':
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    else:
        raise ValueError("Transforms for mode %s are not implemented." % mode)

    return transformations


################################
# tsv files loaders
################################

def load_data(train_val_path, diagnoses_list,
              split, n_splits=None, baseline=True):

    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()

    if n_splits is None:
        train_path = path.join(train_val_path, 'train')
        valid_path = path.join(train_val_path, 'validation')

    else:
        train_path = path.join(train_val_path, 'train_splits-' + str(n_splits),
                               'split-' + str(split))
        valid_path = path.join(train_val_path, 'validation_splits-' + str(n_splits),
                               'split-' + str(split))

    print("Train", train_path)
    print("Valid", valid_path)

    for diagnosis in diagnoses_list:

        if baseline:
            train_diagnosis_path = path.join(
                train_path, diagnosis + '_baseline.tsv')
        else:
            train_diagnosis_path = path.join(train_path, diagnosis + '.tsv')

        valid_diagnosis_path = path.join(
            valid_path, diagnosis + '_baseline.tsv')

        train_diagnosis_df = pd.read_csv(train_diagnosis_path, sep='\t')
        valid_diagnosis_df = pd.read_csv(valid_diagnosis_path, sep='\t')

        train_df = pd.concat([train_df, train_diagnosis_df])
        valid_df = pd.concat([valid_df, valid_diagnosis_df])

    train_df.reset_index(inplace=True, drop=True)
    valid_df.reset_index(inplace=True, drop=True)

    return train_df, valid_df


def load_data_test(test_path, diagnoses_list):

    test_df = pd.DataFrame()

    for diagnosis in diagnoses_list:

        test_diagnosis_path = path.join(test_path, diagnosis + '_baseline.tsv')
        test_diagnosis_df = pd.read_csv(test_diagnosis_path, sep='\t')
        test_df = pd.concat([test_df, test_diagnosis_df])

    test_df.reset_index(inplace=True, drop=True)

    return test_df

