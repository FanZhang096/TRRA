# coding: utf8
import os
import torch
from torch import nn
# import torch_optimizer as optim
import importlib
from models import all_models
# from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataLoaders.ADNI_dataloaders import MRIDatasetSlice, get_transforms, load_data
from tools.train_utils import train
from tools.val_test_utils import test_cnn


# The following settings can speed up the convolution operation in some cases
# torch.backends.cudnn.benchmark = True

# Dictionary for model configuration
mdlParams = dict()
# mdlParams['pathBase'] = _init_paths.this_dir
# Import model config
config_file = 'ADNI.ADNI'
model_cfg = importlib.import_module('cfgs.' + config_file)
mdlParams_model = model_cfg.init(mdlParams)
mdlParams.update(mdlParams_model)

for i in range(3):

    mdlParams['tsv_path_train'] = '/home/zxt/4T/Alzheimer_disease_codes_0701/data/adni_ad_cn/ADNI_labels_%s/train' % (i+1)
    mdlParams['tsv_path_test'] = '/home/zxt/4T/Alzheimer_disease_codes_0701/data/adni_ad_cn/ADNI_labels_%s/test' % (i+1)

    mdlParams['output_dir'] = (mdlParams['output_dir_base'] + mdlParams['model_type']
                               + '_learning_rate_' + str(mdlParams['learning_rate'])
                               + '_weight_decay_' + str(mdlParams['weight_decay']) + '_labels_' + str(i+1))

    print(mdlParams['output_dir'])
    # Set visible devices
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    transformations_train = get_transforms(mdlParams['transforms_name'], mdlParams['minmaxnormalization'], 'train', mdlParams['N'], mdlParams['M'], mdlParams['p'], mdlParams['N_color'], mdlParams['N_shape'])
    transformations_val = get_transforms(mdlParams['transforms_name'], mdlParams['minmaxnormalization'], 'validation', mdlParams['N'], mdlParams['M'], mdlParams['p'], mdlParams['N_color'], mdlParams['N_shape'])
    transformations_test = get_transforms(mdlParams['transforms_name'], mdlParams['minmaxnormalization'], 'test', mdlParams['N'], mdlParams['M'], mdlParams['p'], mdlParams['N_color'], mdlParams['N_shape'])
    # Reset model graph
    importlib.reload(all_models)
    # data
    cv = 0
    training_df, valid_df = load_data(
        mdlParams['tsv_path_train'],
        mdlParams['diagnosis'],
        cv,
        n_splits=mdlParams['n_splits'],
        baseline=mdlParams['baseline'])
    data_train = MRIDatasetSlice(caps_directory=mdlParams['caps_directory'], data_file=training_df,
                                 preprocessing=mdlParams['preprocessing'], transformations=transformations_train,
                                 mri_plane=mdlParams['mri_plane'], discarded_slices=mdlParams['discarded_slices'])
    data_valid = MRIDatasetSlice(caps_directory=mdlParams['caps_directory'], data_file=valid_df,
                                 preprocessing=mdlParams['preprocessing'], transformations=transformations_val,
                                 mri_plane=mdlParams['mri_plane'], discarded_slices=mdlParams['discarded_slices'])

    train_loader = DataLoader(
        data_train,
        batch_size=mdlParams['trainBatchSize'],
        shuffle=True,
        num_workers=mdlParams['num_workers'],
        pin_memory=True
    )

    valid_loader = DataLoader(
        data_valid,
        batch_size=mdlParams['valBatchSize'],
        shuffle=False,
        num_workers=mdlParams['num_workers'],
        pin_memory=True
    )

    # model
    modelVars = dict()
    modelVars['model'] = all_models.getModel(mdlParams['model_type'])()
    if 'Dense' in mdlParams['model_type']:
        num_ftrs = modelVars['model'].classifier.in_features
        modelVars['model'].classifier = nn.Linear(num_ftrs, mdlParams['numClasses'])
        # print(modelVars['model'])
    elif 'Resnet' in mdlParams['model_type']:
        num_ftrs = modelVars['model'].fc.in_features
        modelVars['model'].fc = nn.Linear(num_ftrs, mdlParams['numClasses'])
        # print(modelVars['model'])
    # for effficientnet and regnet
    elif 'efficientnet' in mdlParams['model_type']:
        num_ftrs = modelVars['model']._fc.in_features
        modelVars['model']._fc = nn.Linear(num_ftrs, mdlParams['numClasses'])
        # print(modelVars['model'])
    elif 'vgg' in mdlParams['model_type']:
        num_ftrs = modelVars['model'].classifier[6].in_features
        modelVars['model'].classifier[6] = nn.Linear(num_ftrs, mdlParams['numClasses'])
    else:
        num_ftrs = modelVars['model'].last_linear.in_features
        modelVars['model'].last_linear = nn.Linear(num_ftrs, mdlParams['numClasses'])
    # multi gpu support
    modelVars['model'] = modelVars['model'].cuda()
    if len(mdlParams['numGPUs']) > 1:
        modelVars['model'] = nn.DataParallel(modelVars['model'], mdlParams['numGPUs'])

    # Define criterion and optimizer
    modelVars['criterion'] = torch.nn.CrossEntropyLoss()
    if mdlParams['optimizer'] == 'SGD':
        modelVars['optimizer'] = torch.optim.SGD(modelVars['model'].parameters(), lr=mdlParams['learning_rate'], momentum=0.9)
    elif mdlParams['optimizer'] == 'RMSprop':
        modelVars['optimizer'] = torch.optim.RMSprop(modelVars['model'].parameters(), lr=mdlParams['learning_rate'], alpha=0.9)
    elif mdlParams['optimizer'] == 'Adam':
        modelVars['optimizer'] = torch.optim.Adam(modelVars['model'].parameters(), lr=mdlParams['learning_rate'], weight_decay=mdlParams['weight_decay'])

    # Define output directories
    model_dir = os.path.join(
        mdlParams['output_dir'], 'fold-%i' % cv, 'models')

    print('Beginning the training task')
    # if load ad cn model when train pmci smci
    if mdlParams['use_AD_CN_model']:
        print('load ad cn model params')
        param_dict = torch.load(mdlParams['AD_CN_model_path'])
        modelVars['model'].load_state_dict(param_dict['model'])

    train(modelVars['model'], train_loader, valid_loader, modelVars['criterion'],
          modelVars['optimizer'], False, model_dir, mdlParams, cv)
    test_cnn(modelVars['model'], mdlParams['output_dir'], train_loader, "train",
             cv, modelVars['criterion'], mdlParams, gpu=True)
    test_cnn(modelVars['model'], mdlParams['output_dir'], valid_loader, "validation",
             cv, modelVars['criterion'], mdlParams, gpu=True)
