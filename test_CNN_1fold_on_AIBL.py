# coding: utf8
import os
import torch
from torch import nn
import importlib
import _init_paths
from models import all_models
from torch.utils.data import DataLoader
from dataLoaders.ADNI_dataloaders import MRIDatasetSlice, get_transforms, load_data_test
from tools.val_test_utils import test_cnn


# The following settings can speed up the convolution operation in some cases
# torch.backends.cudnn.benchmark = True

# Dictionary for model configuration
mdlParams = dict()
mdlParams['pathBase'] = _init_paths.this_dir
# Import model config
config_file = 'AIBL.AIBL'
model_cfg = importlib.import_module('cfgs.' + config_file)
mdlParams_model = model_cfg.init(mdlParams)
mdlParams.update(mdlParams_model)


for i in range(3):

    mdlParams['tsv_path_test'] = '/home/zf/AD_codes/Alzheimer_disease_codes_0701/data/aibl_ad_cn/test'

    mdlParams['output_dir'] = (mdlParams['output_dir_base'] + mdlParams['model_type']
                               + '_learning_rate_' + str(mdlParams['learning_rate'])
                               + '_weight_decay_' + str(mdlParams['weight_decay']) + '_labels_' + str(i+1))

    print(mdlParams['output_dir'])
    cv = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'  # 指定用哪几个GPU
    transformations_val = get_transforms(mdlParams['transforms_name'], mdlParams['minmaxnormalization'],
                                         'validation', mdlParams['N'], mdlParams['M'], mdlParams['p'],
                                         mdlParams['N_color'], mdlParams['N_shape'])
    transformations_test = get_transforms(mdlParams['transforms_name'], mdlParams['minmaxnormalization'], 'test', mdlParams['N'], mdlParams['M'], mdlParams['p'])
    # Reset model graph
    importlib.reload(all_models)

    # data
    test_df = load_data_test(
        mdlParams['tsv_path_test'],
        mdlParams['diagnosis'])
    print(mdlParams['tsv_path_test'])
    data_test = MRIDatasetSlice(caps_directory=mdlParams['caps_directory'], data_file=test_df,
                                preprocessing=mdlParams['preprocessing'], transformations=transformations_test,
                                mri_plane=mdlParams['mri_plane'], discarded_slices=mdlParams['discarded_slices'])

    # Use argument load to distinguish training and testing
    test_loader = DataLoader(
        data_test,
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
    # Define criterion
    modelVars['criterion'] = torch.nn.CrossEntropyLoss()
    # Define output directories
    model_dir = os.path.join(
        mdlParams['output_dir'], 'fold-%i' % cv, 'models')
    print(mdlParams['output_dir'])
    test_cnn(modelVars['model'], mdlParams['output_dir'], test_loader, "test",
             cv, modelVars['criterion'], mdlParams, gpu=True)

