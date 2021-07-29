

def init(mdlParams_):
    mdlParams = {}
    """
     data
     """
    # AD vs CN

    mdlParams['tsv_path_train'] = '/home/zxt/4T/Alzheimer_disease_codes_0701/data/adni_ad_cn/ADNI_labels_1/train'
    mdlParams['tsv_path_test'] = '/home/zxt/4T/Alzheimer_disease_codes_0701/data/adni_ad_cn/ADNI_labels_1/test'
    # Data is loaded from here
    mdlParams['caps_directory'] = '/home/zxt/4T/ADNI_AD_CN_CAPS1/'
    mdlParams['preprocessing'] = 't1-linear'

    """
    change before train
    """
    # Save results and model here
    mdlParams['output_dir_base'] = '/home/zxt/4T/Alzheimer_disease_codes_0701/results/'
    # save the temporary files
    mdlParams['temporary_val'] = '/home/zxt/4T/Alzheimer_disease_codes_0701/results/temporary_val/'
    # Model name
    mdlParams['model_type'] = 'efficientnet_b1'
    # mdlParams['dropout_rate'] = 0.5
    # mdlParams['drop_connect_rate'] = 0.2

    # Transforms name  'wen_paper' 'plain' 'rand-augment'  'rand-augment-23'
    # 'random-rand-augment' 'two-stage-random-rand-augment'
    mdlParams['transforms_name'] = 'two-stage-random-rand-augment'

    # for rand-augment(16, 23) and random-rand-augment
    mdlParams['N'] = 7
    mdlParams['M'] = 5

    # for 'two-stage-random-rand-augment'
    # AD classification task : 0.9 5 2
    # MCI conversion prediction task: 1 2 2
    mdlParams['p'] = 0.9
    mdlParams['N_color'] = 5
    mdlParams['N_shape'] = 2

    """
    train
    """
    # Initial learning rate
    mdlParams['numGPUs'] = [0, 1]
    # mdlParams['learning_rate'] = 5e-5 * len(mdlParams['numGPUs'])
    mdlParams['learning_rate'] = 5e-5

    # baseline?
    mdlParams['baseline'] = True
    mdlParams['minmaxnormalization'] = True
    mdlParams['epochs'] = 40
    # Batch size
    mdlParams['img_per_gpu_train'] = 64
    mdlParams['img_per_gpu_val'] = 128
    # Batch size for train or normal val
    mdlParams['trainBatchSize'] = mdlParams['img_per_gpu_train'] * len(mdlParams['numGPUs'])
    # batch size for val
    mdlParams['valBatchSize'] = mdlParams['img_per_gpu_val'] * len(mdlParams['numGPUs'])

    # diagnosis: ["AD", "CN"] or ["pMCI", "sMCI"]
    mdlParams['diagnosis'] = ["AD", "CN"]

    mdlParams['n_splits'] = None
    mdlParams['split'] = 0
    # num_workers
    mdlParams['num_workers'] = 8
    # mri_direction, 0-Sagittal plane(left-right).1-Coronal plane(front-back).2-axial plane.(top-bottom)
    mdlParams['mri_plane'] = 0
    '''
    discarded_slices
    0-discard 20 slices left and right. (169-40)
    1- discard slices left:20 right:28.
    2- discarded_slices=[55,24]
    '''
    if mdlParams['mri_plane'] == 0:
        mdlParams['discarded_slices'] = 20
    elif mdlParams['mri_plane'] == 1:
        mdlParams['discarded_slices'] = 20
    elif mdlParams['mri_plane'] == 2:
        mdlParams['discarded_slices'] = 20

    # early stopping
    # for loss
    # mdlParams['patience'] = 10
    # mdlParams['tolerance'] = 0.05
    # for val_acc
    mdlParams['patience'] = 20
    mdlParams['tolerance'] = 0.01
    # Optimizer of choice for training. (default=Adam) Choices=["SGD", "Adadelta", "Adam", "RAdam"].
    mdlParams['optimizer'] = "Adam"
    mdlParams['weight_decay'] = 1e-3
    mdlParams['selection_threshold'] = 0.0
    mdlParams['mode'] = 'slice'
    mdlParams['numCV'] = 1
    mdlParams['numClasses'] = 2
    mdlParams['beginning_epoch'] = 0
    # load model trained using ad,cn data when train pmci vs smci
    mdlParams['use_AD_CN_model'] = False
    mdlParams['AD_CN_model_path'] = 'models/efficientnet_b1/checkpoint_1.pth.tar'
    return mdlParams
