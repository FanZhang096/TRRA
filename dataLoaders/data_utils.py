# coding: utf8
import numpy as np


FILENAME_TYPE = {'full': '_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w',
                 'cropped': '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w',
                 'skull_stripped': '_space-Ixi549Space_desc-skullstripped_T1w'}


def find_image_path(caps_dir, participant_id, session_id, preprocessing):
    from os import path
    if preprocessing == "t1-linear":
        image_path = path.join(caps_dir, 'subjects', participant_id, session_id,
                               't1_linear',
                               participant_id + '_' + session_id +
                               FILENAME_TYPE['cropped'] + '.nii.gz')
    elif preprocessing == "t1-extensive":
        image_path = path.join(caps_dir, 'subjects', participant_id, session_id,
                               't1', 'spm', 'segmentation', 'normalized_space',
                               participant_id + '_' + session_id +
                               FILENAME_TYPE['skull_stripped'] + '.nii.gz')
    else:
        raise ValueError(
            "Preprocessing %s must be in ['t1-linear', 't1-extensive']." %
            preprocessing)

    return image_path

