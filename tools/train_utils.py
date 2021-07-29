# coding: utf8

import numpy as np
import os
from time import time
from .earlystopping import EarlyStopping
from .iotools import check_and_clean, save_checkpoint
from .val_test_utils import test, soft_voting_in_training, mode_level_to_tsvs
import shutil


""" 
CNN train
"""


def train(model, train_loader, valid_loader, criterion, optimizer, resume, model_dir, options, split):

    if not resume:
        check_and_clean(model_dir)

    # Initialize variables
    best_valid_accuracy = 0.0
    best_valid_loss = np.inf
    epoch = options['beginning_epoch']

    model.train()  # set the module to training mode

    # early_stopping = EarlyStopping('min', min_delta=options['tolerance'], patience=options['patience'])
    early_stopping = EarlyStopping('max', min_delta=options['tolerance'], patience=options['patience'])
    mean_loss_valid = None
    image_level_validation = dict()
    image_level_validation["accuracy"] = None

    while epoch < options['epochs'] and not early_stopping.step_early(image_level_validation["accuracy"]):
        print("At %d-th epoch." % epoch)
        model.zero_grad()
        tend = time()
        total_time = 0

        for i, data in enumerate(train_loader, 0):
            t0 = time()
            total_time = total_time + t0 - tend
            optimizer.zero_grad()
            imgs, labels = data['image'].cuda(), data['label'].cuda()
            train_output = model(imgs)
            loss = criterion(train_output, labels)
            # Back propagation
            loss.backward()
            optimizer.step()

            del imgs, labels, loss

            tend = time()
            # print('load data%s' %i)
        print('Mean time per batch loading (train):', total_time / len(train_loader) * train_loader.batch_size)

        model.zero_grad()

        # train_result, results_train = test(model, train_loader, True, criterion, mode='slice')
        # mode_level_to_tsvs(options['temporary_val'], train_result, results_train, split, "best_accuracy",
        #                    options['mode'],
        #                    dataset="train")
        # # image_level bacc
        # image_level_train = soft_voting_in_training(options['temporary_val'], split,
        #                                                  selection="best_accuracy",
        #                                                  mode=options['mode'], dataset="train",
        #                                                  selection_threshold=options['selection_threshold'])
        # mean_loss_train = results_train["total_loss"] / (len(train_loader) * train_loader.batch_size)
        # slice_level bacc of valid_set
        results_df, results_valid = test(model, valid_loader, True, criterion, mode=options['mode'])
        mode_level_to_tsvs(options['temporary_val'], results_df, results_valid, split, "best_accuracy",
                           options['mode'],
                           dataset="validation")
        # image_level bacc of valid_set
        image_level_validation = soft_voting_in_training(options['temporary_val'], split, selection="best_accuracy",
                                                         mode=options['mode'], dataset="validation",
                                                         selection_threshold=options['selection_threshold'])
        mean_loss_valid = results_valid["total_loss"] / (len(valid_loader) * valid_loader.batch_size)
        print('mean_loss_valid is ', mean_loss_valid)
        # print('mean_loss_train is ', mean_loss_train)
        model.train()
        # schedlue_lr.step()
        # print('Now the learning rate is ', schedlue_lr.get_last_lr())
        global_step = (epoch + 1) * len(train_loader)
        print('image level validation metrics is', image_level_validation)
        shutil.rmtree(options['temporary_val'])

        accuracy_is_best = image_level_validation["accuracy"] > best_valid_accuracy
        loss_is_best = mean_loss_valid < best_valid_loss
        best_valid_accuracy = max(image_level_validation["accuracy"], best_valid_accuracy)
        best_valid_loss = min(mean_loss_valid, best_valid_loss)

        save_checkpoint({'model': model.state_dict(),
                         'epoch': epoch,
                         'valid_loss': mean_loss_valid,
                         'valid_acc': image_level_validation["accuracy"]},
                        accuracy_is_best, loss_is_best,
                        model_dir)
        # Save optimizer state_dict to be able to reload
        save_checkpoint({'optimizer': optimizer.state_dict(),
                         'epoch': epoch,
                         'name': options['optimizer'],
                         },
                        False, False,
                        model_dir,
                        filename='optimizer.pth.tar')

        epoch += 1

    os.remove(os.path.join(model_dir, "optimizer.pth.tar"))
    os.remove(os.path.join(model_dir, "checkpoint.pth.tar"))

