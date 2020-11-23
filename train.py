# Imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import os
import h5py as h5
from torch.utils.data import Dataset, DataLoader # Gives easier dataset managment and creates mini batches
import math
import random
import numpy as np
from datetime import datetime
import csv
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from hparams import *
from models import archs_dict


def chunks(list_to_chunk, batch_size):
    '''Chunk list such that len of each split indexes list <= batch_size'''
    random.seed(1) # for shuffling indexes before batching
    random.shuffle(list_to_chunk)

    chunked_list = list((list_to_chunk[i:i+batch_size] for i in range(0, len(list_to_chunk), batch_size)))
    return chunked_list


class spdataset(Dataset):
    def __init__(self, datapath, nb):
        self.hfile = h5.File(datapath, "r")
        self.data = [] # number_of_files_in_batch , feat_dim, seq_en
        self.targets = [] # correct labels, e.g. 0, 1, 2
        self.nb = nb # number of batches
        self.batches = []
        classes_set = set()

        self.total_files = 0

        for group in self.hfile.keys():
            index_root = len(self.data)
            target_set = set() # check how many of targets appear in each group

            for dset in self.hfile[group].keys():
                dset_matrix = self.hfile[group][dset][:]
                dset_data = dset_matrix[:,:-1]
                dset_data = torch.tensor(dset_data)
                dset_data = dset_data - torch.mean(dset_data, 0) # normalise - subtract mean
                dset_data = dset_data / torch.std(dset_data) # normalise - divide by stdev

                dset_target = dset_matrix[0,-1]
                dset_target = torch.tensor(dset_target)
                target_set.add(int(dset_target.item()))

                self.data.append(dset_data)
                self.targets.append(dset_target)

            numfiles = len(self.hfile[group].keys())
            self.total_files += numfiles
            indexes_list = [index_root + i for i in range(numfiles)]
            split_indexes = chunks(indexes_list, batch_size)
            self.batches.extend(split_indexes)

        global num_classes
        num_classes = len(target_set)
        global idim
        idim = dset_data.size(-1) # softcode idim - different for mfcc, fbank etc.

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        data = [self.data[i] for i in self.batches[idx]]
        targets = [self.targets[i] for i in self.batches[idx]]
        stacked_data = torch.stack(data, dim=0)
        stacked_targets = torch.stack(targets, dim=0)
        stacked_targets = torch.unsqueeze(stacked_targets, 1)
        return stacked_data, stacked_targets

    def batch_size(self, idx):
        return self.data[idx].shape[1]


def create_dataset(hdf_file):
    traindata = spdataset("TRAIN_"+hdf_file, batch_size)
    validdata = spdataset("VALID_"+hdf_file, batch_size)
    testdata = spdataset("TEST_"+hdf_file, batch_size)
    print(f"Created datasets from {hdf_file}. Idim={idim}, num_classes={num_classes}.")
    return traindata, validdata, testdata


def load_model(model_arch):
    print('Making model')
    if model_arch is not None:
        if "_" in model_arch: # meaning LSTM + Linear or LSTM + CNN
            model = archs_dict[model_arch](idim, num_classes, nhu, nhu2, num_layers, dropout).to(device)
        else:
            model = archs_dict[model_arch](idim, num_classes, nhu, num_layers).to(device)
    else:
        print('*** Please specify model architecture and try again ***')

    criterion = nn.CrossEntropyLoss()

    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    return model, criterion, optimizer


def load_datasets():
    train_loader = torch.utils.data.DataLoader(traindata, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validdata, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testdata, shuffle=True)
    return train_loader, valid_loader, test_loader


def get_save_file_info():
    exp_name = f'{opt}_lr{lr}_nhu-{nhu}_nhu2-{nhu2}_nl{num_layers}_{model_arch}{num_cnns}'
    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y_%H:%M")
    dir_name = f'../experiments/{save_dir}/{exp_name}/'
    # all experiments with same hparams saved in same dir with unique exp names
    try:
        os.mkdir(dir_name) # make exp dir if it doesn't exist yet
        print(f'Created {dir_name}')
    except:
        pass

    file_name = dir_name+exp_name+'_'+date_time
    return file_name


def train_model(train_loader, valid_loader, csvfile):
    # init variables as large values for guaranteed improvement first epoch
    prev_valid_loss = 999
    lowest_valid_loss = 999
    lowest_valid_error = 999
    same_epoch_valid_error = 999 # in cases of NaN early stopping
    lowest_valid_epoch = 999 # in cases of NaN early stopping
    num_increase = 0 # stop training after nth time valid loss > than previous valid loss

    # logs
    train_losses = []
    valid_losses = []
    train_errors = []
    valid_errors = []

    for epoch in range(maxiter):
        # clear previous epoch losses
        train_epoch_loss = 0
        train_epoch_error = 0
        valid_epoch_loss = 0
        valid_epoch_error = 0

        early_stopping = False # continue training until early stopping = True

        # TRAIN SET
        # reset loss
        running_samples = 0 # n samples != n batches
        running_loss = 0
        running_corrects = 0

        model.train()
        for data, targets in train_loader:
            data = data.float().squeeze_(0).to(device=device)
            targets = targets.long().squeeze_(0).squeeze_(-1).to(device=device)
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)

            # gradient descent or adam step
            optimizer.step()

            _, predictions = scores.max(1)
            running_samples += predictions.size(0) # number of utts, not number of batches
            running_loss += loss.item()
            running_corrects += torch.sum(predictions == targets.data)

        train_epoch_loss = running_loss / len(train_loader.dataset) # number of batches in dataset
        train_epoch_acc = running_corrects.float() / float(running_samples)*100 # number of utts
        train_epoch_error = 100 - train_epoch_acc.item()

        train_losses.append(train_epoch_loss)
        train_errors.append(train_epoch_error)


        # VALID SET
        running_samples = 0
        running_loss = 0
        running_corrects = 0

        model.eval()
        for data, targets in valid_loader:
            data = data.float().squeeze_(0).to(device=device)
            targets = targets.long().squeeze_(0).squeeze_(-1).to(device=device)
            scores = model(data)
            loss = criterion(scores, targets)

            _, predictions = scores.max(1)
            running_samples += predictions.size(0) # number of utts, not number of batches
            running_loss += loss.item()
            running_corrects += torch.sum(predictions == targets.data)

        valid_epoch_loss = running_loss / len(valid_loader.dataset) # number of batches in dataset
        valid_epoch_acc = running_corrects.float() / float(running_samples)*100 # number of files
        valid_epoch_error = 100 - valid_epoch_acc.item()

        valid_losses.append(valid_epoch_loss)
        valid_errors.append(valid_epoch_error)

        # To check if model is improving - early stopping decision + logs info
        if valid_epoch_loss > lowest_valid_loss: # model not improving
            num_increase += 1 # stop once num_increase reaches num_threshold
            if num_increase < num_threshold:
                print(f'\t\tValid loss going up for {num_increase} consecutive epochs.')
            elif num_increase == num_threshold: # stop training
                print(f'\t\tValid loss increased for {num_threshold} consecutive epochs, ending training at epoch {epoch+1}')
                early_stopping = True

        elif valid_epoch_loss < lowest_valid_loss: # save model with best loss
            print('\t\tModel improving, will keep running (new loss {:.4f} lower than prev loss)'.format(lowest_valid_loss - valid_epoch_loss))
            num_increase = 0 # reset so we get consecutive n increases only
            lowest_valid_loss = valid_epoch_loss
            same_epoch_valid_error = valid_epoch_error
            lowest_valid_epoch = epoch+1
            torch.save(model.state_dict(), model_save_path_LOSS) # save best model for accuracy check
            print('saved as best loss so far')

        if valid_epoch_error < lowest_valid_error: # save model with best error
            num_increase = 0 # reset because model technically improving
            lowest_valid_error = valid_epoch_error
            same_epoch_valid_loss = valid_epoch_loss
            torch.save(model.state_dict(), model_save_path_ERROR) # save best model for accuracy check
            print('saved as best error so far')

        if valid_epoch_loss != valid_epoch_loss: # nan != nan - stop if loss being returned is nan
            print('Stopping due to NaN values')
            early_stopping = True
            NaN = True

        print('Epoch {} \tTrain loss: {:.4f} \tValid loss: {:.4f}'.format(epoch+1, train_epoch_loss, valid_epoch_loss))
        print('\t\tTrain error: {:.4f} \tValid error: {:.4f}'.format(train_epoch_error, valid_epoch_error))
        csvwriter.writerow([epoch+1, train_epoch_loss, valid_epoch_loss, train_epoch_error, valid_epoch_error, datetime.now()])
        torch.save(model.state_dict(), model_save_path) # save most recent model for checkpoints
        if early_stopping == True:
            break

    if early_stopping == True and epoch+1 == 1:
        NaN = True
        print('NaN from start, no best losses')
    else:
        csvwriter.writerow(['', 'Best valid loss',lowest_valid_loss, 'Best valid error', lowest_valid_error])

        print('Finished Training \nFinal training loss: {:.4f} \tFinal valid loss: {:.4f}\tBest valid loss: {:.4f} \tEpochs before overfitting {}\n'.format(train_epoch_loss, valid_epoch_loss, min(valid_losses), lowest_valid_epoch))

    final_epoch = epoch

    return train_losses, valid_losses, train_errors, valid_errors, same_epoch_valid_loss, same_epoch_valid_error, lowest_valid_epoch, final_epoch


def plot_losses(n_epochs, train_losses, valid_losses):
    fig = plt.figure()
    epochs = np.arange(1, n_epochs+2)
    plt.plot(epochs, train_losses, label='train loss')
    plt.plot(epochs, valid_losses, label='valid loss')
    plt.xlabel('epochs')
    plt.legend(loc=2)
    plt.title('Train loss vs. Valid loss')
    plt.grid()
    plt.savefig(file_name+'_LOSS.pdf')
    plt.show()
    print('Plotted losses')


def plot_errors(n_epochs, train_errors, valid_errors):
    fig = plt.figure()
    epochs = np.arange(1, n_epochs+2)
    plt.plot(epochs, train_errors, label='train error')
    plt.plot(epochs, valid_errors, label='valid error')
    plt.xlabel('epochs')
    plt.legend(loc=0)
    plt.title('Train error vs. Valid error')
    plt.grid()
    plt.savefig(file_name+'_ERROR.pdf')
    plt.show()
    print('Plotted errors')


def check_accuracy(loader, model, conf_mat_file, dset):
    '''Inference on trained model, save confusion matrix to file'''
    running_samples = 0
    running_loss = 0
    running_corrects = 0

    with open(conf_mat_file, 'a') as txt:
        targets_list = []
        predictions_list = []

        model.eval()

        print(f"\nChecking accuracy on {dset.upper()} data")
        txt.write(f"{dset}\n\n")

        with torch.no_grad():
            for data, targets in loader:
                data = data.float().squeeze_(0).to(device=device)
                targets = targets.long().squeeze_(0).squeeze_(-1).to(device=device)

                scores = model(data)
                loss = criterion(scores, targets)
                _, predictions = scores.max(1)

                running_samples += predictions.size(0) # number of utts, not number of batches
                running_loss += loss.item()
                running_corrects += torch.sum(predictions == targets.data)

                # make targets and predictions lists
                for i in range(targets.size(0)):
                    targets_list.append(int(targets[i].item()))

                for i in range(predictions.size(0)):
                    predictions_list.append(int(predictions[i].item()))

            avg_loss = running_loss / len(loader.dataset) # number of batches in dataset
            sk_acc_num = accuracy_score(targets_list, predictions_list, normalize=False) # num correct
            sk_acc = accuracy_score(targets_list, predictions_list, normalize=True)*100 # fraction

            print(f"SK acc: Got {sk_acc_num} / {running_samples} with accuracy {sk_acc}, error {100-sk_acc}, loss {avg_loss}")

            # make confusion matrix
            conf_mat = confusion_matrix(targets_list, predictions_list)

            # get F1 scores
            f1_weighted = f1_score(targets_list, predictions_list, average='weighted') # accounts for imbalance
            f1_macro = f1_score(targets_list, predictions_list, average='macro') # metrics for each label
            f1_micro = f1_score(targets_list, predictions_list, average='micro') # metrics globally, total Tpos, Tneg

            print('F1 scores: weighted {:.2f}, macro {:.2f}, micro {:.2f}'.format(f1_weighted, f1_macro, f1_micro))
            txt.write('F1 scores: weighted {:.2f}, macro {:.2f}, micro {:.2f}\n'.format(f1_weighted, f1_macro, f1_micro))

            # write to file
            class_header = [str(i) for i in range(num_classes)]
            # class_header = [str(i) for i in target_set_final]
            class_header = 'Class num: ' + '\t'.join(class_header)# + '\n'
            print(class_header)
            txt.write(class_header+ '\n')

            targ_counts = [targets_list.count(i) for i in range(num_classes)]
            str_targ_counts = [str(int) for int in targ_counts]
            str_targ_counts = 'Targets:  ' + '\t'.join(str_targ_counts)# + '\n'
            print(str_targ_counts)
            txt.write(str_targ_counts+ '\n')

            pred_counts = [predictions_list.count(i) for i in range(num_classes)]
            str_pred_counts = [str(int) for int in pred_counts]
            str_pred_counts = 'Predicted:' + '\t'.join(str_pred_counts)# + '\n'
            print(str_pred_counts)
            txt.write(str_pred_counts+ '\n')

            print(conf_mat)
            b = np.matrix(conf_mat)
            np.savetxt(txt, b, fmt="%d")
            txt.write('\n\n')

    return conf_mat

def plot_conf_mat(mat, dset, labels, title=""):
    plt.figure(figsize=(7,7))
    ax = plt.gca()
    plt.imshow(mat,cmap="summer")
    for i in range(len(mat)):
        for j in range(len(mat)):
            plt.text(i,j,"{:d}".format(mat[i,j]),horizontalalignment="center",verticalalignment="center")
    plt.text(-1, len(mat)/2-0.5, "Predicted",horizontalalignment="center",verticalalignment="center", rotation='vertical')
    plt.text(len(mat)/2-0.5, len(mat), "Ground Truth",horizontalalignment="center",verticalalignment="center")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_name+f'_HeatMap6x6Narrow{dset}.pdf')
    plt.show()



"""################################ MAIN ################################
(if inference only, change train = False and add model load path in hparams)"""
if __name__ == "__main__":

    # Create and load datasets
    traindata, validdata, testdata = create_dataset(hdf_file)
    train_loader, valid_loader, test_loader = load_datasets()
    # Load model
    model, criterion, optimizer = load_model(model_arch)
    print(f"Loaded model and datasets")


    # Log info
    file_name = get_save_file_info()
    model_save_path = f'{file_name}.pt'
    model_save_path_LOSS = f'{file_name}_BEST_LOSS.pt'
    model_save_path_ERROR = f'{file_name}_BEST_ERROR.pt'

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'Model has {pytorch_total_params} parameters')

    param_headers = ['num_classes', 'batch_size', 'lr', 'epochs', 'num_layers',
        'nhu_lstm', 'nhu_linears', 'dropout', 'total params', 'model', 'optimiser']
    params = [num_classes, batch_size, lr, maxiter, num_layers, nhu, nhu2, dropout,
        pytorch_total_params, model_arch, opt]
    results_headers = ['epoch', 'train_loss', 'valid_loss', 'train_error', 'valid_error', 'time']
    print(f"Model hyperparameters: {params}")


    # TRAIN MODEL
    if train == True:
        print('Training network')
        with open(file_name+'.csv', 'w') as csvfile:
            print('Writing to '+file_name)
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(param_headers)
            csvwriter.writerow(params)
            csvwriter.writerow(['feature file', hdf_file])
            csvwriter.writerow(results_headers)

            train_losses, valid_losses, train_errors, valid_errors, same_epoch_valid_loss, \
            same_epoch_valid_error, lowest_valid_epoch, final_epoch = train_model(train_loader, valid_loader, csvfile)

        # SUMMARY CSV - log only most important info for each experiment
        with open('../experiments/results_summary.csv', 'a') as summaryfile:
            csvwriter2 = csv.writer(summaryfile)
            csvwriter2.writerow([min(valid_losses), same_epoch_valid_error,
            lowest_valid_epoch, num_classes, batch_size, opt, lr, maxiter, num_layers,
            nhu, nhu2, pytorch_total_params, model_arch, dropout, extra, file_name, hdf_file])
        print('Wrote to results_summary.csv')

        plot_losses(final_epoch, train_losses, valid_losses)
        plot_errors(final_epoch, train_errors, valid_errors)


    if inference == True:
        # load model with lowest loss from file:
        best_loss_model = model # initialise to get right shape (LSTM hparams)
        best_loss_model.load_state_dict(torch.load(model_save_path_LOSS)) # load in states
        # load model with lowest error from file:
        best_error_model = model
        best_error_model.load_state_dict(torch.load(model_save_path_ERROR))

        conf_mat_file = file_name + '_CONF_MAT.txt'

        with open(conf_mat_file, 'w') as txt:
            txt.write('Used features from ' + hdf_file + '\n')
            txt.write('BEST LOSS MODEL\n')
            txt.write('\nAccuracy check results:\n\n')

        # save error, lowest_valid_loss, conf_mat to txt file (also write all hyperparams)
        conf_mat_train = check_accuracy(train_loader, best_loss_model, conf_mat_file, "train")
        conf_mat_valid = check_accuracy(valid_loader, best_loss_model, conf_mat_file, "valid")
        # conf_mat_test = check_accuracy(test_loader, best_loss_model, conf_mat_file, "test")

        # make heat map of confusion matrices
        plot_conf_mat(conf_mat_train, "train", broad_labels, "Confusion matrix of original classes")
        plot_conf_mat(conf_mat_valid, "valid", broad_labels, "Confusion matrix of original classes")

        if best_loss_model != best_error_model: # if a different model had lower error
            with open(conf_mat_file, 'a') as txt:
                txt.write('Used features from ' + hdf_file + '\n')
                txt.write('\n\nBEST ERROR MODEL\n')
                txt.write('\nAccuracy check results:\n\n')

            conf_mat_train2 = check_accuracy(train_loader, best_error_model, conf_mat_file, "train")
            conf_mat_valid2 = check_accuracy(valid_loader, best_error_model, conf_mat_file, "valid")
            # conf_mat_test2 = check_accuracy(test_loader, best_error_model, conf_mat_file, "test")

            plot_conf_mat(conf_mat_train, "train", broad_labels, "Confusion matrix of original classes")
            plot_conf_mat(conf_mat_valid, "valid", broad_labels, "Confusion matrix of original classes")

            print("Wrote to {}".format(conf_mat_file))
        else:
            pass

    print("Finished.")
