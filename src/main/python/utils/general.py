
import numpy as np
from scipy.stats import t as qt
import os
import io
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import argparse
from os import path
import copy
import pickle
import gzip

def print_stats(folds, micro_list, macro_list):
    #print(micro_list)
    med_mic = np.mean(micro_list)*100
    error_mic = abs(qt.isf(0.975, df=(folds-1))) * \
        np.std(micro_list, ddof=1)/np.sqrt(len(micro_list))*100
    med_mac = np.mean(macro_list)*100
    error_mac = abs(qt.isf(0.975, df=(folds-1))) * \
        np.std(macro_list, ddof=1)/np.sqrt(len(macro_list))*100
    print("Micro\tMacro")
    print("{:.2f}({:.2f})\t{:.2f}({:.2f})".format(
        med_mic, error_mic, med_mac, error_mac))
    return med_mic, error_mic, med_mac, error_mac


def print_in_file(msg, filename):
    with open(filename, 'a') as arq:
        arq.write(msg+"\n")


def get_data(inputdir, f):

    X_train, y_train = load_svmlight_file(
        inputdir+"train"+str(f)+".gz", dtype=np.float64)
    X_test, y_test = load_svmlight_file(
        inputdir+"test"+str(f)+".gz", dtype=np.float64)

    # Same vector size
    if (X_train.shape[1] > X_test.shape[1]):
        X_test, y_test = load_svmlight_file(
            inputdir+"test"+str(f)+".gz", dtype=np.float64, n_features=X_train.shape[1])
    elif (X_train.shape[1] < X_test.shape[1]):
        X_train, y_train = load_svmlight_file(
            inputdir+"train"+str(f)+".gz", dtype=np.float64, n_features=X_test.shape[1])

    n_classes = int(max(np.max(y_train), np.max(y_test)))+1

    return X_train, y_train, X_test, y_test, n_classes


def get_y_train(args, train_idx):
    with open(os.path.join(args.splitdir, 'score.txt'), 'r') as arq:
        y = np.array(list(map(str.rstrip, arq.readlines())))
    y_train = y[train_idx]
    return y_train


def get_array(X, idxs):
    return [X[idx] for idx in idxs]


def readfile(filename):
    with io.open(filename, 'rt', newline='\n', encoding='utf8', errors='ignore') as filein:
        return filein.readlines()


def load_splits_ids(folddir):
    splits = []
    with open(folddir, encoding='utf8', errors='ignore') as fileout:
        for line in fileout.readlines():
            train_index, test_index = line.split(';')
            train_index = list(map(int, train_index.split()))
            test_index = list(map(int, test_index.split()))
            splits.append((train_index, test_index))
    return splits


def save_splits_ids(splits, folddir):
    with open(folddir, 'w', encoding='utf8', errors='ignore') as fileout:
        for train_index, test_index in splits:
            line = ' '.join(list(map(str, train_index))) + ';' + \
                ' '.join(list(map(str, test_index))) + '\n'
            fileout.write(line)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def createPath(p):
	if not path.exists(p):
		os.makedirs(p)

def load_splits_ids_for_is(args, DATAIN):
	old_splits = load_splits_ids(f"{DATAIN}/split_{args.nfolds}.csv")
	is_splits  = load_splits_ids(f"{args.path_selection}/{args.dataset}/split_{args.nfolds}_{args.ismethod}_idxinfold.csv")
	splits = []
	for f in range(args.nfolds):
		old_train, test_index = old_splits[f]
		new_train, _ = is_splits[f]
		train_index = [old_train[t] for t in new_train]

		splits.append( (train_index, test_index) )
	return splits

def get_splits(splits_filename):

    with open(splits_filename, "rb") as splits_file:
        return pickle.load(splits_file)


def checkpoint_splits(splits_df, filename):

    with open(filename, "wb") as split_file:
        pickle.dump(splits_df, split_file)



def translate_train_idxinfold(is_splits, old_splits):

    splits_to_save_translated = copy.copy(is_splits)

    nfolds = splits_to_save_translated.shape[0]
    
    for f in range(nfolds):
        old_train_idxs = old_splits.loc[f].train_idxs
        new_train_idxs = is_splits.loc[f].train_idxs

        train_index = [old_train_idxs[t] for t in new_train_idxs]

        splits_to_save_translated.loc[f].train_idxs = train_index

    return splits_to_save_translated



def load_splits_ids(folddir):
    splits = []
    with open(folddir, encoding='utf8', errors='ignore') as fileout:
        for line in fileout.readlines():
            train_index, test_index = line.split(';')
            train_index = list(map(int, train_index.split()))
            test_index = list(map(int, test_index.split()))
            splits.append((train_index, test_index))
    
    return splits