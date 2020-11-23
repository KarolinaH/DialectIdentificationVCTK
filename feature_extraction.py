import wave
import sox
from python_speech_features import logfbank
from python_speech_features import mfcc
from python_speech_features import delta
import numpy as np
import scipy.io.wavfile as wav
import os
import pickle
import h5py
import datetime
import random
import csv
import math
from collections import defaultdict

def build_labels_dict(labels_csv, cat_breadth):#, category_breadth):
    """Builds dictionary of speaker:label. Category breadth: 'narrow' is highly specific, 'broad' is
    geographically less specific, 'original' is just 3 dialects but ignores much available data"""
    print('\nBuilding labels dictionary...')

    label_cat = {'narrow':5, 'broad':6, 'original':7}
    parent_labels = {'narrow':{'EngN':0, 'EngMid':1, 'EngS':2, 'Welsh':3, 'IreUlst':4, 'Irish':5, 'ScotHigh':6,
                               'ScotLow':7, 'USN':8, 'USE':9, 'USS':10, 'USW-Can':11, 'USMid':12, 'USNYC':13,
                               'Aus':14, 'NZ':15, 'Indian':16, 'SA':17},
                     'broad': {'ENG-W':0, 'SCT-IRE':1, 'NA':2, 'OCE':3, 'IND':4, 'SA':5},
                     'original':{'GB-EAW': 0, 'GB-SCT': 1, 'US': 2}}
    labels_nums_dict = parent_labels[cat_breadth]

    label2speaker = defaultdict(list)
    labels_dict = {}
    lines = []
    labels_set = set()
    with open(labels_csv, 'r') as f:
        for line in f:
            line = line[:-1] # remove '\n'
            line = line.split(',')
            lines.append(line)

        del lines[0] # remove header line
        random.seed(1) # shuffle in case aligner document is ordered by location, so test and valid sets
        random.shuffle(lines) # made later don't have speakers from just one place for each dialect label

        label_freq = {} # get frequency of each label
        for line in lines:
            speaker = 'p'+line[0] # to match VCTK format e.g. p225
            if line[label_cat[cat_breadth]]:
                label = line[label_cat[cat_breadth]]
                label = labels_nums_dict[label]
                labels_dict[speaker] = label
                label2speaker[label].append(speaker)

                if (label in label_freq):
                    label_freq[label] += 1
                else:
                    label_freq[label] = 1

    print('FINISHED LABELS DICTIONARY')

    return labels_dict, labels_nums_dict, label_freq, label2speaker


def build_durations_dict(wav_path='../data/wav16_all/', pickle_file=None, sample_rate=None):
    """Collects duration of each wav file in wav_path. If speaker of wav file does not have a label in labels_doc,
    all files by this speaker are omitted to avoid wasted computation (measuring duration, extracting features))"""

    print('Building durations dictionary...')
    path = wav_path
    missing = set()
    utt2dur_dict = {} # dur of individual utterances
    speaker_durs = {}  # speaker : total duration for that speaker

    for speaker in os.listdir(path):
        speaker_sum = 0  # start sum of durs for specific speaker
        if speaker in labels_dict:
            # check that speaker has a label - if not, omit to avoid wasted computation later
            speaker_path = os.path.join(path, speaker)
            for file in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, file)
                # duration = sox.file_info.duration(file_path)
                with wave.open(file_path, "rb") as f:
                    frames = f.getnframes()
                    if sample_rate is None:
                        rate = f.getframerate()
                    duration = frames / float(sample_rate)

                    utt2dur_dict[file] = duration
                    speaker_sum += duration
        else:
            if speaker not in missing:
                missing.add(speaker)  # collect speakers without labels

        speaker_sum = round(speaker_sum, 3)
        if speaker not in missing:
            speaker_durs[speaker] = speaker_sum

    print(f"The following {len(missing)} speakers have no labels and have been omitted: {missing}")

    if pickle_file is not None:
        pickle.dump(utt2dur_dict, open(pickle_file, "wb"))
        print(f"FINISHED AND PICKLED in {pickle_file}")
    else:
        print('FINISHED DURATIONS DICTIONARY')

    total_dur = round(sum(speaker_durs.values()), 3)
    total_dur_str = str(datetime.timedelta(seconds=round(total_dur)))
    print('TOTAL DUR OF WAVS:', total_dur, 'seconds', total_dur_str)

    return utt2dur_dict, speaker_durs, total_dur


def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]



def new_split_sets(label2speaker):
    all_train_speakers = []
    all_valid_speakers = []
    all_test_speakers = []
    utt2dur_train, utt2dur_valid, utt2dur_test = {}, {}, {}
    for label, speakers in label2speaker.items():
        n_speakers = len(speakers)
        if n_speakers > 1:
            # ordered to prioritise a large train set
            train_split = int(n_speakers // 1.25) # 80%, no partial speakers
            valid_split = math.ceil((n_speakers - train_split) / 2) # upper ceiling - overlap rather than splitting
            test_split = valid_split #- 1 # indexing from end to allow overlap

            train_speakers = speakers[:train_split]
            valid_speakers = speakers[train_split : train_split + valid_split]
            test_speakers = speakers[train_split + test_split : ] # index to end
            if not test_speakers: # if no speakers left for test, overlap with valid
                test_speakers = valid_speakers # limits overlap to only when test would otherwise be empty
        elif n_speakers == 1:
            train_split, valid_split, test_split = 1, 0, 0
            train_speakers = speakers[:] # better to have more data, even if we don't test it
            valid_speakers = []
            test_speakers = []

        all_train_speakers.extend(train_speakers)
        all_valid_speakers.extend(valid_speakers)
        all_test_speakers.extend(test_speakers)

    print('train', all_train_speakers)
    print('valid', all_valid_speakers)
    print('test', all_test_speakers)

    #for split_set in [all_train_speakers, all_valid_speakers, all_test_speakers]
    for speaker in all_train_speakers:
        for utt in utt2dur_dict.keys():
            if utt[:4] == speaker:
                utt2dur_train[utt] = utt2dur_dict[utt]

    for speaker in all_valid_speakers:
        for utt in utt2dur_dict.keys():
            if utt[:4] == speaker:
                utt2dur_valid[utt] = utt2dur_dict[utt]

    for speaker in all_test_speakers:
        for utt in utt2dur_dict.keys():
            if utt[:4] == speaker:
                utt2dur_test[utt] = utt2dur_dict[utt]


    return utt2dur_train, utt2dur_valid, utt2dur_test


def group_utts_by_duration(utt2dur, print_info=False):
    ## TAKE self.sorted_utt2dur_dict ??
    """Sorts utt2dur list from LONGEST to SHORTEST, Thus the first element of its keys/values list will be
    the longest of all. We use this as our starting point and set a minimum at 90% of the longest length.
    In each loop, we compare the first element with the minimum and if it fits within the group's range, we append it to
    the group and remove it from the list, making the next longest element the new list[0] and its 90% the new minimum.
    We repeat this until the duration falls below the minimum, at which point we create a new group and repeat the process
    until all utterances have been grouped."""

    print('Beginning utterance grouping by duration...')
    sorted_utt2dur_dict = {k: v for k, v in sorted(utt2dur.items(), key=lambda item: item[1], reverse=True)}
    pop_dur_list = []  # list of sorted durations to pop from in grouping
    pop_utt_list = []  # list of utterances sorted by corresponding duration, to pop from group
    groups_dict = {}  # groups dict will be group_name : [list of utterances of similar duration]
    i = 0  # i for group naming

    for utt, dur in sorted_utt2dur_dict.items():
        pop_utt_list.append(utt)
        pop_dur_list.append(dur)

    while len(pop_utt_list) >= 1:
        longest = pop_dur_list[
            0]  # first element of list is the longest (of remaining elements not yet popped from list)
        minimum = longest * 0.9
        group_name = f"group_{i}"
        groups_dict[group_name] = []

        try:
            while pop_dur_list[0] >= minimum:
                groups_dict[group_name].append(pop_utt_list[0])  # append utt to group members list
                ## remove utt and dur we just appended from both lists to make next longest utt list[0]
                pop_utt_list.pop(0)
                pop_dur_list.pop(0)
        except:
            pass  # we have reached the end - no more audio files

        i += 1
        if print_info == True:
            print(f"\nNew group: {group_name}")
            print(f"remaining utts to be grouped: {len(pop_utt_list)}")
            print(f"longest in group: {longest}, \tminimum (90% of longest): {minimum}")
            print(f"Final size of {group_name} : {len(groups_dict[group_name])}")

    print('FINISHED GROUPING')
    return groups_dict


def pad_with_silence(orig_dir_path='../data/wav16_all/', padded_dir_path='../data/padded_wav16/', utt2dur=None, print_info=False):
    """Identifies longest element in each group as grouped by function [GROUP_FUNCTION_NAME] and sets its duration as
    the target duration for the group. Gets duration of each element and adds silence to pad difference. Creates a new
    file in the given destination (original + padding) - will give OVERWRITE WARNING if code rerun for same destination"""

    print('Beginning padding - this may take a while...')
    ## make new directory for padded wav files - pass if already created
    try:
        os.mkdir(padded_dir_path)
    except:
        pass

    for group in groups_dict:
        ## make new group directory - pass if already created
        try:
            os.mkdir(padded_dir_path + group)
        except:
            pass

        ## find longest wav in group, set as target to pad to - always the first element due to sorted nature of grouping
        longest_in_group = groups_dict[group][0]  #
        target_dur = utt2dur[longest_in_group]  # duration of longest_in_group

        if print_info == True:
            print('\n' + group)
            print('target duration: {:0.4f}'.format(target_dur))

        for file in groups_dict[group]:
            pre_padded_dur = utt2dur[file]
            pad_dur = (target_dur - pre_padded_dur)  # difference from target

            file_dir = file[:4]
            orig_wav = os.path.join(orig_dir_path, file_dir, file)
            padded_wav = os.path.join(padded_dir_path, group, file)

            ## make sox Transformers, add padding, create padded file (will give OVERWRITE warning if already run)
            tfm = sox.Transformer()
            tfm.pad(end_duration=pad_dur)
            tfm.build(orig_wav, padded_wav)

    print(f"FINISHED PADDING, padded files in {padded_dir_path}")


def extract_feat_and_label(feat_type, padded_dir_path, pickle_features_file=None, print_info=False):
    '''This function extracts Mel filterbank  or MFCC features, iterating through files
    in each group. Each file's speaker info is extracted and used to find the speaker's
    dialect label, which is used to create a label vector (all zeros except the first value,
    which is the label number). This label vector is appended as the final column of the feature array.'''
    features = {}

    print('Beginning feature extraction - this may take a while...')

    for group in os.listdir(padded_dir_path): # padded_wav16_mfcc/
        if print_info == True:
            print(f"Extracting features from {group}, {len(group)} files")
        for file in os.listdir(os.path.join(padded_dir_path,group)):
            speaker = file[0:4]
            label = labels_dict[speaker]

            ## extract features to np array
            (rate,sig) = wav.read(os.path.join(padded_dir_path,group,file))

            if feat_type == 'fbank':
                feat = logfbank(sig, rate, winlen=0.025,winstep=0.01,
                  nfilt=40,nfft=512,lowfreq=0,highfreq=None,preemph=0.97)

            elif feat_type == 'mfcc':
                mfcc_feat = mfcc(sig, rate, appendEnergy=True) # try appendEnergy = True, False
                # Deltas
                d_mfcc_feat = delta(mfcc_feat, 4)
                # Deltas-Deltas
                dd_mfcc_feat = delta(d_mfcc_feat, 4)
                # transpose
                mfcc_feat = np.transpose(mfcc_feat)
                d_mfcc_feat = np.transpose(d_mfcc_feat)
                dd_mfcc_feat = np.transpose(dd_mfcc_feat)
                # concatenate above three features
                concat_feat = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat))
                feat = np.transpose(concat_feat)

            ## create label vector
            label_vector = np.zeros((feat.shape[0], feat.shape[1]+1))
            label_vector[0,-1] = label # replace 0 in first row of last column with speaker label
            label_vector[:,:-1] = feat # replace all values except those of label column with fbank features
            feat_lab = label_vector # create final array with features and label column

            features[file[0:8]] = feat_lab

    if pickle_features_file is not None:
        pickle.dump(features, open(pickle_features_file, "wb"))
        print(f"Pickled features dict in {pickle_features_file}")

    print('FINISHED EXTRACTING FEATURE + LABEL MATRICES')
    return features


def save_features_to_hdf5(hdf5_file_name, features, print_info=False):
    '''This function creates an HDF5 file and saves the extracted features in hdf5_file/group/wav_file_name/features.
    Each dataset (fbank features array) has label and speaker attributes (metadata).'''

    print('Beginning saving to HDF5 file...')
    f = h5py.File(hdf5_file_name, "w")
    with h5py.File(hdf5_file_name, "a") as f:
        for group in groups_dict.keys():
            hdf_grp = group

            for file in groups_dict[group]:
                dset_name = file[:8]  # or file [:8]
                fbank_and_lab = features[dset_name]
                dset = f.create_dataset(group + '/' + dset_name, data=fbank_and_lab)
                dset.attrs['label'] = labels_dict[file[:4]]
                dset.attrs['speaker'] = file[:4]

            if print_info == True:
                print(f"group {group} finished")

    print(f"FINISHED SAVING TO HDF5 FILE: {hdf5_file_name}\n")



"""################################ MAIN ################################"""
if __name__ == "__main__":

    dialect_csv = "../data/dialect_splits3.csv"
    dialect_split = "original"
    data_path = "../data/wav16_all/"
    padded_data_path = "../data/padded_wav16_DELETE/"
    try:
        os.mkdir(padded_data_path) # if doesn't exist yet
    feat_type = "fbank"
    hdf_save_file = "MFB.hdf5"

    labels_dict, labels_nums_dict, label_freq, label2speaker = build_labels_dict(dialect_csv, dialect_split)
    utt2dur_dict, speaker_durs, total_dur = build_durations_dict(data_path, sample_rate=16000)
    utt2dur_train, utt2dur_valid, utt2dur_test = new_split_sets(label2speaker)
    print(f"Finished splitting \ttest: {len(utt2dur_test)}\tvalid: {len(utt2dur_valid)}\ttrain: {len(utt2dur_train)}\n")

    dset_dict = {"train":utt2dur_train, "valid":utt2dur_valid, "test":utt2dur_test}
    for dset in ["train", "valid", "test"]:
        groups_dict = group_utts_by_duration(dset_dict[dset])
        pad_with_silence(data_path, f"{padded_data_path}{dset}_padded_wav16/", dset_dict[dset])
        features = extract_feat_and_label(feat_type, padded_dir_path=f"{padded_data_path}{dset}_padded_wav16/")
        save_features_to_hdf5(f"{dset}_{hdf_save_file}", features)

    print("Finished feature extraction")
