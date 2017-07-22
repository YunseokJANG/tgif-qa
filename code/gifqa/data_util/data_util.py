#-*- coding: utf-8 -*-
import numpy as np
import re


def clean_str(string, downcase=True):
    """
    Tokenization/string cleaning for strings.
    Taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`(_____)]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower() if downcase else string.strip()

def recover_word(string):
    string = re.sub(r" \'s", "\'s", string)
    string = re.sub(r" ,", ",", string)
    return string

def clean_blank(blank_sent):
    clean_sent = clean_str(blank_sent).split()
    return ['<START>' if x == '_____' else x for x in clean_sent]


def clean_root(string):
    """
    Remove unexpected character in root.
    """
    return string


def pad_sequences(sequences, pad_token="[PAD]", pad_location="LEFT", max_length=None):
    """
    Pads all sequences to the same length. The length is defined by the longest sequence.
    Returns padded sequences.
    """
    if not max_length:
        max_length = max(len(x) for x in sequences)

    result = []
    for i in range(len(sequences)):
        sentence = sequences[i]
        num_padding = max_length - len(sentence)
        if num_padding == 0:
            new_sentence = sentence
        elif num_padding < 0:
            new_sentence = sentence[:num_padding]
        elif pad_location == "RIGHT":
            new_sentence = sentence + [pad_token] * num_padding
        elif pad_location == "LEFT":
            new_sentence = [pad_token] * num_padding + sentence
        else:
            print("Invalid pad_location. Specify LEFT or RIGHT.")
        result.append(new_sentence)
    return result


def convert_sent_to_index(sentence, word_to_index):
    """
    Convert sentence consisting of string to indexed sentence.
    """
    return [word_to_index[word] if word in word_to_index.keys() else 0 for word in sentence]


def batch_iter(data, batch_size, seed=None, fill=True):
    """
    Generates a batch iterator for a dataset.
    """
    random = np.random.RandomState(seed)
    data_length = len(data)
    num_batches = int(data_length / batch_size)
    if data_length % batch_size != 0:
        num_batches += 1
    # Shuffle the data at each epoch
    shuffle_indices = random.permutation(np.arange(data_length))
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_length)
        selected_indices = shuffle_indices[start_index:end_index]
        # If we don't have enough data left for a whole batch, fill it
        # randomly
        if fill is True and end_index >= data_length:
            num_missing = batch_size - len(selected_indices)
            selected_indices = np.concatenate([selected_indices, random.randint(0, data_length, num_missing)])
        yield [data[i] for i in selected_indices]


def fsr_iter(fsr_data, batch_size, random_seed=42, fill=True):
    """
    fsr_data : one of LSMDCData.build_data(), [[video_features], [sentences], [roots]]
    return per iter : [[feature]*batch_size, [sentences]*batch_size, [roots]*batch]

    Usage:
        train_data, val_data, test_data = LSMDCData.build_data()
        for features, sentences, roots in fsr_iter(train_data, 20, 10):
            feed_dict = {model.video_feature : features,
                         model.sentences : sentences,
                         model.roots : roots}
    """

    train_iter = batch_iter(list(zip(*fsr_data)), batch_size, fill=fill, seed=random_seed)
    return map(lambda batch: zip(*batch), train_iter)


def preprocess_sents(descriptions, word_to_index, max_length):

    descriptions = [clean_str(sent).split() for sent in descriptions]
    descriptions = pad_sequences(descriptions, max_length=max_length)
    # sentence를 string list 에서 int-index list로 바꿈.
    descriptions = [convert_sent_to_index(sent, word_to_index) for sent in descriptions]

    return descriptions
    # remove punctuation mark and special chars from root.


def preprocess_roots(roots, word_to_index):

    roots = [clean_root(root) for root in roots]
    # convert string to int index.
    roots = [word_to_index[root] if root in word_to_index.keys() else 0 for root in roots]

    return roots


def pad_video(video_feature, dimension):
    '''
    Fill pad to video to have same length.
    Pad in Left.
    video = [pad,..., pad, frm1, frm2, ..., frmN]
    '''
    padded_feature = np.zeros(dimension)
    max_length = dimension[0]
    current_length = video_feature.shape[0]
    num_padding = max_length - current_length
    if num_padding == 0:
        padded_feature = video_feature
    elif num_padding < 0:
        steps = np.linspace(0, current_length, num=max_length, endpoint=False, dtype=np.int32)
        padded_feature = video_feature[steps]
    else:
        padded_feature[num_padding:] = video_feature

    return padded_feature


def fill_mask(max_length, current_length, zero_location='LEFT'):
    num_padding = max_length - current_length
    if num_padding <= 0:
        mask = np.ones(max_length)
    elif zero_location == 'LEFT':
        mask = np.ones(max_length)
        for i in range(num_padding):
            mask[i] = 0
    elif zero_location == 'RIGHT':
        mask = np.zeros(max_length)
        for i in range(current_length):
            mask[i] = 1

    return mask
