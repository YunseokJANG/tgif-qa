import numpy as np
#from gifqa.util import log
from util import log

import os.path
import sys
import random
import h5py
import itertools
import re
import tqdm

import pandas as pd
import data_util
import hickle as hkl
import cPickle as pkl

from IPython import embed

__PATH__ = os.path.abspath(os.path.dirname(__file__))

def assert_exists(path):
    assert os.path.exists(path), 'Does not exist : {}'.format(path)

# PATHS
TGIF_DATA_DIR = os.path.normpath(os.path.join(__PATH__, '../../dataset/tgif'))
TYPE_TO_CSV = {'FrameQA': 'Train_frameqa_question.csv',
               'Count': 'Train_count_question.csv',
               'Trans': 'Train_transition_question.csv',
               'Action' : 'Train_action_question.csv'}
assert_exists(TGIF_DATA_DIR)

VIDEO_FEATURE_DIR = os.path.join(TGIF_DATA_DIR, 'features')
assert_exists(VIDEO_FEATURE_DIR)

eos_word = '<EOS>'

class DatasetTGIF():
    '''
    API for TGIF dataset
    '''
    def __init__(self,
                 dataset_name='train',
                 image_feature_net='resnet',
                 layer='pool5',
                 max_length=80,
                 use_moredata=False,
                 max_n_videos=None,
                 data_type=None,
                 dataframe_dir=None,
                 vocab_dir=None):
        self.dataframe_dir = dataframe_dir
        self.vocabulary_dir = vocab_dir
        self.use_moredata = use_moredata
        self.dataset_name = dataset_name
        self.image_feature_net = image_feature_net
        self.layer = layer
        self.max_length = max_length
        self.max_n_videos = max_n_videos
        self.data_type = data_type
        self.data_df = self.read_df_from_csvfile()

        if max_n_videos is not None:
            self.data_df = self.data_df[:max_n_videos]
        self.ids = list(self.data_df.index)
        if dataset_name == 'train':
            random.shuffle(self.ids)

        self.feat_h5 = self.read_tgif_from_hdf5()

    def __del__(self):
        if self.image_feature_net.upper() == "CONCAT":
            self.feat_h5["c3d"].close()
            self.feat_h5["resnet"].close()
        else:
            self.feat_h5.close()

    def __len__(self):
        if self.max_n_videos is not None:
            if self.max_n_videos <= len(self.data_df):
                return self.max_n_videos
        return len(self.data_df)

    def read_tgif_from_hdf5(self):
        '''
        resnet  > res5c, pool5
        c3d     > fc6, conv5b
        concat  > fc, conv
        '''
        if self.image_feature_net.upper() == "CONCAT":
            if self.layer.lower() == "fc":
                c3d_file = os.path.join(VIDEO_FEATURE_DIR, "TGIF_C3D_fc6.hdf5")
                resnet_file = os.path.join(VIDEO_FEATURE_DIR, "TGIF_RESNET_pool5.hdf5")
            elif self.layer.lower() == "conv":
                c3d_file = os.path.join(VIDEO_FEATURE_DIR, "TGIF_C3D_conv5b.hdf5")
                resnet_file = os.path.join(VIDEO_FEATURE_DIR, "TGIF_RESNET_res5c.hdf5")
            return {"c3d":h5py.File(c3d_file,'r'), "resnet":h5py.File(resnet_file,'r')}
        else:
            feature_file = os.path.join(VIDEO_FEATURE_DIR, "TGIF_" + self.image_feature_net.upper() + "_" + self.layer.lower() + ".hdf5")
            assert_exists(feature_file)
            log.info("Load %s hdf5 file: %s", self.image_feature_net.upper(), feature_file)
            return h5py.File(feature_file, 'r')

    def read_df_from_csvfile(self):
        assert self.data_type in ['FrameQA', 'Count', 'Trans', 'Action'], 'Should choose data type '

        if self.data_type == 'FrameQA':
            train_data_path = os.path.join(self.dataframe_dir, 'Train_frameqa_question.csv')
            test_data_path = os.path.join(self.dataframe_dir, 'Test_frameqa_question.csv')
            self.total_q = pd.DataFrame().from_csv(os.path.join(self.dataframe_dir,'Total_frameqa_question.csv'), sep='\t')
        elif self.data_type == 'Count':
            train_data_path = os.path.join(self.dataframe_dir, 'Train_count_question.csv')
            test_data_path = os.path.join(self.dataframe_dir, 'Test_count_question.csv')
            self.total_q = pd.DataFrame().from_csv(os.path.join(self.dataframe_dir,'Total_count_question.csv'), sep='\t')
        elif self.data_type == 'Trans':
            train_data_path = os.path.join(self.dataframe_dir, 'Train_transition_question.csv')
            test_data_path = os.path.join(self.dataframe_dir, 'Test_transition_question.csv')
            self.total_q = pd.DataFrame().from_csv(os.path.join(self.dataframe_dir,'Total_transition_question.csv'), sep='\t')
        elif self.data_type == 'Action':
            train_data_path = os.path.join(self.dataframe_dir, 'Train_action_question.csv')
            test_data_path = os.path.join(self.dataframe_dir, 'Test_action_question.csv')
            self.total_q = pd.DataFrame().from_csv(os.path.join(self.dataframe_dir,'Total_action_question.csv'), sep='\t')

        assert_exists(train_data_path)
        assert_exists(test_data_path)

        if self.dataset_name == 'train':
            data_df = pd.read_csv(train_data_path, sep='\t')
        elif self.dataset_name == 'test':
            data_df = pd.read_csv(test_data_path, sep='\t')

        data_df = data_df.set_index('vid_id')
        data_df['row_index'] = range(1, len(data_df)+1) # assign csv row index
        return data_df

    @property
    def n_words(self):
        ''' The dictionary size. '''
        if not hasattr(self, 'word2idx'):
            raise Exception('Dictionary not built yet!')
        return len(self.word2idx)

    def __repr__(self):
        if hasattr(self, 'word2idx'):
            return '<Dataset (%s) with %d videos and %d words>' % (
                self.dataset_name, len(self), len(self.word2idx))
        else:
            return '<Dataset (%s) with %d videos -- dictionary not built>' % (
                self.dataset_name, len(self))

    def split_sentence_into_words(self, sentence, eos=True):
        '''
        Split the given sentence (str) and enumerate the words as strs.
        Each word is normalized, i.e. lower-cased, non-alphabet characters
        like period (.) or comma (,) are stripped.
        When tokenizing, I use ``data_util.clean_str``
        '''
        try:
            words = data_util.clean_str(sentence).split()
        except:
            print sentence
            sys.exit()
        if eos:
            words = words + [eos_word]
        for w in words:
            if not w:
                continue
            yield w

    def build_word_vocabulary(self, all_captions_source=None,
                              word_count_threshold=0,):
        '''
        borrowed this implementation from @karpathy's neuraltalk.
        '''
        log.infov('Building word vocabulary (%s) ...', self.dataset_name)

        if all_captions_source is None:
            all_captions_source = self.get_all_captions()

        # enumerate all sentences to build frequency table
        word_counts = {}
        nsents = 0
        for sentence in all_captions_source:
            nsents += 1
            for w in self.split_sentence_into_words(sentence):
                word_counts[w] = word_counts.get(w, 0) + 1

        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
        log.info("Filtered vocab words (threshold = %d), from %d to %d",
                 word_count_threshold, len(word_counts), len(vocab))

        # build index and vocabularies
        self.word2idx = {}
        self.idx2word = {}

        self.idx2word[0] = '.'
        self.idx2word[1] = 'UNK'
        self.word2idx['#START#'] = 0
        self.word2idx['UNK'] = 1
        for idx, w in enumerate(vocab, start=2):
            self.word2idx[w] = idx
            self.idx2word[idx] = w
        import cPickle as pkl
        pkl.dump(self.word2idx, open(os.path.join(self.vocabulary_dir, 'word_to_index_%s.pkl'%self.data_type), 'w'))
        pkl.dump(self.idx2word, open(os.path.join(self.vocabulary_dir, 'index_to_word_%s.pkl'%self.data_type), 'w'))

        word_counts['.'] = nsents
        bias_init_vector = np.array([1.0*word_counts[w] if i>1 else 0 for i, w in self.idx2word.iteritems()])
        bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
        bias_init_vector = np.log(bias_init_vector)
        bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
        self.bias_init_vector = bias_init_vector

        #self.total_q = pd.DataFrame().from_csv(os.path.join(dataframe_dir,'Total_desc_question.csv'), sep='\t')
        answers = list(set(self.total_q['answer'].values))
        self.ans2idx = {}
        self.idx2ans = {}
        for idx, w in enumerate(answers):
            self.ans2idx[w]=idx
            self.idx2ans[idx]=w
        pkl.dump(self.ans2idx, open(os.path.join(self.vocabulary_dir, 'ans_to_index_%s.pkl'%self.data_type), 'w'))
        pkl.dump(self.idx2ans, open(os.path.join(self.vocabulary_dir, 'index_to_ans_%s.pkl'%self.data_type), 'w'))

        # Make glove embedding.
        import spacy
        nlp = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')

        max_length = len(vocab)
        GLOVE_EMBEDDING_SIZE = 300

        glove_matrix = np.zeros([max_length,GLOVE_EMBEDDING_SIZE])
        for i in range(len(vocab)):
            w = vocab[i]
            w_embed = nlp(u'%s' % w).vector
            glove_matrix[i,:] = w_embed

        vocab = pkl.dump(glove_matrix, open(os.path.join(self.vocabulary_dir, 'vocab_embedding_%s.pkl'%self.data_type), 'w'))
        self.word_matrix = glove_matrix

    def load_word_vocabulary(self):

        word_matrix_path = os.path.join(self.vocabulary_dir, 'vocab_embedding_%s.pkl'%self.data_type)

        word2idx_path = os.path.join(self.vocabulary_dir, 'word_to_index_%s.pkl'%self.data_type)
        idx2word_path = os.path.join(self.vocabulary_dir, 'index_to_word_%s.pkl'%self.data_type)
        ans2idx_path = os.path.join(self.vocabulary_dir, 'ans_to_index_%s.pkl'%self.data_type)
        idx2ans_path = os.path.join(self.vocabulary_dir, 'index_to_ans_%s.pkl'%self.data_type)

        if not (os.path.exists(word_matrix_path) and os.path.exists(word2idx_path) and \
                os.path.exists(idx2word_path) and os.path.exists(ans2idx_path) and \
                os.path.exists(idx2ans_path)):
            self.build_word_vocabulary()

        with open(word_matrix_path, 'r') as f:
            self.word_matrix = pkl.load(f)
        log.info("Load word_matrix from pkl file : %s", word_matrix_path)

        with open(word2idx_path, 'r') as f:
            self.word2idx = pkl.load(f)
        log.info("Load word2idx from pkl file : %s", word2idx_path)

        with open(idx2word_path, 'r') as f:
            self.idx2word = pkl.load(f)
        log.info("Load idx2word from pkl file : %s", idx2word_path)

        with open(ans2idx_path, 'r') as f:
            self.ans2idx = pkl.load(f)
        log.info("Load answer2idx from pkl file : %s", ans2idx_path)

        with open(idx2ans_path, 'r') as f:
            self.idx2ans = pkl.load(f)
        log.info("Load idx2answers from pkl file : %s", idx2ans_path)


    def share_word_vocabulary_from(self, dataset):
        assert hasattr(dataset, 'idx2word') and hasattr(dataset, 'word2idx'), \
            'The dataset instance should have idx2word and word2idx'
        assert (isinstance(dataset.idx2word, dict) or isinstance(dataset.idx2word, list)) \
                and isinstance(dataset.word2idx, dict), \
            'The dataset instance should have idx2word and word2idx (as dict)'

        if hasattr(self, 'word2idx'):
            log.warn("Overriding %s' word vocabulary from %s ...", self, dataset)

        self.idx2word = dataset.idx2word
        self.word2idx = dataset.word2idx
        self.ans2idx = dataset.ans2idx
        self.idx2ans = dataset.idx2ans
        if hasattr(dataset, 'word_matrix'):
            self.word_matrix = dataset.word_matrix


    # Dataset Access APIs (batch loading, sentence etc)
    def iter_ids(self, shuffle=False):

        #if self.data_type == 'Trans':
        if shuffle:
            random.shuffle(self.ids)
        for key in self.ids:
            yield key

    def get_all_captions(self):
        '''
        Iterate caption strings associated in the vid/gifs.
        '''
        qa_data_df = pd.DataFrame().from_csv(os.path.join(self.dataframe_dir, TYPE_TO_CSV[self.data_type]), sep='\t')

        all_sents = []
        for row in qa_data_df.iterrows():
            all_sents.extend(self.get_captions(row))
        self.data_type
        return all_sents

    def get_captions(self, row):
        if self.data_type == 'FrameQA':
            columns = ['description', 'question', 'answer']
        elif self.data_type == 'Count':
            columns = ['question']
        elif self.data_type == 'Trans':
            columns = ['question', 'a1', 'a2', 'a3', 'a4', 'a5']
        elif self.data_type == 'Action':
            columns = ['question', 'a1', 'a2', 'a3', 'a4', 'a5']

        sents = [row[1][col] for col in columns if not pd.isnull(row[1][col])]
        return sents


    def load_video_feature(self, key):
        key_df = self.data_df.loc[key,'key']
        video_id = str(key_df)

        if self.image_feature_net == 'resnet':
            assert self.layer.lower() in ['pool5', 'res5c']
            video_feature = np.array(self.feat_h5[video_id])
            if self.layer.lower() == 'res5c':
                video_feature = np.transpose(
                    video_feature.reshape([-1,2048,7,7]), [0, 2, 3, 1])
                assert list(video_feature.shape[1:]) == [7, 7, 2048]
            elif self.layer.lower() == 'pool5':
                video_feature = np.expand_dims(video_feature, axis=1)
                video_feature = np.expand_dims(video_feature, axis=1)
                assert list(video_feature.shape[1:]) == [1, 1, 2048]

        elif self.image_feature_net.lower() == 'c3d':
            assert self.layer.lower() in ['fc6', 'conv5b']
            video_feature = np.array(self.feat_h5[video_id])

            if self.layer.lower() == 'fc6':
                if len(video_feature.shape) == 1:
                    video_feature = np.expand_dims(video_feature, axis=0)
                video_feature = np.expand_dims(video_feature, axis=1)
                video_feature = np.expand_dims(video_feature, axis=1)
                assert list(video_feature.shape[1:]) == [1, 1, 4096]
            elif self.layer.lower() == 'conv5b':
                if len(video_feature.shape) == 4:
                    video_feature = np.expand_dims(video_feature, axis=0)
                video_feature = np.transpose(
                    video_feature.reshape([-1,1024,7,7]), [0,2,3,1])
                assert list(video_feature.shape[1:]) == [7, 7, 1024]

        elif self.image_feature_net.lower() == 'concat':
            assert self.layer.lower() in ['fc', 'conv']
            c3d_feature = np.array(self.feat_h5["c3d"][video_id])
            resnet_feature = np.array(self.feat_h5["resnet"][video_id])
            if len(c3d_feature.shape) == 1:
                c3d_feature = np.expand_dims(c3d_feature, axis=0)
            #if len(resnet_feature.shape) == 1:
            #    resnet_feature = np.expand_dims(resnet_feature, axis=0)

            if not len(c3d_feature) == len(resnet_feature):
                max_len = min(len(c3d_feature),len(resnet_feature))
                c3d_feature = c3d_feature[:max_len]
                resnet_feature = resnet_feature[:max_len]

            if self.layer.lower() == 'fc':
                video_feature = np.concatenate((c3d_feature, resnet_feature),
                                                axis=len(c3d_feature.shape)-1)
                video_feature = np.expand_dims(video_feature, axis=1)
                video_feature = np.expand_dims(video_feature, axis=1)
                assert list(video_feature.shape[1:]) == [1, 1, 4096+2048]
            elif self.layer.lower() == 'conv':
                c3d_feature = np.transpose(c3d_feature.reshape([-1,1024,7,7]), [0,2,3,1])
                resnet_feature = np.transpose(resnet_feature.reshape([-1,2048,7,7]), [0, 2, 3, 1])
                video_feature = np.concatenate((c3d_feature, resnet_feature),
                                               axis=len(c3d_feature.shape)-1)
                assert list(video_feature.shape[1:]) == [7, 7, 1024+2048]

        return video_feature

    def get_video_feature_dimension(self):
        if self.image_feature_net == 'resnet':
            assert self.layer.lower() in ['fc1000', 'pool5', 'res5c']
            if self.layer.lower() == 'res5c':
                return (self.max_length, 7, 7, 2048)
            elif self.layer.lower() == 'pool5':
                return (self.max_length, 1, 1, 2048)
        elif self.image_feature_net.lower() == 'c3d':
            if self.layer.lower() == 'fc6':
                return (self.max_length, 1, 1, 4096)
            elif self.layer.lower() == 'conv5b':
                return (self.max_length, 7, 7, 1024)
        elif self.image_feature_net.lower() == 'concat':
            assert self.layer.lower() in ['fc', 'conv']
            if self.layer.lower() == 'fc':
                return (self.max_length, 1, 1, 4096+2048)
            elif self.layer.lower() == 'conv':
                return (self.max_length, 7, 7, 1024+2048)

    def get_video_feature(self, key):
        video_feature = self.load_video_feature(key)
        return video_feature

    def convert_sentence_to_matrix(self, sentence, eos=True):
        '''
        Convert the given sentence into word indices and masks.
        WARNING: Unknown words (not in vocabulary) are revmoed.

        Args:
            sentence: A str for unnormalized sentence, containing T words

        Returns:
            sentence_word_indices : list of (at most) length T,
                each being a word index
        '''
        sent2indices = [self.word2idx[w] if w in self.word2idx else 1 for w in
                        self.split_sentence_into_words(sentence,eos)] # 1 is UNK, unknown
        T = len(sent2indices)
        length = min(T, self.max_length)
        return sent2indices[:length]

    def get_video_mask(self, video_feature):
        video_length = video_feature.shape[0]
        return data_util.fill_mask(self.max_length, video_length, zero_location='LEFT')

    def get_sentence_mask(self, sentence):
        sent_length = len(sentence)
        return data_util.fill_mask(self.max_length, sent_length, zero_location='RIGHT')

    def get_question(self, key):
        '''
        Return question string for given key.
        '''
        question = self.data_df.loc[key, ['question','description']].values
        if len(list(question.shape)) > 1:
            question = question[0]
        question = question[0]
        return self.convert_sentence_to_matrix(question, eos=False)

    def get_question_mask(self, sentence):
        sent_length = len(sentence)
        return data_util.fill_mask(self.max_length, sent_length, zero_location='LEFT')

    def get_answer(self, key):
        answer = self.data_df.loc[key, ['answer','type']].values

        if len(list(answer.shape)) > 1:
            answer = answer[0]

        anstype = answer[1]
        answer = answer[0]

        return answer, anstype

    def get_FrameQA_result(self, chunk):
        batch_size = len(chunk)
        batch_video_feature_convmap = np.zeros(
            [batch_size] + list(self.get_video_feature_dimension()), dtype=np.float32)

        # Question, Right most aligned
        batch_question = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_question_right = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_video_mask = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_question_mask = np.zeros([batch_size, self.max_length], dtype=np.uint32)

        batch_debug_sent = np.asarray([None] * batch_size)
        batch_answer = np.zeros([batch_size, 1])
        batch_answer_type = np.zeros([batch_size, 1])
        questions = []

        for k in xrange(batch_size):
            key = chunk[k]
            video_feature = self.get_video_feature(key)
            video_mask = self.get_video_mask(video_feature)

            batch_video_feature_convmap[k, :] = data_util.pad_video(
                video_feature, self.get_video_feature_dimension())
            batch_video_mask[k] = video_mask

            answer, answer_type = self.get_answer(key)
            if str(answer) in self.ans2idx:
                answer = self.ans2idx[answer]
            else:
                # unknown token, check later
                answer = 1
            question = self.get_question(key)
            question_mask = self.get_question_mask(question)
            # Left align
            batch_question[k, :len(question)] = question
            # Right align
            batch_question_right[k, -len(question):] = question
            #questions.append(question)
            #batch_question_mask.append(len(question)) #question_mask
            batch_question_mask[k] = question_mask
            question_pad = np.zeros([self.max_length])
            question_pad[:len(question)] = question
            questions.append(question_pad)
            batch_answer[k] = answer
            batch_answer_type[k] = float(int(answer_type))
            batch_debug_sent[k] = self.data_df.loc[key, 'question']

        ret = {
            'ids': chunk,
            'video_features': batch_video_feature_convmap,
            'question_words': batch_question,
            'question_words_right': batch_question_right,
            'video_mask': batch_video_mask,
            'question_mask': batch_question_mask,
            'answer': batch_answer,
            'answer_type': batch_answer_type,
            'debug_sent': batch_debug_sent
        }
        return ret

    def get_Count_question(self, key):
        '''
        Return question string for given key.
        '''
        question = self.data_df.loc[key, 'question']
        return self.convert_sentence_to_matrix(question, eos=False)

    def get_Count_question_mask(self, sentence):
        sent_length = len(sentence)
        return data_util.fill_mask(self.max_length, sent_length, zero_location='RIGHT')

    def get_Count_answer(self, key):
        return self.data_df.loc[key, 'answer']

    def get_Count_result(self, chunk):
        batch_size = len(chunk)
        batch_video_feature_convmap = np.zeros(
            [batch_size] + list(self.get_video_feature_dimension()), dtype=np.float32)

        # Question, Right most aligned
        batch_question = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_question_right = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_video_mask = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_question_mask = np.zeros([batch_size, self.max_length], dtype=np.uint32)

        batch_debug_sent = np.asarray([None] * batch_size)
        batch_answer = np.zeros([batch_size, 1])

        for k in xrange(batch_size):
            key = chunk[k]
            video_feature = self.get_video_feature(key)
            video_mask = self.get_video_mask(video_feature)

            batch_video_feature_convmap[k, :] = data_util.pad_video(
                video_feature, self.get_video_feature_dimension())
            batch_video_mask[k] = video_mask

            answer = max(self.get_Count_answer(key), 1)

            question = self.get_Count_question(key)
            question_mask = self.get_Count_question_mask(question)
            # Left align
            batch_question[k, :len(question)] = question
            # Right align
            batch_question_right[k, -len(question):] = question
            batch_question_mask[k] = question_mask
            batch_answer[k] = answer
            batch_debug_sent[k] = self.data_df.loc[key, 'question']

        ret = {
            'ids': chunk,
            'video_features': batch_video_feature_convmap,
            'question_words': batch_question,
            'question_words_right': batch_question_right,
            'video_mask': batch_video_mask,
            'question_mask': batch_question_mask,
            'answer': batch_answer,
            'debug_sent': batch_debug_sent
        }
        return ret

    def get_Trans_dict(self, key):
        a1 = self.data_df.loc[key, 'a1'].strip()
        a2 = self.data_df.loc[key, 'a2'].strip()
        a3 = self.data_df.loc[key, 'a3'].strip()
        a4 = self.data_df.loc[key, 'a4'].strip()
        a5 = self.data_df.loc[key, 'a5'].strip()
        question = self.data_df.loc[key, 'question'].strip()
        row_index = self.data_df.loc[key, 'row_index']

        # as list of sentence strings
        candidates = [a1, a2, a3, a4, a5]
        answer = self.data_df.loc[key, 'answer']

        candidates_to_indices = [self.convert_sentence_to_matrix(question + ' ' + x)
                                 for x in candidates]
        return {
            'answer' : answer,
            'candidates': candidates_to_indices,
            'raw_sentences': candidates,
            'row_indices' : row_index,
            'question' : question
        }

    def get_Trans_matrix(self, candidates, is_left=True):
        candidates_matrix = np.zeros([5, self.max_length], dtype=np.uint32)
        for k in range(5):
            sentence = candidates[k]
            if is_left:
                candidates_matrix[k, :len(sentence)] = sentence
            else:
                candidates_matrix[k, -len(sentence):] = sentence
        return candidates_matrix

    def get_Trans_mask(self, candidates):
        mask_matrix = np.zeros([5, self.max_length], dtype=np.uint32)
        for k in range(5):
            mask_matrix[k] = data_util.fill_mask(self.max_length,
                                                 len(candidates[k]),
                                                 zero_location='RIGHT')
        return mask_matrix

    def get_Trans_result(self, chunk):
        batch_size = len(chunk)
        batch_video_feature_convmap = np.zeros(
            [batch_size] + list(self.get_video_feature_dimension()), dtype=np.float32)

        batch_candidates = np.zeros([batch_size, 5, self.max_length], dtype=np.uint32)
        batch_candidates_right = np.zeros([batch_size, 5, self.max_length], dtype=np.uint32)
        batch_answer = np.zeros([batch_size], dtype=np.uint32)

        batch_video_mask = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_candidates_mask = np.zeros([batch_size, 5, self.max_length], dtype=np.uint32)

        batch_debug_sent = np.asarray([None] * batch_size)
        batch_raw_sentences = np.asarray([[None]*5 for _ in range(batch_size)]) # [batch_size, 5]
        batch_row_indices = np.asarray([-1] * batch_size)

        batch_questions = []

        for k in xrange(batch_size):
            key = chunk[k]

            MC_dict = self.get_Trans_dict(key)
            candidates = MC_dict['candidates']
            raw_sentences = MC_dict['raw_sentences']
            answer = int(MC_dict['answer'])
            question = MC_dict['question']


            video_feature = self.get_video_feature(key)
            candidates_matrix = self.get_Trans_matrix(candidates)
            candidates_matrix_right = self.get_Trans_matrix(candidates, is_left=False)

            video_mask = self.get_video_mask(video_feature)
            candidates_mask = self.get_Trans_mask(candidates)

            batch_video_feature_convmap[k, :] = data_util.pad_video(
                video_feature, self.get_video_feature_dimension())

            batch_candidates[k] = candidates_matrix
            batch_candidates_right[k] = candidates_matrix_right
            batch_raw_sentences[k, :] = raw_sentences
            batch_answer[k] = answer
            batch_video_mask[k] = video_mask
            batch_candidates_mask[k] = candidates_mask
            batch_row_indices[k] = MC_dict['row_indices']
            batch_questions.append(question)

            batch_debug_sent[k] = self.data_df.loc[key, 'a'+str(int(answer+1))]

        ret = {
            'ids': chunk,
            'video_features': batch_video_feature_convmap,
            'candidates': batch_candidates,
            'candidates_right': batch_candidates_right,
            'answer': batch_answer,
            'raw_sentences': batch_raw_sentences,
            'video_mask': batch_video_mask,
            'candidates_mask': batch_candidates_mask,
            'debug_sent': batch_debug_sent,
            'row_indices' : batch_row_indices,
            'question': batch_questions,
        }
        return ret

    def get_Action_result(self, chunk):
        return self.get_Trans_result(chunk)

    def next_batch(self, batch_size=64, include_extra=False, shuffle=True):
        if not hasattr(self, '_batch_it'):
            self._batch_it = itertools.cycle(self.iter_ids(shuffle=shuffle))

        chunk = []
        for k in xrange(batch_size):
            key = next(self._batch_it)
            chunk.append(key)
        if self.data_type == 'FrameQA':
            return self.get_FrameQA_result(chunk)
        # Make custom function to make batch!
        elif self.data_type == 'Count':
            return self.get_Count_result(chunk)
        elif self.data_type == 'Trans':
            return self.get_Trans_result(chunk)
        elif self.data_type == 'Action':
            return self.get_Action_result(chunk)
        else:
            raise Exception('data_type error in next_batch')

    def batch_iter(self, num_epochs, batch_size, shuffle=True):
        for epoch in xrange(num_epochs):
            steps_in_epoch = int(len(self) / batch_size)

            for s in range(steps_in_epoch+1):
                yield self.next_batch(batch_size, shuffle=shuffle)

    def split_dataset(self, ratio=0.1):
        data_split = DatasetTGIF(dataset_name=self.dataset_name,
                                 image_feature_net=self.image_feature_net,
                                 layer=self.layer,
                                 max_length=self.max_length,
                                 use_moredata=self.use_moredata,
                                 max_n_videos=self.max_n_videos,
                                 data_type=self.data_type,
                                dataframe_dir=self.dataframe_dir,
                                 vocab_dir=self.vocabulary_dir)

        data_split.ids = self.ids[-int(ratio*len(self.ids)):]
        self.ids = self.ids[:-int(ratio*len(self.ids))]
        return data_split
