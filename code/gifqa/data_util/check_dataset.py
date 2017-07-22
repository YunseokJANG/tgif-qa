import unittest
import numpy as np

from gifqa.datasets import data_util
from gifqa.datasets.tgif import DatasetTGIF


def print_answer(answer_matrix):
    length = answer_matrix.shape[0]
    result = []
    for i in range(length):
        idx = list(answer_matrix[i]).index(1)
        result.append(idx)
    return np.array(result)


class TestDatasets(unittest.TestCase):

    max_length = 40
    dataset_res = DatasetTGIF(dataset_name='train', max_length=max_length)
    dataset_res.build_word_vocabulary()
    assert hasattr(dataset_res, 'idx2word')


    def test_basic(self):
        self.assertTrue(len(self.dataset_res) > 0)

    def test_next_batch(self):
        # when using resnet
        batch = self.dataset_res.next_batch(5, include_extra=True, shuffle=False)
        print batch.keys()

        self.assertTrue('ids' in batch)
        self.assertTrue('video_features' in batch)
        self.assertTrue('caption_words' in batch)
        print batch['video_features'][0].shape
        self.assertTrue(list(batch['video_features'][0].shape) == [self.max_length, 1, 1, 2048])

        batch = self.dataset_res.next_batch(64)
        self.assertEqual(len(batch['ids']), 64)

#        # when using c3d
#        batch = self.dataset_c3d.next_batch(5, include_extra=True, shuffle=False)
#        print batch.keys()
#
#        self.assertTrue('video_id' in batch)
#        self.assertTrue('video_features' in batch)
#        self.assertTrue('caption_words' in batch)
#        self.assertTrue(list(batch['video_features'][0].shape) == [self.max_length, 1, 1, 4096])
#
#        batch = self.dataset_c3d.next_batch(64)
#        self.assertEqual(len(batch['video_id']), 64)

    def test_batch_correction(self):
        batch_length = 100
        batch = self.dataset_res.next_batch(batch_length, include_extra=True, shuffle=True)
        ids = batch['ids']
        video_features = batch['video_features']
        caption_words = batch['caption_words']
        video_mask = batch['video_mask']
        caption_mask = batch['caption_mask']
        debug_sent = batch['debug_sent']

#        print caption_mask
#        print caption_words
#        print video_mask
#        print np.squeeze(video_features).sum(axis=2)
#        print attribute_mask
#        print attribute
        self.assertEqual(list(debug_sent), list(self.dataset_res.data_df.loc[ids, 'desc1']))
        self.assertEqual(caption_words.shape, caption_mask.shape)
        self.assertEqual(list(video_features.shape), [batch_length, self.max_length, 1, 1, 2048])
        self.assertEqual(list(video_mask.shape), [batch_length, self.max_length])

    def test_data_util_fill_mask(self):
        # fill_mask
        max_length = 10
        current_length = 3
        left_mask = data_util.fill_mask(max_length, current_length, zero_location='RIGHT')
        right_mask = data_util.fill_mask(max_length, current_length, zero_location='LEFT')
        self.assertTrue((left_mask == np.array([1]*3 + [0]*7)).all())
        self.assertTrue((right_mask == np.array([0]*7 + [1]*3)).all())

        max_length = 10
        current_length = 20
        left_mask = data_util.fill_mask(max_length, current_length, zero_location='RIGHT')
        right_mask = data_util.fill_mask(max_length, current_length, zero_location='LEFT')
        self.assertTrue((left_mask == np.array([1]*10)).all())
        self.assertTrue((right_mask == np.array([1]*10)).all())

    def test_data_util_pad_video(self):
        max_length = 10
        video_feature = np.random.normal(size=[3, 2, 2, 4])
        padded_video = data_util.pad_video(video_feature, [max_length, 2, 2, 4])
        self.assertTrue((padded_video[:7] == np.zeros([7, 2, 2, 4])).all())

        max_length = 3
        video_feature = np.random.normal(size=[5, 2, 2, 4])
        padded_video = data_util.pad_video(video_feature, [max_length, 2, 2, 4])

        max_length = 3
        video_feature = np.random.normal(size=[3, 2, 2, 4])
        padded_video = data_util.pad_video(video_feature, [max_length, 2, 2, 4])
        self.assertTrue((padded_video == video_feature).all())

    def test_build_word_vocabulary(self):
        V = self.dataset_res.n_words
        print 'vocab size = ', V

        self.assertEqual(len(self.dataset_res.word2idx), V)
        self.assertEqual(len(self.dataset_res.idx2word), V)


if __name__ == '__main__':
    unittest.main()
