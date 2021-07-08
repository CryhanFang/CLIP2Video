#coding:utf-8
# @Time : 2021/6/19
# @Author : Han Fang
# @File: dataloader_msrvtt_frame.py
# @Version: version 1.0

import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
from dataloaders.rawframe_util import RawFrameExtractor


class MSRVTT_single_sentence_dataLoader(Dataset):
    """MSRVTT dataset loader for single sentence

    Attributes:
        csv_path:  video id of sub set
        features_path: frame directory
        tokenizer: tokenize the word
        max_words: the max number of word
        feature_framerate: frame rate for sampling video
        max_frames: the max number of frame
        image_resolution: resolution of images

    """
    def __init__(
            self,
            csv_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
    ):
        self.data = pd.read_csv(csv_path)
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer

        # frame extractor to sample frames from video
        self.frameExtractor = RawFrameExtractor(framerate=feature_framerate, size=image_resolution)

        # start and end token
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        """length of data loader

        Returns:
            length: length of data loader
        """

        length = len(self.data)
        return length

    def _get_text(self, caption):
        """get tokenized word feature

        Args:
            caption: caption

        Returns:
            pairs_text: tokenized text
            pairs_mask: mask of tokenized text
            pairs_segment: type of tokenized text

        """

        # tokenize word
        words = self.tokenizer.tokenize(caption)

        # add cls token
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]

        # add end token
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        # convert token to id according to the vocab
        input_ids = self.tokenizer.convert_tokens_to_ids(words)

        # add zeros for feature of the same length
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        # ensure the length of feature to be equal with max words
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words
        assert len(segment_ids) == self.max_words
        pairs_text = np.array(input_ids)
        pairs_mask = np.array(input_mask)
        pairs_segment = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment

    def _get_rawvideo(self, video_id):
        """get sampled frames

        Args:
            video_id: id of video

        Returns:
            video: sampled frame
            video_mask: mask of video
        """

        video_mask = np.zeros((1, self.max_frames), dtype=np.long)

        # 1 x L x 1 x 3 x H x W
        video = np.zeros((1, self.max_frames, 1, 3,
                          self.frameExtractor.size, self.frameExtractor.size), dtype=np.float)

        # video_path
        video_path = os.path.join(self.features_path, video_id)

        # get sampling frames
        raw_video_data = self.frameExtractor.get_video_data(video_path, self.max_frames)
        raw_video_data = raw_video_data['video']

        # L x 1 x 3 x H x W
        if len(raw_video_data.shape) > 3:
            raw_video_data_clip = raw_video_data
            # L x T x 3 x H x W
            raw_video_slice = self.frameExtractor.process_raw_data(raw_video_data_clip)

            # max_frames x 1 x 3 x H x W
            if self.max_frames < raw_video_slice.shape[0]:
                    sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                    video_slice = raw_video_slice[sample_indx, ...]
            else:
                video_slice = raw_video_slice

            # set max length, and save video mask and frames
            slice_len = video_slice.shape[0]
            video_mask[0][:slice_len] = [1] * slice_len
            video[0][:slice_len, ...] = video_slice

        else:
            print("get raw video error, skip it.")

        return video, video_mask

    def __getitem__(self, idx):
        """forward method
        Args:
            idx: id
        Returns:
            pairs_text: tokenized text
            pairs_mask: mask of tokenized text
            pairs_segment: type of tokenized text
            video: sampled frames
            video_mask: mask of sampled frames
        """

        video_id = self.data['video_id'].values[idx]
        sentence = self.data['sentence'].values[idx]

        # obtain text data
        pairs_text, pairs_mask, pairs_segment = self._get_text(sentence)

        #obtain video data
        video, video_mask = self._get_rawvideo(video_id)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask



class MSRVTT_multi_sentence_dataLoader(Dataset):
    """MSRVTT dataset loader for multi-sentence

    Attributes:
        csv_path:  video id of sub set
        json_path: total information of video
        features_path: frame directory
        tokenizer: tokenize the word
        max_words: the max number of word
        feature_framerate: frame rate for sampling video
        max_frames: the max number of frame
        image_resolution: resolution of images

    """

    def __init__(
            self,
            csv_path,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
    ):
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.sample_len = 0



        # store the pairs for video and text
        train_video_ids = list(self.csv['video_id'].values)
        self.sentences_dict = {}
        for itm in self.data['sentences']:
            if itm['video_id'] in train_video_ids:
                self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])

        # set the length of paris for one epoch
        self.sample_len = len(self.sentences_dict)


        # frame extractor to sample frames from video
        self.frameExtractor = RawFrameExtractor(framerate=feature_framerate, size=image_resolution, train="train")

        # start and end token
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        """length of data loader

        Returns:
            length: length of data loader
        """

        length = self.sample_len
        return length

    def _get_text(self, caption):
        """get tokenized word feature

        Args:
            caption: caption

        Returns:
            pairs_text: tokenized text
            pairs_mask: mask of tokenized text
            pairs_segment: type of tokenized text

        """

        # tokenize word
        words = self.tokenizer.tokenize(caption)

        # add cls token
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]

        # add end token
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        # convert token to id according to the vocab
        input_ids = self.tokenizer.convert_tokens_to_ids(words)

        # add zeros for feature of the same length
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        # ensure the length of feature to be equal with max words
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words
        assert len(segment_ids) == self.max_words
        pairs_text = np.array(input_ids)
        pairs_mask = np.array(input_mask)
        pairs_segment = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment

    def _get_rawvideo(self, video_id):
        """get sampled frame

        Args:
            video_id: id of video

        Returns:
            video: sampled frame
            video_mask: mask of video
        """

        video_mask = np.zeros((1, self.max_frames), dtype=np.long)

        # 1 x L x 1 x 3 x H x W
        video = np.zeros((1, self.max_frames, 1, 3,
                          self.frameExtractor.size, self.frameExtractor.size), dtype=np.float)

        # video_path
        video_path = os.path.join(self.features_path, video_id)

        # get sampling frames
        raw_video_data = self.frameExtractor.get_video_data(video_path, self.max_frames)
        raw_video_data = raw_video_data['video']

        # L x 1 x 3 x H x W
        if len(raw_video_data.shape) > 3:
            raw_video_data_clip = raw_video_data
            # L x T x 3 x H x W
            raw_video_slice = self.frameExtractor.process_raw_data(raw_video_data_clip)

            # max_frames x 1 x 3 x H x W
            if self.max_frames < raw_video_slice.shape[0]:
                    sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                    video_slice = raw_video_slice[sample_indx, ...]
            else:
                video_slice = raw_video_slice

            # set max length, and save video mask and frames
            slice_len = video_slice.shape[0]
            video_mask[0][:slice_len] = [1] * slice_len
            video[0][:slice_len, ...] = video_slice

        else:
            print("get raw video error, skip it.")

        return video, video_mask

    def __getitem__(self, idx):
        """forward method
        Args:
            idx: id
        Returns:
            pairs_text: tokenized text
            pairs_mask: mask of tokenized text
            pairs_segment: type of tokenized text
            video: sampled frames
            video_mask: mask of sampled frames
        """


        video_id, caption = self.sentences_dict[idx]

        # obtain text data
        pairs_text, pairs_mask, pairs_segment = self._get_text(caption)

        #obtain video data
        video, video_mask = self._get_rawvideo(video_id)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask

