#coding:utf-8
# @Time : 2021/6/19
# @Author : Han Fang
# @File: raw_frame_util.py
# @Version: version 1.0

import torch
import numpy as np
from PIL import Image
# pytorch=1.7.1
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import CenterCrop
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
import os



class RawFrameExtractor():
    """frame extractor for a given of directory with video

    Attributes:
        centercrop: center crop for pre-preprocess
        size: resolution of images
        framerate: frame rate for sampling
        transform: transform method for pre-process
        train: set train for random sampling in the uniform interval
    """

    def __init__(self, centercrop=False, size=224, framerate=-1, train='subset'):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(self.size)
        self.train = True if train == 'train' else False

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


    def video_to_tensor(self, video_file, max_frame, preprocess, sample_fp=0):
        """sample video into tensor
        Args:
            video_file: location of video file
            max_frame: max frame number
            preprocessL preprocess method
            sample_fp: sampling rate

        Returns:
            image_input: sample frames
        """

        assert sample_fp > -1
        video_name = os.listdir(video_file)
        video_name.sort()

        # Pre-uniform sampling
        current_frame = len(video_name) // sample_fp
        current_sample_indx = np.linspace(0, len(video_name) - 1, num=current_frame, dtype=int)

        # if the length of current_sample_indx is already less than max_frame, just use the current version to tensor
        # else continue to uniformly sample the frames whose length is max_frame
        # when training, the frames are sampled randomly in the uniform split interval
        if max_frame >=  current_sample_indx.shape[0]:
            frame_index = np.arange(0, current_sample_indx.shape[0])
        else:
            frame_index = np.linspace(0, current_sample_indx.shape[0] - 1, num=max_frame, dtype=int)

            if self.train:
                step_len = (frame_index[1] - frame_index[0]) // 2
                if step_len > 2:
                    random_index = np.random.randint(-1 * step_len, step_len, (frame_index.shape[0] - 2))
                    zero_index = np.zeros((1))
                    index = np.concatenate((zero_index, random_index, zero_index))
                    frame_index = frame_index + index

        # pre-process frames
        images = []
        for index in frame_index:
            image_path = os.path.join(video_file, video_name[current_sample_indx[int(index)]])
            images.append(preprocess(Image.open(image_path).convert("RGB")))

        # convert into tensor
        if len(images) > 0:
            video_data = torch.tensor(np.stack(images))
        else:
            video_data = torch.zeros(1)
        return {'video': video_data}


    def get_video_data(self, video_path, max_frame):
        """get video data
        Args:
            video_path: id
            max_frame: max frame number

        Returns:
            image_input: sample frames
        """

        image_input = self.video_to_tensor(video_path, max_frame, self.transform, sample_fp=self.framerate)

        return image_input

    def process_raw_data(self, raw_video_data):
        """reshape the raw video
        Args:
            raw_video_data: sampled frames

        Returns:
            tensor: reshaped tensor
        """

        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

