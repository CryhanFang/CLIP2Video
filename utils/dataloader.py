#coding:utf-8
# @Time : 2021/6/19
# @Author : Han Fang
# @File: dataloader.py
# @Version: version 1.0
import sys
import os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_msrvtt_frame import MSRVTT_single_sentence_dataLoader
from dataloaders.dataloader_msrvtt_frame import MSRVTT_multi_sentence_dataLoader
from dataloaders.dataloader_msvd_frame import MSVD_multi_sentence_dataLoader
from dataloaders.dataloader_vatexEnglish_frame import VATEXENGLISH_multi_sentence_dataLoader
from dataloaders.dataloader_msrvttfull_frame import MSRVTTFULL_multi_sentence_dataLoader


def dataloader_vatexEnglish_train(args, tokenizer):
    """return dataloader for training VATEX with English annotations
    Args:
        args: hyper-parameters
        tokenizer: tokenizer
    Returns:
        dataloader: dataloader
        len(vatexEnglish_dataset): length
        train_sampler: sampler for distributed training
    """

    vatexEnglish_dataset = VATEXENGLISH_multi_sentence_dataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(vatexEnglish_dataset)

    dataloader = DataLoader(
        vatexEnglish_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(vatexEnglish_dataset), train_sampler


def dataloader_vatexEnglish_test(args, tokenizer, subset="test"):
    """return dataloader for testing VATEX with English annotations in multi-sentence captions
    Args:
        args: hyper-parameters
        tokenizer: tokenizer
    Returns:
        dataloader: dataloader
        len(vatexEnglish_dataset): length
    """

    vatexEnglish_dataset = VATEXENGLISH_multi_sentence_dataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    dataloader = DataLoader(
        vatexEnglish_dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(vatexEnglish_dataset)



def dataloader_msrvtt_train(args, tokenizer):
    """return dataloader for training msrvtt-9k
    Args:
        args: hyper-parameters
        tokenizer: tokenizer
    Returns:
        dataloader: dataloader
        len(msrvtt_train_set): length
        train_sampler: sampler for distributed training
    """

    msrvtt_train_set = MSRVTT_multi_sentence_dataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_train_set)

    dataloader = DataLoader(
        msrvtt_train_set,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_train_set), train_sampler

def dataloader_msrvtt_test(args, tokenizer):
    """return dataloader for testing 1k-A protocol
    Args:
        args: hyper-parameters
        tokenizer: tokenizer
    Returns:
        dataloader: dataloader
        len(msrvtt_test_set): length
    """

    msrvtt_test_set = MSRVTT_single_sentence_dataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    dataloader = DataLoader(
        msrvtt_test_set,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(msrvtt_test_set)

def dataloader_msrvttfull_test(args, tokenizer):
    """return dataloader for testing full protocol
    Args:
        args: hyper-parameters
        tokenizer: tokenizer
    Returns:
        dataloader: dataloader
        len(msrvtt_test_set): length
    """
    msrvtt_test_set = MSRVTTFULL_multi_sentence_dataLoader(
        subset='test',
        csv_path=args.val_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    dataloader = DataLoader(
        msrvtt_test_set,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(msrvtt_test_set)

def dataloader_msvd_train(args, tokenizer):
    """return dataloader for training msvd
    Args:
        args: hyper-parameters
        tokenizer: tokenizer
    Returns:
        dataloader: dataloader
        len(msvd_dataset): length
        train_sampler: sampler for distributed training
    """

    msvd_dataset = MSVD_multi_sentence_dataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)

    dataloader = DataLoader(
        msvd_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msvd_dataset), train_sampler

def dataloader_msvd_test(args, tokenizer, subset="test"):
    """return dataloader for testing msvd in multi-sentence captions
    Args:
        args: hyper-parameters
        tokenizer: tokenizer
    Returns:
        dataloader: dataloader
        len(msvd_dataset): length
    """

    msvd_test_set = MSVD_multi_sentence_dataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    dataloader = DataLoader(
        msvd_test_set,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(msvd_test_set)


