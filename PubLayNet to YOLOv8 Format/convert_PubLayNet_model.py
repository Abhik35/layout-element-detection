#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import numpy as np
import os
import sys
import json
import torch

NUM_PUBLAYNET_CLS = 6

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a PubLayNet pre-trained model for fine-tuning on another target dataset')
    parser.add_argument(
        '--PubLayNet_model', dest='D:\Sourajit Dey\figure and table detection\model_final.pkl',
        help='Pretrained network weights file path',
        default=None, type=str)
    parser.add_argument(
        '--lookup_table', dest='lookup_table',
        help='Blob conversion lookup table',
        type=json.loads)
    parser.add_argument(
        '--output', dest='out_file_name',
        help='Output file path',
        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    args.NUM_TARGET_CLS = len(args.lookup_table)
    return args

def convert_PubLayNet_blobs_to_target_blobs(model_dict):
    for k, v in model_dict.items():
        if isinstance(v, torch.Tensor) and len(v.shape) > 0:
            if v.shape[0] == NUM_PUBLAYNET_CLS or v.shape[0] == 4 * NUM_PUBLAYNET_CLS:
                PubLayNet_blob = model_dict[k]
                print(f'Converting PUBLAYNET blob {k} with shape {PubLayNet_blob.shape}')
                target_blob = convert_PubLayNet_blob_to_target_blob(PubLayNet_blob, args.lookup_table)
                print(f' -> converted shape {target_blob.shape}')
                model_dict[k] = target_blob

def convert_PubLayNet_blob_to_target_blob(PubLayNet_blob, lookup_table):
    PubLayNet_shape = PubLayNet_blob.shape
    leading_factor = int(PubLayNet_shape[0] / NUM_PUBLAYNET_CLS)
    tail_shape = list(PubLayNet_shape[1:])
    assert leading_factor == 1 or leading_factor == 4

    PubLayNet_blob = PubLayNet_blob.reshape([NUM_PUBLAYNET_CLS, -1] + tail_shape)
    std = PubLayNet_blob.std()
    mean = PubLayNet_blob.mean()
    target_shape = [args.NUM_TARGET_CLS] + list(PubLayNet_blob.shape[1:])
    target_blob = (torch.randn(*target_shape) * std + mean).float()

    for i in range(args.NUM_TARGET_CLS):
        PubLayNet_cls_id = lookup_table[i]
        if PubLayNet_cls_id >= 0:
            target_blob[i] = PubLayNet_blob[PubLayNet_cls_id]

    target_shape = [args.NUM_TARGET_CLS * leading_factor] + tail_shape
    return target_blob.reshape(target_shape)

def remove_momentum(model_dict):
    keys_to_remove = [k for k in model_dict.keys() if k.endswith('_momentum')]
    for k in keys_to_remove:
        del model_dict[k]

def load_and_convert_PubLayNet_model(args):
    model_dict = load_object(args.PubLayNet_model_file_name)
    remove_momentum(model_dict)
    convert_PubLayNet_blobs_to_target_blobs(model_dict)
    return model_dict

if __name__ == '__main__':
    args = parse_args()
    print(args)
    assert os.path.exists(args.PubLayNet_model_file_name), 'Weights file does not exist'
    weights = load_and_convert_PubLayNet_model(args)
    save_object(weights, args.out_file_name)
    print(f'Wrote blobs to {args.out_file_name}:')
    print(sorted(weights.keys()))
