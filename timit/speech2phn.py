#!/usr/bin/python

import os
import sys
import copy
import time 
import torch
import argparse
import numpy as np
import configparser
import torch.nn as nn
from torch.autograd import Variable

from steps.utils import *
from steps.ctc_model import *
from steps.ctcDecoder import *
from steps.ctcDecoder import GreedyDecoder
from local.make_spectrum import *

supported_rnn = {'nn.LSTM':nn.LSTM, 'nn.GRU': nn.GRU, 'nn.RNN':nn.RNN}
supported_activate = {'relu':nn.ReLU, 'tanh':nn.Tanh, 'sigmoid':nn.Sigmoid}
out_map = {"phone":"phn", "char":"wrd"}
audio_conf = {"sample_rate":16000, 'window_size':0.025, 'window_stride':0.01, 'window': 'hamming'}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def speech2phn(model, wav_path, map_dict, decoder):
    windows = {'hamming':scipy.signal.hamming, 'hann':scipy.signal.hann, 'blackman':scipy.signal.blackman,
                'bartlett':scipy.signal.bartlett}
    audio_conf = {"sample_rate":16000, 'window_size':0.025, 'window_stride':0.01, 'window': 'hamming'}
    utt_mat = parse_audio(wav_path, audio_conf, windows, normalize=True)
    utt_mat = F_Mel(torch.FloatTensor(utt_mat), audio_conf)
    utt_mat = torch.reshape(utt_mat, (1,1,utt_mat.size(0),utt_mat.size(1)))
    #model.to(device)
    #utt_mat.to(device)
    probs = model(utt_mat)
    probs = probs.data.cpu()
    return decoder.decode(probs, None)


if __name__ == '__main__':
    #Define Model
    package = torch.load('./log/model_dev.pkl')
    cf = configparser.ConfigParser()
    try:
        cf.read('./conf/ctc_model_setting.conf')
    except:
        print("conf file not exists")
        sys.exit(1)
    rnn_param = package["rnn_param"]
    add_cnn = package["add_cnn"]
    cnn_param = package["cnn_param"]
    num_class = package["num_class"]
    feature_type = package['epoch']['feature_type']
    n_feats = package['epoch']['n_feats']
    out_type = package['epoch']['out_type']
    drop_out = package['_drop_out']
    try:
        mel = package['epoch']['mel']
    except:
        mel = False
    
    USE_CUDA = cf.getboolean('Training', 'use_cuda')
    beam_width = cf.getint('Decode', 'beam_width')
    lm_alpha = cf.getfloat('Decode', 'lm_alpha')
    decoder_type =  cf.get('Decode', 'decode_type')
    data_set = cf.get('Decode', 'eval_dataset')
    
    model = CTC_Model(rnn_param=rnn_param, add_cnn=add_cnn, 
      cnn_param=cnn_param, num_class=num_class, drop_out=drop_out)
    model.load_state_dict(package['state_dict'])
    model.eval()

    import pickle
    f = open('./decode_map_48-39/map_dict.pkl', 'rb')
    map_dict = pickle.load(f)
    f.close()

    class_file = '/home/fran/Documents/CTC_pytorch_data/data_prepare/phone_list.txt'
    _, int2class = process_map_file(class_file)
    # Greedy
    decoder  = GreedyDecoder(int2class, space_idx=-1, blank_index=0)

    phones = speech2phn(model, './test/SI1573.WAV', map_dict, decoder)
    print('decoded:'+phones[0])