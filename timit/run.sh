#!/bin/bash

#Author: Richardfan
#2017.11.1    Training acoustic model and decode with phoneme-level bigram
#2018.3.16    Add RNNLM and decode with RNNLM
#2018.4.30    Replace the h5py with ark and simplify the data_loader.py

#Top script of one entire experiment

. path.sh

stage=1
TIMIT_DIR='/home/fran/Downloads/TIMIT'
lm_path='/data_prepare/LM/bigram.arpa'
CONF_FILE='./conf/ctc_model_setting.conf'
LOG_DIR='./log/'
MAP_FILE='./decode_map_48-39/map_dict.pkl'
feats='Spect'                          #Fbank, MFCC, Spect

if [ ! -z $1 ]; then
    stage=$1
fi

if [ $stage -le 0 ]; then
    echo ========================================================
    echo "                   Data Preparing                     "
    echo ========================================================

    ./local/timit_data_prep.sh $TIMIT_DIR || exit 1
    
    ./local/make_feat.sh $feats || exit 1

fi

if [ $stage -le 1 ]; then
    echo ========================================================
    echo "                  Acoustic Model                      "
    echo ========================================================

    if ! [ -d $LOG_DIR ]; then
        mkdir -p $LOG_DIR
    fi
    python steps/ctc_train.py --conf $CONF_FILE --log-dir $LOG_DIR || exit 1;
fi

if [ $stage -le 2 ]; then
    echo ========================================================
    echo "                   RNNLM Model                        "
    echo ========================================================
fi

if [ $stage -le 3 ]; then
    echo ========================================================
    echo "                     Decoding                         "
    echo ========================================================

    python steps/test.py --conf $CONF_FILE --map-48-39 $MAP_FILE --lm-path $lm_path || exit 1;
fi

