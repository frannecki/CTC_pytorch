KALDI_ROOT=../../kaldi/
KenLM_ROOT=~/kenlm/build/

export PATH=$KenLM_ROOT/:$PATH
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export WARP_CTC_PATH=~/Documents/warp-ctc/build
export PATH=$PATH:~/env/eigen
export PATH=$PATH:~/Documents/kaldi/src/featbin
export PATH=$PATH:~/Documents/kaldi/src/cudafeatbin
