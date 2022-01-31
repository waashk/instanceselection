#!/usr/bin/bash

MYHOME=`pwd`
cd ../resources/
MYHOME=`pwd`
echo $MYHOME

mkdir -p pretrained_models/albert/base
cd pretrained_models/albert/base
wget https://huggingface.co/albert-base-v2/resolve/main/pytorch_model.bin
wget https://huggingface.co/albert-base-v2/resolve/main/config.json
cd $MYHOME

mkdir -p pretrained_models/bart/base
cd pretrained_models/bart/base
wget https://huggingface.co/facebook/bart-base/resolve/main/pytorch_model.bin
wget https://huggingface.co/facebook/bart-base/resolve/main/config.json
cd $MYHOME

mkdir -p pretrained_models/distilbert/base
cd pretrained_models/distilbert/base
wget https://huggingface.co/distilbert-base-cased/resolve/main/pytorch_model.bin
wget https://huggingface.co/distilbert-base-cased/resolve/main/config.json
cd $MYHOME

mkdir -p pretrained_models/gpt2/base
cd pretrained_models/gpt2/base
wget https://huggingface.co/gpt2/resolve/main/pytorch_model.bin
wget https://huggingface.co/gpt2/resolve/main/config.json
cd $MYHOME

mkdir -p pretrained_models/roberta/base
cd pretrained_models/roberta/base
wget https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin
wget https://huggingface.co/roberta-base/resolve/main/config.json
cd $MYHOME

mkdir -p pretrained_models/xlnet/base
cd pretrained_models/xlnet/base
wget https://huggingface.co/xlnet-base-cased/resolve/main/pytorch_model.bin
wget https://huggingface.co/xlnet-base-cased/resolve/main/config.json
cd $MYHOME

mkdir -p pretrained_models/bert/base
cd pretrained_models/bert/base
wget https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin
wget https://huggingface.co/bert-base-uncased/resolve/main/config.json
cd $MYHOME
