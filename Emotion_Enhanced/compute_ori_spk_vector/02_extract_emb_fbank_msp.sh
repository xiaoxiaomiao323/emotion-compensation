#!/bin/bash

config=extract_ecapa_f_ecapa_vox.yaml
echo ${config}
outdir=output_ori_spk_vector
## extract librispeech-360
# source_dir=/root/data/
# source_dir=/data/zhangyuxiang/data/Emotion Speech Dataset
dset=MSP
python extract_emb.py ${config} ../${dset}/wav.scp $outdir/$dset fbank