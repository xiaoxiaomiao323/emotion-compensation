#!/bin/bash

config=extract_ecapa_f_ecapa_vox.yaml
echo ${config}
outdir=/data/zhangyuxiang/anonymization/ohnn-main/inference_emo/compute_ori_spk_vector/output_ori_spk_vector
## extract librispeech-360
# source_dir=/root/data/
# source_dir=/data/zhangyuxiang/data/Emotion Speech Dataset
# dset=voxconverse
dset=ESD
# python compute_ori_spk_vector/extract_emb.py ${config} $source_dir/$dset/select.lst $outdir/$dset fbank
python extract_emb.py ${config} ../ESD/full_english.txt $outdir/$dset fbank
