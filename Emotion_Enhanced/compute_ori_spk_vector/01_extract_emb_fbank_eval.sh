#!/bin/bash

config=extract_ecapa_f_ecapa_vox.yaml
echo ${config}
outdir=output_ori_spk_vector
## extract librispeech-360
# source_dir=/root/data/
# source_dir=/data/zhangyuxiang/data/Emotion Speech Dataset
# dset=voxconverse
for dset in libri_dev_{enrolls,trials_f,trials_m} \
	libri_test_{enrolls,trials_f,trials_m} \
		IEMOCAP_dev IEMOCAP_test; do
# python compute_ori_spk_vector/extract_emb.py ${config} $source_dir/$dset/select.lst $outdir/$dset fbank
    echo $dset
    python extract_emb.py ${config} data/${dset}/wav.scp $outdir/$dset fbank
done