#!/bin/bash
# ==============================================================================
# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Xiaoxiao Miao (xiaoxiaomiao@nii.ac.jp)
# All rights reserved.
# ==============================================================================

source env.sh


#download pretrain models
if [ ! -e "pretrained_models_anon_xv/" ]; then
    if [ -f pretrained_models_anon_xv.tar.gz ];
    then
        rm pretrained_models_anon_xv.tar.gz
    fi
    echo -e "${RED}Downloading pre-trained model${NC}"

    wget https://zenodo.org/record/6529898/files/pretrained_models_anon_xv.tar.gz
    tar -xzvf pretrained_models_anon_xv.tar.gz
    cd pretrained_models_anon_xv/
    wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
    cd $home
fi

anon=_ohnn_pre_esd_msp_sad1
extract_config=configs/extract_ecapa_f_ecapa_vox.yaml


   echo -e "${RED}Try pre-trained model${NC}"
   #for model_type in {multilan_fbank_xv_ssl_freeze,libri_tts_clean_100_fbank_xv_ssl_freeze}; do
   model_type=libri_tts_clean_100_fbank_xv_ssl_freeze   
   xv_dir=../../results/
   for step in 0 1; do
   for dset in IEMOCAP_dev IEMOCAP_test savee; do
       python adapted_from_facebookresearch/inference.py --input_test_file ../../data/$dset/wav.scp \
		    --xv_dir $xv_dir/${dset}${anon} \
		    --checkpoint_file pretrained_models_anon_xv/HiFi-GAN/$model_type \
		    --output_dir output/$model_type/${dset}${anon}$step \
		    --step $step
   	done
     done
   echo -e "${RED}Please check generated waveforms from pre-trained model in ./pretrained_models/output"
    



