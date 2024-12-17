#!/bin/bash
##SBATCH --gres=gpu:tesla_v100:1
#SBATCH --time=3-00:00:00


#LATENT_CODE_NUM=10
#spk_xv_dir=/home/smg/miao/speech_brain/speechbrain/recipes/VoxCeleb/SpeakerRec/xvectors/anon
# spk_xv_dir=ohnn_xv
norm_type=inter
# flag=_2fc1

# <<!
# dset=vox2_age_test
# boundary=vox2_age_${norm_type}_balance_gender$flag
# output=results/${dset}_balance_gender_boundries$flag
# !

# python prepare_scp.py

# dset=vox2dev
# boundary=../interfacegan/boundaries/vox2_gender_${norm_type}$flag
boundary=boundaries_esd_msp
# output=results/$dset$flag
spk_xv_dir=compute_ori_spk_vector/output_ori_spk_vector
scp_dir=data



#!/bin/bash

# Set the base directory
base_dir="compute_ori_spk_vector/output_ori_spk_vector/"

# Find all xvector.scp files and replace the string
find "$base_dir" -type f -name "xvector.scp" | while read file; do
    # Replace the old path with the current working directory
    sed -i "s|/data/zhangyuxiang/anonymization/ohnn-main/inference_emo/|$PWD\/|g" "$file"
done


for dset in IEMOCAP_dev IEMOCAP_test savee; do
output=results/${dset}_ohnn_pre_esd_msp_sad1

LATENT_CODE_NUM=$(cat ${spk_xv_dir}/${dset}/xvector.scp | wc -l)

echo $LATENT_CODE_NUM

python edit_spk_emo_pre_esd_msp_sad1.py \
    -b $boundary \
    -i ${spk_xv_dir}/${dset}_ohnn/xvector.scp \
    -n "$LATENT_CODE_NUM" \
    -u2e "$scp_dir/${dset}/utt2emo" \
    -norm ${norm_type} \
    -o $output

done
