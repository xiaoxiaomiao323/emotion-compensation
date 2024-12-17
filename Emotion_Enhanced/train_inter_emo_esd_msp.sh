#!/bin/bash
##SBATCH --gres=gpu:tesla_v100:1
#SBATCH --time=3-00:00:00

#emo
ATTRIBUTE_NAME=emo
# spk_vec=ohnn_xv/xvector.scp 
spk_vec=ESD/esd_msp/xvector.scp
att_label=ESD/esd_msp/
# att_label=data/vox2/spk_gender.pkl
norm_type=inter
# num=453151
num=25298
# flag=angry

for emo in angry happy neutral sad; do
    echo "emo: $emo"
    python train_boundary.py \
        -o boundaries_esd_msp/ESD_MSP_"$ATTRIBUTE_NAME"_"$norm_type"_${emo} \
        -nt $num \
        -c $spk_vec \
        -s $att_label/${emo}.pkl \
        -norm ${norm_type} \
        -r 0.8
done
