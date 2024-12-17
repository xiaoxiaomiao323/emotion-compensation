## Adapting General Disentanglement-Based Speaker Anonymization for Enhanced Emotion Preservation

This is an implementation of the paper - [Adapting General Disentanglement-Based Speaker Anonymization for Enhanced Emotion Preservation](https://arxiv.org/abs/2408.05928)

The authors are Xiaoxiao Miao, Yuxiang Zhang, Xin Wang, Natalia Tomashenko, Donny Cheng Lock Soh, Ian Mcloughlin


Audio samples can be found here: https://xiaoxiaomiao323.github.io/ohnn-emo-audio/


## How to run

`git clone https://github.com/xiaoxiaomiao323/emotion-compensation.git`

`cd emotion-compensation`

- Training the svm boundary of vectors

`bash train_inter_emo_esd_msp.sh`

- Editing vector

`bash edit_inter_emo_pre_esd_msp_sad1.sh`

- Generate anonymized speech using edited vectors, 

`cd gen`

`bash scripts/install.sh`

`bash demo.sh` this will use original data as input, please download savee and IEMOCAP original dataset and change data/*/wav.scp to your data directory before you run this command.



## Acknowledgments
This study is supported by JST, PRESTO Grant JPMJPR23P9, Japan, Ministry of Education, Singapore, under its Academic Research Tier 1 (R-R13-A405-0005) and its SIT's Ignition grant (STEM) (R-IE3-A405-0005) andin part by National Research Foundation Singapore for (a) AI Singapore Programme (award AISG2-GC-2022-004) (b) with Infocomm Media Development Authority (Digital Trust Centre award DTC-RGC-07).
## License
The whole project follows the Attribution-NonCommercial 4.0 International License




