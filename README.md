# Audio Samples from [Adapting General Disentanglement-Based Speaker Anonymization for Enhanced Emotion Preservation](https://arxiv.org/abs/2408.05928)

### Paper:
Submitted to *Computer Speech & Language*  

### Authors:
Xiaoxiao Miao, Yuxiang Zhang, Xin Wang, Natalia Tomashenko, Donny Cheng Lock Soh, Ian McLoughlin  

---

## Audio Samples Table

### SAVEE Dataset

| UTT ID     | Emotion | Speaker ID | Original Audio | OH SAVEE Audio | P3 SAVEE Audio |
|------------|---------|------------|----------------|----------------|----------------|
| JK-sa11    | Sad     | JK         | [Listen](wavs-csl/SAVEE/sad/JK_sa11.wav) | [Listen](wavs-csl/SAVEE/sad/JK_sa11_gen.wav) | [Listen](wavs-csl/SAVEE/sad/JK_sa11_gen%207.wav) |
| JE-n14     | Neutral | JE         | [Listen](wavs-csl/SAVEE/neutral/JE_n14.wav) | [Listen](wavs-csl/SAVEE/neutral/JE_n14_gen.wav) | [Listen](wavs-csl/SAVEE/neutral/JE_n14_gen%208.wav) |
| DC-a10     | Angry   | DC         | [Listen](wavs-csl/SAVEE/angry/DC_a10.wav) | [Listen](wavs-csl/SAVEE/angry/DC_a10_gen.wav) | [Listen](wavs-csl/SAVEE/angry/DC_a10_gen_10.wav) |
| KL-h11     | Happy   | KL         | [Listen](wavs-csl/SAVEE/happy/KL_h11.wav) | [Listen](wavs-csl/SAVEE/happy/KL_h11_gen.wav) | [Listen](wavs-csl/SAVEE/happy/KL_h11_gen%207.wav) |

---

### IEMOCAP Dataset

| UTT ID                        | Emotion | Speaker ID | Gender | Session ID | Script ID | Original Audio | OH IEMOCAP Audio | P3 IEMOCAP Audio |
|-------------------------------|---------|------------|--------|------------|-----------|----------------|------------------|------------------|
| Ses02F_script02_2_F018        | Sad     | Ses02-F    | Female | Session 02 | Script 02 | [Listen](wavs-csl/IEMOCAP_test/sad/Ses02F_script02_2_F018.wav) | [Listen](wavs-csl/IEMOCAP_test/sad/Ses02F_script02_2_F018_gen.wav) | [Listen](wavs-csl/IEMOCAP_test/sad/Ses02F_script02_2_F018_gen%207.wav) |

---

### Datasets Information

- [IEMOCAP Dataset](https://sail.usc.edu/iemocap/): Acted, multimodal, and multispeaker database.
- [SAVEE Database](http://kahlan.eps.surrey.ac.uk/savee/): Available free of charge for research purposes.
- [MSP-Improv Corpus](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html): Academic license required.
- [Emotional Speech Database (ESD)](https://hltsingapore.github.io/ESD/download.html): Free for research purposes.

---

### Abstract

A general disentanglement-based speaker anonymization system separates speech into content, speaker, and prosody features. This paper explores adapting such systems to preserve emotion effectively. Two strategies are discussed:
1. Integration of pre-trained emotion embeddings.
2. Post-processing anonymized speaker embeddings to enhance emotion cues.

These strategies demonstrate improved emotion preservation while maintaining anonymization, extending potential use cases for downstream tasks.

For more details, read the [full paper](https://arxiv.org/abs/2408.05928).
