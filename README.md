# EMGSE: Acoustic/EMG Fusion for Multimodal Speech Enhancement

The implementation of EMGSE, a neural-network-based multimodal speech enhancement system using sEMG as auxiliary data.

# Introduction
Multimodal learning has been proven to be an effective method to improve speech enhancement (SE) performance, especially in challenging situations such as low signal-to-noise ratios, speech noise, or unseen noise types. In previous studies, several types of auxiliary data have been used to construct multimodal SE systems, such as lip images, electropalatography, or electromagnetic midsagittal articulography. In this paper, we propose a novel EMGSE framework for multimodal SE, which integrates audio and facial electromyography (EMG) signals. Facial EMG is a biological signal containing articulatory movement information, which can be measured in a non-invasive way. Experimental results show that the proposed EMGSE system can achieve better performance than the audio-only SE system. The benefits of fusing EMG signals with acoustic signals for SE are notable under challenging circumstances. Furthermore, this study reveals that cheek EMG is sufficient for SE.

For more detail please check our <a href="https://ieeexplore.ieee.org/document/9747179" target="_blank">Paper</a>

### Setup ###

You can apply our environment settings by

 ``` js
 pip install -r requirements.txt
 ```

### Database ###

Please download the databases from these websites:

1. [CSL-EMG_Array Corpus](https://www.uni-bremen.de/csl/forschung/lautlose-sprachkommunikation/csl-emg-array-corpus) 
2. [100 noise types](http://web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/HuCorpus.html) (unavailable now)
3. Noise types used for inference in this paper are in the **doc** folder.

Please kindly cite our paper if you find this code useful.

    @inproceedings{wang2022emgse,
      title={EMGSE: Acoustic/EMG Fusion for Multimodal Speech Enhancement},
      author={Wang, Kuan-Chen and Liu, Kai-Chun and Wang, Hsin-Min and Tsao, Yu},
      booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      pages={1116--1120},
      year={2022},
      organization={IEEE}
    }

If you use the CSL-EMG_Array Corpus, please also cite:

    @inproceedings{diener2020csl,
      title={CSL-EMG\_Array: An Open Access Corpus for EMG-to-Speech Conversion.},
      author={Diener, Lorenz and Vishkasougheh, Mehrdad Roustay and Schultz, Tanja},
      booktitle={INTERSPEECH},
      pages={3745--3749},
      year={2020}
    }
