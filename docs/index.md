# Noise data
18 types of noise are used to create EMGSE testing data. 12 of them are speech noise and the rest are nonspeech noise. 

## Speech noise
Speech noise with Mandarine and English is used.

### Chinese speech noise

Noise type| Audio file|
--------------|------| 
1 female talker |<audio src="noise/Chinese speech noise/one_female_chinese.wav" controls="" preload=""></audio> |
1 male talker |<audio src="noise/Chinese speech noise/one_male_chinese.wav" controls="" preload=""></audio>|
1 female and 1 male talkers  |<audio src="noise/Chinese speech noise/one_female_one_male_chinese.wav" controls="" preload=""></audio>|
2 female talkers  |<audio src="noise/Chinese speech noise/two_female_chinese.wav" controls="" preload=""></audio>|
2 male talkers  |<audio src="noise/Chinese speech noise/two_female_chinese.wav" controls="" preload=""></audio>|
2 male and 1 female talkers  |<audio src="noise/Chinese speech noise/two_male_one_female_chinese.wav" controls="" preload=""></audio>|
 

### English speech noise

Noise type| Audio file|
------------|--------| 
1 female talker |<audio src="noise/English speech noise/one_female_english.wav" controls="" preload=""></audio> |
1 male talker |<audio src="noise/English speech noise/one_male_english.wav" controls="" preload=""></audio>|
1 female and 1 male talkers  |<audio src="noise/English speech noise/one_female_one_male_english.wav" controls="" preload=""></audio>|
2 female talkers  |<audio src="noise/English speech noise/two_female_english.wav" controls="" preload=""></audio>|
2 male talkers  |<audio src="noise/English speech noise/two_female_english.wav" controls="" preload=""></audio>|
2 male and 1 female talkers  |<audio src="noise/English speech noise/two_male_one_female_english.wav" controls="" preload=""></audio>|

## Nonspeech noise

Noise type| Audio file|
--------------|-----| 
Car|<audio src="noise/car noise.wav" controls="" preload=""></audio> |   
Engine|<audio src="noise/engine noise.wav" controls="" preload=""></audio>|
Pink|<audio src="noise/pink noise.wav" controls="" preload=""></audio>|
White|<audio src="noise/white noise.wav" controls="" preload=""></audio>|
Street noise (1)|<audio src="noise/street noise(1).wav" controls="" preload=""></audio>|
Street noise (2)|<audio src="noise/street noise(2).wav" controls="" preload=""></audio>|

# Experimental Results 

## SE Performance of EMGSE and baseline

Utterance 40  : "You can see the rain curtains are wrapping."

Speaker       : number 5, male

Noise type    : 1 English male talker (speech noise)

Source|  SNR -11dB| SNR 4dB|
--------------|-----|-----|
Ground truth |<audio src="Noise/car noise.wav" controls="" preload=""></audio>|<audio src="Noise/car noise.wav" controls="" preload=""></audio>|
Noisy |
Enhanced(Baseline)|
Enhanced(EMGSE)|
Enhanced(EMGSE 28ch)|

Utterance 40 : "You can see the rain curtains are wrapping."

Speaker      : number 5, male

Noise type   : Street noise (2) (nonspeech noise)

   Source      |      SNR -11dB     |       SNR 4dB     |
--------------|-----|-----|
Ground truth |<audio src="wavfile/utter40/Spk5_Block1-Initial_0040.wav" controls="" preload=""></audio><img src="wavfile/utter40/Spk5_Block1-Initial_0040.png" alt="40_clean">|<audio src="wavfile/utter40/Spk5_Block1-Initial_0040.wav" controls="" preload=""></audio><img src="wavfile/utter40/Spk5_Block1-Initial_0040.png" alt="40_clean">|
Noisy |<audio src="wavfile/utter40/street/Noisy/Spk5_Block1-Initial_0040_street_-11.wav" controls="" preload=""></audio><img src="wavfile/utter40/street/Noisy/Spk5_Block1-Initial_0040_street_-11.png" alt="40_street_noisy_-11">|<audio src="wavfile/utter40/street/Noisy/Spk5_Block1-Initial_0040_street_4.wav" controls="" preload=""></audio><img src="wavfile/utter40/street/Noisy/Spk5_Block1-Initial_0040_street_4.png" alt="40_street_noisy_4">|
Enhanced(Baseline)|<audio src="wavfile/utter40/street/baseline/Spk5_Block1-Initial_0040_enh_base_street_-11.wav" controls="" preload=""></audio><img src="wavfile/utter40/street/baseline/Spk5_Block1-Initial_0040_enh_base_street_-11.png" alt="">|<audio src="wavfile/utter40/street/baseline/Spk5_Block1-Initial_0040_enh_base_street_4.wav" controls="" preload=""></audio><img src="wavfile/utter40/street/baseline/Spk5_Block1-Initial_0040_enh_base_street_4.png" alt="">|
Enhanced(EMGSE)|<audio src="wavfile/utter40/street/EMGSE/Spk5_Block1-Initial_0040_enh_emgse_street_-11.wav" controls="" preload=""></audio><img src="wavfile/utter40/street/EMGSE/Spk5_Block1-Initial_0040_enh_emgse_street_-11.png" alt="">|<audio src="wavfile/utter40/street/EMGSE/Spk5_Block1-Initial_0040_enh_emgse_street_4.wav" controls="" preload=""></audio><img src="wavfile/utter40/street/EMGSE/Spk5_Block1-Initial_0040_enh_emgse_street_4.png" alt="">|
Enhanced(EMGSE 28ch)|<audio src="wavfile/utter40/street/EMGSE28/Spk5_Block1-Initial_0040_enh_emgse28_street_-11.wav" controls="" preload=""></audio><img src="wavfile/utter40/street/EMGSE28/Spk5_Block1-Initial_0040_enh_emgse28_street_-11.png" alt="">|<audio src="wavfile/utter40/street/EMGSE28/Spk5_Block1-Initial_0040_enh_emgse28_street_4.wav" controls="" preload=""></audio><img src="wavfile/utter40/street/EMGSE28/Spk5_Block1-Initial_0040_enh_emgse28_street_4.png" alt="">|

## The phenomenon of missing syllables in EMGSE

Utterance 19: "You can see the rain curtains are wrapping."

Utterance 29: "You can see the rain curtains are wrapping."

Speaker      : number 5, male

Noise type : Car noise

Source| Utterance 19| Utterance 29|
--------------|-----|-----|
Ground truth|
Noisy|
Enhanced(Baseline)|
Enhanced(EMGSE)|



 
<div align="center"></div>
<p style="text-align: center;"> </p>

