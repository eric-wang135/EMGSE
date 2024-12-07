speaker_idx=$1

#python data_arrange.py --corpus_path ../../EMG-master/main/CSL-EMG_Array/ --speaker_idx $speaker_idx
#python sEMG_feature_extract.py --speaker_idx $speaker_idx
#python add_noise.py --data_type train --speaker_idx $speaker_idx
#python add_noise.py --data_type test --speaker_idx $speaker_idx
#python gen_pt.py --speaker_idx $speaker_idx --norm_emg minmax --norm_spec minmax --encoded --context 15
#python main.py --mode train --train_path ./trainpt_1spk${speaker_idx} --test_noisy ./data_1spk${speaker_idx}/test/noisy --test_clean ./data_1spk${speaker_idx}_tdf/test/clean --model EMGSE_all --encoded --context 15 --norm_emg minmax --norm_spec minmax --output
python main.py --mode test --test_noisy ./data_1spk${speaker_idx}/test/noisy --test_clean ./data_1spk${speaker_idx}_tdf/test/clean --model EMGSE_all --encoded --context 15 --norm_emg minmax --norm_spec minmax --output


