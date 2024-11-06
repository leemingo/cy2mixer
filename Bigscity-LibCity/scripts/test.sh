#Ensure that the METR-LA data is located in the directory ("./Bigscity-Libcity/raw_data/METR-LA")

cd ../
python run_model.py --task traffic_state_pred --model Cy2Mixer --dataset METR_LA