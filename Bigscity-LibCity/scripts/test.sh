cd ../
python run_model.py --task traffic_state_pred --dataset PEMSD4 --model MTGNN > ../logs/MTGNN_PEMSD4.out
python run_model.py --task traffic_state_pred --dataset PEMSD7 --model MTGNN > ../logs/MTGNN_PEMSD7.out
python run_model.py --task traffic_state_pred --dataset PEMSD8 --model MTGNN > ../logs/MTGNN_PEMSD8.out

python run_model.py --task traffic_state_pred --dataset PEMSD4 --model GTS > ../logs/GTS_PEMSD4.out
python run_model.py --task traffic_state_pred --dataset PEMSD7 --model GTS > ../logs/GTS_PEMSD7.out
python run_model.py --task traffic_state_pred --dataset PEMSD8 --model GTS > ../logs/GTS_PEMSD8.out

python run_model.py --task traffic_state_pred --dataset PEMSD4 --model HGCN > ../logs/HGCN_PEMSD4.out
python run_model.py --task traffic_state_pred --dataset PEMSD7 --model HGCN > ../logs/HGCN_PEMSD7.out
python run_model.py --task traffic_state_pred --dataset PEMSD8 --model HGCN > ../logs/HGCN_PEMSD8.out

python run_model.py --task traffic_state_pred --dataset PEMSD4 --model GWNET > ../logs/GWNET_PEMSD4.out
python run_model.py --task traffic_state_pred --dataset PEMSD7 --model GWNET > ../logs/GWNET_PEMSD7.out
python run_model.py --task traffic_state_pred --dataset PEMSD8 --model GWNET > ../logs/GWNET_PEMSD8.out

python run_model.py --task traffic_state_pred --dataset PEMSD4 --model GMAN > ../logs/GMAN_PEMSD4.out
python run_model.py --task traffic_state_pred --dataset PEMSD7 --model GMAN > ../logs/GMAN_PEMSD7.out
python run_model.py --task traffic_state_pred --dataset PEMSD8 --model GMAN > ../logs/GMAN_PEMSD8.out
