ulimit -n 100000
export CUDA_LAUNCH_BLOCKING=1
python3 main.py --cfg cfg/RIR_s1.yml --gpu 0

