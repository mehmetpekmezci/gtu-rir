mkdir -p logs
TS=$(date '+%Y%m%d%H%M%S')
nohup ./run_training.sh >& logs/nohup.out.$TS &
