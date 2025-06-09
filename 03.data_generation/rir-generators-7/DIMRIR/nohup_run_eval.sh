mkdir -p logs
TS=$(date '+%Y%m%d%H%M%S')
nohup ./run_eval.sh >& logs/nohup.out.$TS &
