DATE=$(date '+%Y%m%d%H%M%S')
python3 main.py >& $DATE.log &
echo $! > run.pid
echo tail -f $DATE.log

