while true
do
   nohup python3 get_temp_hum_sensor.py  
   rm -f nohup.out
   sleep 600 # 10 minnutes
done
