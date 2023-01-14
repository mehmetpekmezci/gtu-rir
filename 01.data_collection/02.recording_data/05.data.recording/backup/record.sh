

python3 gen_log_sine_sweep.py log_sine_sweep.wav

INPUT_DEVICES=$(arecord -L | grep hw:CARD| grep -v plughw)
input_device_no=0

for input_device in $INPUT_DEVICES
do
	input_device_no=$((input_device_no+1))

	CHANNEL_COUNT=1
	echo $input_device | grep PCH > /dev/null
	if [ $? = 0 ]
	then
		CHANNEL_COUNT=2
	fi

	rm -f microphone.$input_device_no.wav

	echo ffmpeg -f alsa -channels $CHANNEL_COUNT -sample_rate 44100 -i  $input_device -t 10 microphone.$input_device_no.wav  
	ffmpeg -f alsa -channels $CHANNEL_COUNT -sample_rate 44100 -i  $input_device -t 10 microphone.$input_device_no.wav >&/dev/null &
done

aplay log_sine_sweep.wav

#hw:CARD=PCH,DEV=0
#hw:CARD=Microphone,DEV=0
#hw:CARD=Microphone_1,DEV=0
