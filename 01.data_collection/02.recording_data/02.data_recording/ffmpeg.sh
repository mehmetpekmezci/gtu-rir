        rm -f /var/tmp/temp.wav
        ffmpeg  -f  alsa  -ac  1  -sample_rate 44100 -i hw:$1,0 -t 3 -f s32le /var/tmp/temp.wav
        ffplay  -autoexit  -f  alsa  -ac  1  -sample_rate 44100  -f s32le -t 3  /var/tmp/temp.wav
        rm -f /var/tmp/temp.wav

