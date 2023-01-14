TWAV=/home/mpekmezci/workspace-python/room_impulse_response_phd_thessis/data/single-speaker-clean/transmittedEssSignal.wav
WORKDIR=/home/mpekmezci/workspace-python/room_impulse_response_phd_thessis/data/single-speaker-clean/room-fujitsu.conf.2/2021.06.02_13.22.11/iteration-01/speaker-iteration-1/microphone-iteration-1/active-speaker-0/channelNo-1/

python3 test_rir.py $TWAV $WORKDIR/receivedEssSignal-5.wav.ir.wav.cut.wav
