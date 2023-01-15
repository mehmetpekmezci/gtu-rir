#!/usr/bin/python
import numpy as np
import wave

# https://github.com/vogelchr/sweep_analysis/blob/master/gen_log_sine_sweep.py

# Generate a logarithmic sine sweep as published in
# Farina, Angelom Audio Engineering Society Convention 122, May 2007
# http://www.aes.org/e-lib/browse.cfm?elib=14106

# Copyright (C) 2015 Christian Vogel <vogelchr@vogel.cx>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

exp_dB_Pwr = np.log(10)/10
dB_pwr = lambda dB : np.exp(exp_dB_Pwr * dB)   # conv dB to power factor

exp_dB_Ampl = exp_dB_Pwr / 2.0
dB_amp = lambda dB : np.exp(exp_dB_Ampl * dB)  # conv dB to amplitude factor

def main() :
    import optparse
    parser = optparse.OptionParser(usage='%prog [options] output.wav')

    parser.add_option('-r', '--rate', default=48000, type='int',
        metavar='HZ', help='Sample rate, default is 48 kHz')
    parser.add_option('-I', '--fmin', default=15., type='float',
        metavar='HZ', help='Minimum frequency, default is 15 Hz')
    parser.add_option('-A', '--fmax', default=22e3, type='float',
        metavar='HZ', help='Maximum frequency, default is 22 kHz')
    parser.add_option('-d', '--duration', default=10., type='float',
        metavar='sec', help='Duration of sweep, default is 10s')
    parser.add_option('-f', '--fadein', default=0.01, type='float',
        metavar='sec', help='Fadein duration, default is 10 ms')
    parser.add_option('-s', '--silence', default=1.0, type='float',
        metavar='sec', help='Amount of silence around sweep, default is 1s')
    parser.add_option('-e', '--equalize', default=False, action='store_true',
        help='Equalize sweep with -3dB/octave to ensure uniform power density'+\
            ', default is off.')
    parser.add_option('-a', '--attenuate', default=0.0, type='float',
        metavar='dB', help='Attenuate by dB (e.g. 6 dB = half FS amplitude)'+\
            ', default is no attenuation: the amplitude is 16 bit full scale')
    opts, args = parser.parse_args()

    if len(args) != 1 :
        parser.error('You have to specify exactly one output file.')

    print('Samplerate used is %d Hz'%(opts.rate))
    print('Generating a sine sweep from %f Hz to %f Hz'%(opts.fmin, opts.fmax))

    Nsamp = int(np.ceil(opts.rate*opts.duration)) # number of samples in sweep

    tsteps = np.linspace(0.0, opts.duration, Nsamp, False)

    #
    # Farina:  om1="omega 1", om2="omega 2"
    #  Note that omega1 = fmin*2*pi, omega2 = fmax*2*pi
    #
    #  sin( om1 * T / ln(om2/om1) * (exp(t/T*ln(om2/om1)) - 1) )
    #       \-------------------/         \------------/
    #             alpha                      beta
    #       \------------------------------------------------/
    #                phase
    #
    #  log_fraction = ln(om2/om1) = ln(opts.fmax/opts.fmin)
    #

    log_fraction = np.log(opts.fmax/opts.fmin)             # ln(o2/o1)
    alpha = 2.0*np.pi*opts.fmin*opts.duration/log_fraction # om1*T/ln(...)
    beta = np.log(opts.fmax/opts.fmin)/opts.duration       # ln(...)/T
    phase = alpha*np.exp(tsteps*beta)

    # this is the sin sweep, 0dB == (2^15)-1
    full_scale = 32767 * dB_amp(-opts.attenuate)
    print('Full scale is %d LSB (%f dB FS of 16bit wav)'%(full_scale, -opts.attenuate))
    sin_arr = np.sin(phase)*full_scale

    # apply a pre-emphasis of 3dB/octave so that total power density over
    # the whole sweep is uniform over the whole range from fmin..fmax
    if opts.equalize :
        num_octaves = np.log(opts.fmax/opts.fmin)/np.log(2)

        print('Sweep covers %f octaves, apply a 3dB/octave equalization.'%(
            num_octaves))
        octave_steps = np.linspace(-num_octaves*3.0, 0.0, Nsamp) # -3dB/oct
        sin_arr *= dB_amp(octave_steps)

    # Apply a fade-in to the beginning/end of the sweeps to avoid steps.
    if opts.fadein :
        print('Generating a fadein/fadeout of %f seconds.'%(opts.fadein))
        # linear fadein/fadeout in the shape of sin^2 from 0..pi/2
        ramp = np.power(np.sin(np.linspace(0, np.pi/2.0, int(opts.rate*opts.fadein))),2.0)
        # apply fadein/fadeout to sine sweep
        sin_arr[0:ramp.shape[0]] *= ramp
        sin_arr[-ramp.shape[0]:] *= ramp[::-1]

    if opts.silence :
        print('Prepending and appending %f seconds of silence.'%(opts.silence))
        # prepend/append silence,
        # 16384 (2^14) scales to 1/2 of the full amplitude (-6dBfs)
        silence = np.zeros(int(opts.rate*opts.silence))
        full_array = np.concatenate([silence, sin_arr, silence])
    else :
        full_array = sin_arr

    print('Writing a total of %d samples.'%(full_array.shape[0]))

    # scipy.io.wavfile seems to write float wav files which my other software
    # doesn't understand, so let's just create a 16bit wavefile instead

    writer = wave.open(args[0], 'wb')
    writer.setnchannels(1) # 1 channel, mono
    writer.setsampwidth(2) # 16bit
    writer.setframerate(opts.rate) # sample rate
    writer.writeframes(full_array.astype('int16').tostring())

if __name__ == '__main__' :
    main()
