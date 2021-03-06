import sys
import argparse
import wavio
import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt

FFT_WINDOW = 1024
NUM_BANDS = 16
WINDOW_OVERLAP = 0.25


def preprocess_audio(input_wav):
    data = input_wav.data
    print(f'preprocess audio shape: {data.shape}')
    print(f'  sample rate: {input_wav.rate} Hz')
    print(f'  {data.shape[0]} samples')
    print(f'  data type {data.dtype}')
    print(f'  {data.shape[1]} channels')

    if data.dtype == np.int8:
        data = data.astype(np.float) / 128.
    elif data.dtype == np.int16:
        data = data.astype(np.float) / 32768.
    else:
        print(f'Unhandled input audio type! {data.dtype}')
        sys.exit(-1)

    data = data.mean(axis=1)    # downmix audio tracks to mono

    assert len(data.shape) == 1
    assert np.max(data) <= 1.0
    assert np.min(data) >= -1.0
    return data


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Applies a vocoder effect to an input carrier audio signal frequency-modulated by an input'
                    'modulator signal.'
    )
    argparser.add_argument('-m', '--modulator',
                           help='Input modulator audio signal, in WAV format. Usually this is a human voice.',
                           required=True)
    argparser.add_argument('-c', '--carrier',
                           help='Input carrier audio signal, in WAV format. Use a harmonically-rich source, like'
                                'a piano playing a set of chords or a MIDI synthesized song.',
                           required=True)
    argparser.add_argument('-w', '--waveform',
                           help='Read the input audio files and plot their waveforms then exit.',
                           action='store_true')
    argparser.add_argument('-f', '--frequency',
                           help='Show the frequency spectrum of the first fft window.',
                           action='store_true')
    argparser.add_argument('-o', '--show_output',
                           help='Show the output synthesized waveform.',
                           action='store_true')
    argparser.add_argument('-g', '--voice_gain',
                           type=float,
                           help='Accentuate the higher-frequency components of the voice signal, to make the spoken'
                                'words sound clearer in the synthesized signal. Experiment with values from 10.0-100.0')

    args = argparser.parse_args()

    # 1.1) Read the modulator signal, which is usually a human voice.
    print(f'Reading input modulator file {args.modulator}...')
    modulator_wav = wavio.read(args.modulator)
    modulator_data = preprocess_audio(modulator_wav)
    print()

    # 1.2) Read the input carrier signal, which is usually a chord played on an instrument
    print(f'Reading input carrier file {args.carrier}...')
    carrier_wav = wavio.read(args.carrier)
    carrier_data = preprocess_audio(carrier_wav)
    print()

    # mismatched sample rates cause problems when recombining the synthesized signals
    if modulator_wav.rate != carrier_wav.rate:
        print('Input audio files have mismatching sample rates!')
        sys.exit(-1)

    if args.waveform:
        print('Plotting audio signal waveforms...')
        plt.clf()
        plt.subplot(211)
        plt.title('Modulator Waveform')
        plt.plot(np.arange(modulator_data.shape[0]) / modulator_wav.rate, modulator_data)
        plt.subplot(212)
        plt.title('Carrier Waveform')
        plt.plot(np.arange(carrier_data.shape[0]) / carrier_wav.rate, carrier_data)
        plt.xlabel('Time (sec)')
        plt.show()
        sys.exit(0)

    # match the lengths of both input signals by truncating the larger one
    total_samples = np.min([len(modulator_data), len(carrier_data)])
    print(f'total samples: {total_samples}')
    modulator_data = modulator_data[:total_samples]
    carrier_data = carrier_data[:total_samples]
    assert len(modulator_data) == len(carrier_data)

    # pad the data until it is a multiple of FFT_WINDOW
    padding_remainder = len(modulator_data) % FFT_WINDOW
    if padding_remainder != 0:
        padding = np.zeros(FFT_WINDOW - padding_remainder)
        modulator_data = np.append(modulator_data, padding)
        carrier_data = np.append(carrier_data, padding)
    assert len(modulator_data) % FFT_WINDOW == 0
    assert len(carrier_data) % FFT_WINDOW == 0
    total_samples = len(modulator_data)

    # process the audio file by looping through the signal data in FFT_WINDOW steps
    start = 0
    end = start + FFT_WINDOW
    output_signal = np.zeros(total_samples)
    hanning = np.hanning(FFT_WINDOW)
    while end <= total_samples:
        modulator_window = modulator_data[start:end]
        carrier_window = carrier_data[start:end]
        assert len(modulator_window) == FFT_WINDOW
        assert len(carrier_window) == FFT_WINDOW

        # the modulator and carrier signals are split into multiple frequency bands
        modulator_window = np.multiply(modulator_window, hanning)   # the taper the ends of the window signal using a
        carrier_window = np.multiply(carrier_window, hanning)       # hanning window. Combined with the overlapping
        modulator_fft = rfft(modulator_window)                      # FFT windows, this reduces the buzzing caused by
        carrier_fft = rfft(carrier_window)                          # discontinuities between reconstructed FFT windows.
        assert len(modulator_fft) == len(carrier_fft)

        if args.frequency:
            # fft is symmetric, so we only need to use half
            freqs = rfftfreq(FFT_WINDOW, d=1./modulator_wav.rate)
            plt.clf()
            plt.subplot(211)
            plt.title('Modulator FFT')
            plt.plot(freqs, np.abs(modulator_fft))
            plt.subplot(212)
            plt.title('Carrier FFT')
            plt.plot(freqs, np.abs(carrier_fft))
            plt.xlabel('Frequency (Hz)')
            plt.show()
            sys.exit(0)

        # the level of each modulator band (average value) serves as the gain for the carrier band
        band_width = int(len(modulator_fft) / NUM_BANDS)
        band_start = 0
        band_end = band_start + band_width
        synthesized_fft = np.empty(len(modulator_fft), dtype=np.complex128)
        while band_end <= len(modulator_fft):
            gain = np.mean(np.abs(modulator_fft[band_start:band_end]))

            if args.voice_gain:
                # to reduce the amount of "murmuring" for the output voice, boost the gain on higher frequencies,
                # making the voices sound a little clearer for the synthesized output.
                gain *= 1.0 + (args.voice_gain * band_start / len(modulator_fft))

            synthesized_fft[band_start:band_end] = gain * carrier_fft[band_start:band_end]
            # synthesized_fft[band_start:band_end] = modulator_fft[band_start:band_end]

            band_start = band_end
            band_end = band_start + band_width

        output_signal[start:end] += irfft(synthesized_fft).real

        start = int(start + WINDOW_OVERLAP * FFT_WINDOW)
        end = start + FFT_WINDOW

    normalizer = np.max(np.abs(output_signal))
    output_signal = 0.9 * output_signal / normalizer
    if args.show_output:
        plt.clf()
        plt.title('Synthesized Waveform')
        plt.plot(np.arange(output_signal.shape[0]) / modulator_wav.rate, output_signal)
        plt.show()
        sys.exit()

    print('Saving synthesized output...')
    output_signal = (32768. * output_signal).astype(np.int16)
    wavio.write('vocoded_output.wav', output_signal, modulator_wav.rate)
