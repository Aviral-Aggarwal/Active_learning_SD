import os
import wave
from contextlib import closing

folderpath = r'/home/aviral/Desktop/SP/pyannote-audio/Data/musan/noise/free-sound/'   # make sure to put the 'r' in front
noise_files = [os.path.join(folderpath, name) for name in os.listdir(folderpath)]
outfile = r"/home/aviral/Desktop/SP/Active_learning_SD/APPENDEDnoise.wav"

with closing(wave.open(outfile, 'wb')) as output:

    # find sample rate from first file
    with closing(wave.open(noise_files[0])) as w:
        output.setparams(w.getparams())

    # write each file to output
    for infile in noise_files:
        with closing(wave.open(infile)) as w:
            output.writeframes(w.readframes(w.getnframes()))