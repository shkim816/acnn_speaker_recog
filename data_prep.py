import numpy as np
import os
import time
import shutil

DB_dir = '../voxceleb1_DB/'
save_dir = '../voxceleb1/'

if __name__ == '__main__':
    for r, _, fs in os.walk(DB_dir):
        for f in fs:
            if f[-3:] == 'wav':

                law_wav,samplingfs = sf.read(r + '/' + f)
                if samplingfs != 16000:
                    raise ValueError('Sampling frequency error')

                law_wav = law_wav / max(abs(law_wav))
                spec = np.abs(libcore.stft(law_wav, n_fft = 512, hop_length = 160, win_length = 400))

                base = '/'.join(r.split('/')[-2:])+'/'
                np.save(save_dir + base + f[:-3] + 'npy', spec)

print('=====All Done=====')
