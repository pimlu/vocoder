import tensorflow as tf
import numpy as np
import sigproc as sig
import itertools

import vocoder as voc
import util


import signal
import sys
sigint = False
def signal_handler(sig, frame):
    global sigint
    if sigint:
        sys.exit(1)
    else:
        sigint = True

mel, minv = sig.melbank()

def prep_mel(truth, gritty):
    with tf.compat.v1.Session() as sess:
        waveform = util.read_sound(truth)
        ofreqs = sig.encode(waveform)
        omags, ophase = sig.topolar(ofreqs)

        melmags = tf.matmul(omags, tf.expand_dims(mel, 0))

        gritty_wf = util.read_sound(gritty)
        gritty_freqs = sig.encode(gritty_wf)

        return sess.run([melmags, gritty_freqs])

def run_vocoder(melmags, refreqs):
    with tf.compat.v1.Session() as sess:
        res = voc.mel_loss(melmags, mel, refreqs)

        optimizer = tf.compat.v1.train.AdamOptimizer()

        train = optimizer.minimize(res.loss)
        init = tf.compat.v1.global_variables_initializer()

        sess.run(init)

        signal.signal(signal.SIGINT, signal_handler)
        for i in itertools.count():
            if sigint: break
            (loss, resonance), errs, _ = sess.run(
                [(res.loss, res.loss_resonance),
                (res.err_sum, res.err_stft, res.err_mag),
                train])
            print(('Iter: {}; Loss: {:#.4G}, Error: {:#.4G}; '
                '(Pieces: {:#.4G}, {:#.4G}, {:#.4G})')
                .format(i, loss, *errs, resonance))

        restored = sig.decode(res.vfreq)
        sess.run(util.write_sound(restored, 'samples/flown/polished.wav'))
print('loading file...')
melmags, refreqs = prep_mel('samples/flown/Gound_truth.wav', 'samples/flown/FloWaveNet.wav')
#refreqs += 0.002*np.random.random(refreqs.shape) - 0.001
tf.compat.v1.reset_default_graph()
print('running vocoder...')
run_vocoder(melmags, refreqs)
