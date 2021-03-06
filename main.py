import os
# shhhhh
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
#gl = util.read_sound('samples/output5000.wav')

GL_ITERS = 200

def prep_mel(path):
    with tf.compat.v1.Session() as sess:
        waveform = util.read_sound(path)
        ofreqs = sig.encode(waveform)
        omags, ophase = sig.topolar(ofreqs)

        both = np.matmul(minv, mel)
        melmags = tf.matmul(omags, tf.expand_dims(mel, 0))
        mags = tf.matmul(melmags, tf.expand_dims(minv, 0))
        freqs = sig.frompolar(mags, ophase)

        print('running griffin-lim for {} iterations...'.format(GL_ITERS))
        refreqs = sig.griffin_lim(freqs, iters=GL_ITERS, alpha=0.99)

        restored = sig.decode(refreqs)
        #sess.run(util.write_sound(restored, 'samples/output.wav'))

        #return sess.run([melmags, ofreqs])
        return sess.run(melmags), refreqs

def run_vocoder(melmags, refreqs, outpath):
    with tf.compat.v1.Session() as sess:
        res = voc.mel_loss(melmags, mel, refreqs)

        #optimizer = tf.compat.v1.train.AdagradOptimizer(0.01)
        #optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
        optimizer = tf.compat.v1.train.AdamOptimizer()

        train = optimizer.minimize(res.loss)
        init = tf.compat.v1.global_variables_initializer()

        sess.run(init)

        signal.signal(signal.SIGINT, signal_handler)
        for i in itertools.count():
            if sigint: break
            (loss, resonance), (err_sum, err_stft, err_mag), _ = sess.run(
                [(res.loss, res.loss_resonance),
                (res.err_sum, res.err_stft, res.err_mag),
                train])
            if i % 100 == 0: print(('Iter: {}; Loss: {:#.4G}, Repro Error: {:#.4G}; '
                '(Pieces: {:#.4G}, {:#.4G}, {:#.4G})')
                .format(i, loss, err_sum, err_stft, err_mag, resonance))

        restored = sig.decode(res.vfreq)
        print('writing to {}...'.format(outpath))
        sess.run(util.write_sound(restored, outpath))

def main():
    if len(sys.argv) not in [2,3]:
        print('need 2 or 3 arguments')
        print('usage: main.py infile.wav [outfile.wav]')
        return
    inpath = sys.argv[1]
    outpath = os.path.join(os.path.dirname(inpath), 'out.wav')
    if len(sys.argv) == 3:
        outpath = sys.argv[2]

    # uh... https://stackoverflow.com/a/60699372
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    print('loading {}...'.format(inpath))
    melmags, refreqs = prep_mel(inpath)
    #refreqs += 0.002*np.random.random(refreqs.shape) - 0.001
    tf.compat.v1.reset_default_graph()
    print('running vocoder...')
    run_vocoder(melmags, refreqs, outpath)

main()