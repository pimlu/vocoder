import tensorflow as tf
import numpy as np
import sigproc as sig
import util

waveform = util.read_sound('samples/arctic_raw_16k.wav')
#gl = util.read_sound('samples/output5000.wav')

with tf.compat.v1.Session() as sess:
    freqs = sig.encode(waveform)
    omags, ophase = sig.topolar(freqs)
    mel, minv = sig.melbank()
    both = np.matmul(minv, mel).astype('f')
    mags = tf.matmul(omags, tf.expand_dims(tf.constant(both), 0))
    freqs = sig.frompolar(mags, ophase)
    '''dphase = sig.consec_diff(ophase)
    util.cplot(sess.run(sig.frompolar(omags, dphase)), True)
    iphase = sig.consec_integ(dphase)
    refreqs = sig.frompolar(omags, iphase)
    restored = sig.decode(refreqs)
    sess.run(util.write_sound(restored, 'samples/output.wav'))'''

    #print(freqs.shape)
    #util.diff_plot(waveform)

    refreqs = sig.griffin_lim(freqs, iters=200, alpha=0.99)
    restored = sig.decode(refreqs)
    sess.run(util.write_sound(restored, 'samples/output.wav'))
    '''
    glfreqs = sess.run(sig.encode(gl))
    util.diff_plot(gl, True)'''
