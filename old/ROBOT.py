import tensorflow as tf
import sigproc as sig
import util

waveform = util.read_sound('samples/arctic_raw_16k.wav')
#gl = util.read_sound('samples/output5000.wav')

with tf.compat.v1.Session() as sess:
    freqs = sig.encode(waveform)
    omags, ophase = sig.topolar(freqs);
    dphase = sig.consec_diff(ophase)
    util.cplot(sess.run(sig.frompolar(omags, dphase)), True)
    iphase = sig.consec_integ(dphase)
    refreqs = sig.frompolar(omags, dphase)
    restored = sig.decode(refreqs)
    sess.run(util.write_sound(restored, 'samples/output.wav'))

    #print(freqs.shape)
    #util.diff_plot(waveform)
    '''
    glfreqs = sess.run(sig.encode(gl))
    util.diff_plot(gl, True)
    refreqs = sig.griffin_lim(freqs, iters=50)
    restored = sig.decode(refreqs)
    sess.run(util.write_sound(restored, 'samples/output.wav'))'''
