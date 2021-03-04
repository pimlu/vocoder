import tensorflow as tf
import numpy as np

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.check_COLA.html#scipy.signal.check_COLA
def getstep(flen, fstep):
    return fstep if fstep is not None else flen//4
def encode(signal, flen=512, fstep=None):
    # signal: A batch of float32 time-domain signals in the range [-1, 1] with shape
    # [batch_size, signal_length]. Both batch_size and signal_length may be unknown.
    fstep = getstep(flen, fstep)
    # `stfts` is a complex64 Tensor representing the Short-time Fourier Transform of
    # each signal in `signals`. Its shape is [batch_size, ?, fft_unique_bins]
    # where fft_unique_bins = fft_length // 2 + 1.
    return tf.signal.stft(
        signal,
        frame_length=flen, frame_step=fstep, fft_length=flen
    )
def decode(freqs, flen=512, fstep=None):
    fstep = getstep(flen, fstep)

    return tf.signal.inverse_stft(
        freqs,
        frame_length=flen, frame_step=fstep, fft_length=flen,
        window_fn=tf.signal.inverse_stft_window_fn(fstep)
    )
def topolar(complex):
    return abs(complex), tf.math.angle(complex)
def frompolar(mags, angle):
    return tf.cast(mags, tf.complex64)*tf.complex(tf.cos(angle), tf.sin(angle))
def consec_diff(t, axis=1, len=3):
    pads = lambda p: [[0,0] if i!=axis else p for i in range(len)]
    diff = tf.pad(tensor=t, paddings=pads([0,1])) - tf.pad(tensor=t, paddings=pads([1,0]))
    return diff.__getitem__([*([slice(None)]*axis), slice(-1), ...])
def consec_integ(t, axis=1, len=3):
    return tf.cumsum(t, axis=axis)
def dephase(phase, flen=512, fstep=None):
    fstep = getstep(flen, fstep)

    diff = consec_diff(phase)

def tf_reduce_angle(phase):
    return tf.math.floormod(phase+np.pi, 2*np.pi) - np.pi
def np_reduce_angle(phase):
    return np.mod(phase+np.pi, 2*np.pi) - np.pi
def melbank(rate=16000, flen=512, n_mels=128):
    import vendor
    bank = vendor.get_filterbanks(n_mels, flen, rate).astype('f').T
    return bank, np.linalg.pinv(bank)
def griffin_lim(freqs, flen=512, fstep=None, iters=100, alpha=0, rand=True):
    fstep = getstep(flen, fstep)

    freqs = tf.convert_to_tensor(value=freqs)
    sess = tf.compat.v1.get_default_session()

    mags, phase = sess.run(topolar(freqs))
    if rand:
        phase = np.random.uniform(-np.pi, np.pi, phase.shape).astype('f')

    iphase = tf.compat.v1.placeholder(tf.float32)

    ifreqs = frompolar(mags, iphase)
    refreqs = encode(decode(ifreqs, flen, fstep), flen, fstep)
    remag, rephase = topolar(refreqs)

    pre_phase = phase
    for i in range(iters):
        # if i % 100 == 0: print('Iter: {}'.format(i))
        post_phase = sess.run(rephase, feed_dict={ iphase: phase })
        # momentum term for griffin lim which boosts performance from
        # "A FAST GRIFFIN-LIM ALGORITHM". they don't say momentum anywhere in
        # the paper though, that's my interpretation
        if alpha and i>5:
            vel = np_reduce_angle(phase-pre_phase)
            pre_phase, phase = phase, post_phase + alpha*vel
        else:
            pre_phase = phase = post_phase

    return sess.run(frompolar(mags, post_phase))
