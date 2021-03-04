import tensorflow as tf
import numpy as np
from numpy import pi

def colorize(z):
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + pi)  / (2 * pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2)
    return c

def cplot(w, show=False):
    import pylab as plt
    from colorsys import hls_to_rgb

    plt.figure()
    img = colorize(np.squeeze(w))
    plt.imshow(img)
    if show: plt.show()

def diff_plot(wf, show=False):
    mags, phase = voc.topolar(voc.encode(wf))
    mags = mags*mags
    phase = voc.consec_diff(phase)
    fs = sess.run(voc.frompolar(mags, phase))
    cplot(fs, show)

def read_sound(path, fmt='wav', rate=16000, channels=1):
    if fmt != 'wav': raise 'unsupported in tf2'
    audio_binary = tf.io.read_file(path)
    waveform, sr = tf.audio.decode_wav(audio_binary, desired_channels=channels)
    # TODO use this outside graph
    #if not tf.math.reduce_all(sr == rate): raise 'sample rate mismatch'
    return tf.transpose(a=waveform)

def write_sound(waveform, path, fmt='wav', rate=16000):
    if fmt != 'wav': raise 'unsupported in tf2'
    output_binary = tf.audio.encode_wav(
        tf.transpose(a=waveform),
        sample_rate=rate
    )
    return tf.io.write_file(path, output_binary)

def log_eps(x, eps=0.001):
    return tf.math.log(x+eps)-tf.math.log(eps)
def unlog_eps(x, eps=0.001):
    return tf.exp(x+tf.math.log(eps))-eps
def quartiles(x):
    #not needed anymore, probably works in tf2?
    import tensorflow_probability as tfp
    qs = [0., 25., 50., 75., 100.]
    return tfp.stats.percentile(x, qs)
def write_png(mags, path, scale=13.):
    mags = tf.convert_to_tensor(value=mags)
    scale = tf.convert_to_tensor(value=scale)

    mags = log_eps(mags) / scale
    i16 = 2.**16
    maximum = tf.reduce_max(input_tensor=mags)
    valid = maximum <= 1 - 1/i16
    
    valid = tf.Assert(valid, [maximum])
    with tf.control_dependencies([valid]):
        words = tf.cast(mags * i16, dtype=tf.uint16)
    words = tf.expand_dims(tf.transpose(a=tf.squeeze(words, 0)), -1)

    pshape(words)
    encoded = tf.image.encode_png(words)
    write = tf.io.write_file(path, encoded)
    return write

def read_png(path, scale=13.):
    scale = tf.convert_to_tensor(value=scale)
    encoded = tf.io.read_file(path)
    words = tf.image.decode_png(encoded, dtype=tf.uint16)
    words = tf.expand_dims(tf.transpose(a=tf.squeeze(words, -1)), 0)
    i16 = 2.**16
    mags = tf.cast(words, dtype=tf.float32) / i16
    mags = unlog_eps(mags * scale)
    return mags

def read_flac(iflac):
    import soundfile as sf
    data, sr = sf.read(iflac)
    waveform = np.expand_dims(data.astype('f'), 0)
    return waveform

def waveform_to_png(waveform, opng):
    import sigproc as sig
    ofreqs = sig.encode(waveform)
    omags, ophase = sig.topolar(ofreqs)

    output = write_png(omags, opng)
    return output


def flac_to_png(iflac, opng):
    waveform = tf.compat.v1.py_func(read_flac, [iflac], tf.float32)
    return waveform_to_png(waveform, opng)

def pshape(t, name='tensor'):
    print('{}: {} {}'.format(name, t.dtype, t.get_shape()))
