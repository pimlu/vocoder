import tensorflow as tf
import sigproc as sig
import util

class VocoderResult:
    vfreq = None
    loss = None
    err_sum = None
    err_stft = None
    err_mag = None
    loss_resonance = None

def groups(n):
    for i in range(1, n//2):
        yield range(i,n,i)

# the idea of this was, to add a loss based on the euclidean norm
# of vectors of magnitudes on harmonic frequencies (e.g 100, 200, 300...)
# the hopeful result being, it sounds more natural because it rewards
# placing the magnitude of the sound on the harmonic buckets (because e.g
# the L2 norm of a vector is less than the sum). But it doesn't really
# sound better, so I keep beta = 0 (the term for this loss)
def resonant_loss(vmag):
    n = vmag.get_shape()[-1]
    losses = []
    for g in groups(n):
        grouped = tf.gather(vmag, g, axis=-1)
        norms = tf.norm(tensor=grouped, axis=-1, ord=2)
        losses.append(tf.reduce_mean(input_tensor=norms, axis=-1))
    return tf.reduce_mean(input_tensor=losses)

def mel_loss(melmags, mel, ifreq, flen=512, fstep=None):
    fstep = sig.getstep(flen, fstep)
    alpha = 0.5

    emel = tf.expand_dims(mel, 0)
    '''freq_real = tf.Variable(ifreq.real, name='freq_real', dtype=tf.float32)
    freq_imag = tf.Variable(ifreq.imag, name='freq_imag', dtype=tf.float32)
    vfreq = tf.complex(freq_real, freq_imag)'''
    imag, iphase = sig.topolar(tf.convert_to_tensor(value=ifreq))
    #vmag, vphase: polar variables for state undergoing gradient descent
    vmag = tf.Variable(imag, name='f_mag', dtype=tf.float32)
    util.pshape(tf.convert_to_tensor(value=ifreq), 'ifreq')
    vphase = tf.Variable(iphase, name='f_phase', dtype=tf.float32)
    vfreq = sig.frompolar(vmag, vphase)

    refreq = sig.encode(sig.decode(vfreq, flen, fstep), flen, fstep)
    err_stft = tf.math.real(tf.norm(tensor=refreq - vfreq, ord=2))

    #vmag, _ = sig.topolar(vfreq)
    err_mag = tf.norm(tensor=tf.matmul(vmag, emel) - melmags, ord=2)

    err_sum = alpha*err_stft + (1-alpha)*err_mag

    beta = 0.0 #0.1
    if beta:
        loss_resonance = resonant_loss(vmag)
    else:
        loss_resonance = tf.convert_to_tensor(value=0.0)
    loss = err_sum + beta*loss_resonance

    res = VocoderResult()
    res.vfreq = vfreq
    res.loss = loss
    res.err_sum = err_sum
    res.err_stft = err_stft
    res.err_mag = err_mag
    res.loss_resonance = loss_resonance

    return res
