import tensorflow as tf
import vocoder as voc

audio_binary = tf.io.read_file('samples/arctic_raw_16k.wav')
waveform = tf.contrib.ffmpeg.decode_audio(audio_binary, file_format='wav', samples_per_second=16000, channel_count=1)
waveform = tf.transpose(a=waveform)
sess = tf.compat.v1.Session()
print(sess.run([
    waveform,
    tf.shape(input=waveform),
    tf.contrib.distributions.percentile(waveform, 25.0),
    tf.reduce_max(input_tensor=waveform)
    ]))
waveform =  voc.decode(voc.encode(waveform))
print(sess.run([
    waveform,
    tf.shape(input=waveform),
    tf.contrib.distributions.percentile(waveform, 25.0),
    tf.reduce_max(input_tensor=waveform)
    ]))
output_binary = tf.contrib.ffmpeg.encode_audio(
    tf.transpose(a=waveform),
    file_format='wav',
    samples_per_second=16000
)
sess.run(tf.io.write_file('samples/output.wav', output_binary))

sess.close()