import tensorflow as tf
import numpy as np
import sigproc as sig
import itertools


import vocoder as voc
import util


import signal
import sys
import os

#usage: argv isd ipath, opath. call on librespeech directory
#and an empty directory as opath, it will match tree with images

mel, minv = sig.melbank()
def traverse(ipath, opath):
    print(ipath, opath)
    for root, dirs, files in os.walk(ipath):
        rel = os.path.relpath(root, ipath)
        for d in dirs:
            os.makedirs(os.path.join(opath, rel, d), exist_ok=True)
        for f in files:
            if f[-5:] != '.flac': continue
            iflac = os.path.join(ipath, rel, f)
            opng = os.path.join(opath, rel, f[:-5]+'.png')
            if os.path.isfile(opng): continue
            #print(iflac, opng)
            yield (iflac, opng)

with tf.compat.v1.Session() as sess:

    '''ds = tf.data.Dataset.from_generator(traverse, (tf.string, tf.string), args=sys.argv[1:3])
    
    ds = ds.map(lambda iflac, opng:
        (tf.py_func(util.read_flac, [iflac], tf.float32), opng))
    waveform, opng = ds.make_one_shot_iterator().get_next()
    write = util.waveform_to_png(waveform, opng)
    try:
        while True:
            print(sess.run([write, opng]))
    except tf.errors.OutOfRangeError:
        pass'''

    
    pl_iflac = tf.compat.v1.placeholder(tf.string)
    pl_opng = tf.compat.v1.placeholder(tf.string)
    convert = util.flac_to_png(pl_iflac, pl_opng)
    for (iflac, opng) in traverse(*sys.argv[1:3]):
        print(opng)
        try:
            sess.run(convert, feed_dict={pl_iflac: iflac, pl_opng: opng})
        except Exception as e:
            print('FAILED!', e)
