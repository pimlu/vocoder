# Gradient descent-based vocoder
##### Recovers phase of Mel spectrograms using gradient descent

The basic idea of this project is to try to recover audio data from a spectrogram. There already exists the [Griffin-Lim algorithm family](https://ltfat.github.io/notes/ltfatnote021.pdf) to do this, and it can even perfectly reconstruct audio from the spectrogram in some cases.  However, the challenge in my case is I binned the audio into the [Mel scale](https://en.wikipedia.org/wiki/Mel_scale), compressing the magnitude data by a factor of 2 and making the original unrecoverable.  Griffin-Lim can only be used by creating a blurry version of the original spectrogram from the compressed Mel bins.

How can we (just slightly) improve on Griffin-Lim in this case? Well, there exist state of the art models, but what I tried was to use gradient descent on the uncompressed data, using STFT reconstruction error and error with respect to the Mel spectrogram summed together as loss. This allows it to estimate the underlying magnitudes even though they are underspecified by the Mel spectrogram.

## How to run

In a tensorflow 2 environment, run `python main.py samples/arctic_raw_16k.wav my/out/path.wav`, press ctrl-C to stop processing and dump the output. This demo will convert the input file into a compressed Mel spectrogram, then try to recover the wav using gradient descent.

## Demo files (use headphones)

The difference is marginal. It actually likes the output of Griffin-Lim as parameter initialization, so the process is: run Griffin-Lim for a few iterations, then run gradient descent.

| File | G-L iters | G.D. iters | notes
| --- | --- | -- | ---
| [arctic_raw_16k.wav](samples/arctic_raw_16k.wav) | -- | -- | original unprocessed file
| [griffinlim.wav](samples/griffinlim.wav) | 5000 | -- | ~25s griffin lim processing
| [gl200.wav](samples/gl200.wav) | 200 | -- | short griffin lim processing
| [graddescent.wav](samples/graddescent.wav) | 200 | 5000 | ~25s grad descent processing

Objectively, the reconstruction loss immediately improves when you feed Griffin-Lim output into gradient descent (even after 5000 G-L iterations). Subjectively, I think the Griffin-Lim output sounds a bit phasier/robotic (you can hear it better when you compare to gl200.wav) than gradient descent, but you wouldn't be able to tell without headphones probably.

One caveat is both methods start to drop in subjective quality (even though loss improves) when you let it run for too long (20000+ iterations). I'm not sure why.
## Other files

This repo is a mishmash of old files to be honest, lol. Maybe some of the functions in `util.py` would be useful to you, e.g. for visualizing STFT data. The story is, I made this back in fall 2018 and then upgraded it in 2021 to Tensorflow 2 (kind of, it still uses the graph API) so I could push it online.

`gen_dataset.py` is an unrelated file I used to make PNGs out of the STFT of sound files, which may or may not work.

There is some random legacy stuff in the `old/` directory.
