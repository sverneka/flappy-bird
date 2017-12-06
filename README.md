# flappy-bird

* [Blog](https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html)
* [Original Repo](https://github.com/yanpanlau/Keras-FlappyBird)

# Dependencies:

* Keras
* pygame
* scikit-image
* h5py

# How to Run?

**CPU only**

```
python qlearn.py -m "Run"
```

**GPU version (Theano)**

```
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=0.2 python qlearn.py -m "Run"
```

**Train**

Everything as before, the argument is "Train" instead of "Run".
