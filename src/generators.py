import threading
import numpy as np

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):  # Py3
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def generator(seq, X, y, batch_size, hooks=None):
    while True:
         ix = np.random.randint(0, len(X), batch_size)
         X_aug = seq.augment_images(X[ix])
         y_aug = seq.augment_images(y[ix], hooks=hooks)
         yield X_aug, y_aug

@threadsafe_generator
def combine_generator(gen1, gen2):
    while True:
        yield(gen1.next(), gen2.next())