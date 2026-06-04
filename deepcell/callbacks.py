"""Custom Callbacks for DeepCell"""


import timeit

import numpy as np

import tensorflow as tf


class InferenceTimer(tf.keras.callbacks.Callback):
    """Callback to log inference speed per epoch."""

    def __init__(self, samples=100):
        super().__init__()
        self._samples = int(samples)
        self._batch_times = []
        self._samples_seen = []
        self._timer = None

    def on_predict_begin(self, epoch, logs=None):
        self._batch_times = []
        self._samples_seen = []

    def on_predict_batch_begin(self, batch, logs=None):
        self._timer = timeit.default_timer()

    def on_predict_batch_end(self, batch, logs=None):
        t = timeit.default_timer() - self._timer
        self._batch_times.append(t)
        outputs = logs.get('outputs', np.empty((1,)))
        if isinstance(self.model.output_shape, list):
            outputs = outputs[0]
        self._samples_seen.append(outputs.shape[0])

    def on_predict_end(self, logs=None):
        total_samples = np.sum(self._samples_seen)

        per_sample = [t / float(s) for t, s in
                      zip(self._batch_times, self._samples_seen)]

        avg = np.mean(per_sample)
        std = np.std(per_sample)

        print(f'Average inference speed per sample for {total_samples} total samples: '
              f'{avg:0.5f}s ± {std:0.5f}s.')

    def on_epoch_end(self, epoch, logs=None):
        shape = tuple([self._samples] + list(self.model.input_shape[1:]))
        test_batch = np.random.random(shape)
        self.model.predict(test_batch, callbacks=self)
