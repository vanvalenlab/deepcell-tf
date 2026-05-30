"""Utilities for training neural nets"""


import numpy as np
from tensorflow.keras import callbacks
from tensorflow.python.client import device_lib


def get_callbacks(model_path,
                  save_weights_only=False,
                  lr_sched=None,
                  tensorboard_log_dir=None,
                  reduce_lr_on_plateau=False,
                  monitor='val_loss',
                  verbose=1):
    """Returns a list of callbacks used for training

    Args:
        model_path: (str) path for the ``h5`` model file.
        save_weights_only: (bool) if True, then only the model's weights
            will be saved.
        lr_sched (function): learning rate scheduler per epoch.
            from `~deepcell.utils.train_utils.rate_scheduler`.
        tensorboard_log_dir (str): log directory for tensorboard.
        monitor (str): quantity to monitor.
        verbose (int): verbosity mode, 0 or 1.

    Returns:
        list: a list of callbacks to be passed to ``model.fit()``
    """
    cbs = [
        callbacks.ModelCheckpoint(
            model_path, monitor=monitor,
            save_best_only=True, verbose=verbose,
            save_weights_only=save_weights_only),
    ]

    if lr_sched:
        cbs.append(callbacks.LearningRateScheduler(lr_sched))

    if reduce_lr_on_plateau:
        cbs.append(
            callbacks.ReduceLROnPlateau(
                monitor=monitor, factor=0.1,
                patience=10, verbose=verbose,
                mode='auto', min_delta=0.0001,
                cooldown=0, min_lr=0)
        )

    if tensorboard_log_dir:
        cbs.append(callbacks.TensorBoard(log_dir=tensorboard_log_dir))

    return cbs


def rate_scheduler(lr=.001, decay=0.95):
    """Schedule the learning rate based on the epoch.

    Args:
        lr (float): initial learning rate
        decay (float): rate of decay of the learning rate

    Returns:
        function: A function that takes in the epoch
        and returns a learning rate.
    """
    def output_fn(epoch):
        epoch = np.int_(epoch)
        new_lr = lr * (decay ** epoch)
        return new_lr
    return output_fn


def count_gpus():
    """Get the number of available GPUs.

    Returns:
        int: count of GPUs as integer
    """
    devices = device_lib.list_local_devices()
    gpus = [d for d in devices if d.name.lower().startswith('/device:gpu')]
    return len(gpus)
