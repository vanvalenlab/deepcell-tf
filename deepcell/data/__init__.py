"""Data utilities using ``tf.data``."""



def split_dataset(dataset, val_size, test_size=0):
    """
    Splits a dataset of type tf.data.Dataset into a training, validation, and
    optionally test dataset using given ratios. Fractions are rounded up to
    two decimal places.

    Inspired by: https://stackoverflow.com/a/59696126

    Args:
        dataset (tf.data.Dataset): the input dataset to split.
        val_size (float): the fraction of the validation data
            between 0 and 1.
        test_size (float): the fraction of the test data between 0
            and 1.

    Returns:
        (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset): a tuple of
            (training, validation, test).
    """
    val_percent = round(val_size * 100)
    if not 0 <= val_percent <= 100:
        raise ValueError('val_size must be ∈ [0,1].')

    test_percent = round(test_size * 100)
    if not 0 <= test_percent <= 100:
        raise ValueError('test_size must be ∈ [0,1].')

    if val_percent + test_percent >= 100:
        raise ValueError('sum of val_size and '
                         'test_size must be ∈ [0,1].')

    dataset = dataset.enumerate()
    # TODO: Will cause issues if there are fewer than 100 records
    val_dataset = dataset.filter(lambda f, data: f % 100 <= val_percent)
    train_dataset = dataset.filter(lambda f, data:
                                   f % 100 > test_percent + val_percent)
    test_dataset = dataset.filter(lambda f, data: f % 100 > val_percent and
                                  f % 100 <= val_percent + test_percent)

    # remove enumeration
    train_dataset = train_dataset.map(lambda f, data: data)
    val_dataset = val_dataset.map(lambda f, data: data)
    test_dataset = test_dataset.map(lambda f, data: data)
    return train_dataset, val_dataset, test_dataset


from deepcell.data import tracking
