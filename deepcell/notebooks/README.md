# deepcell.notebooks

`deepcell-tf.notebooks` provides simple methods for generating a parameterized training or visualization Jupyter notebook.  This can allow for rapid testing with different model architectures and parameters.

```python
from deepcell.notebooks import train

notebook_path = train.make_notebook(
    data,
    model_name='resnet50',
    train_type='conv',
    field_size=61,
    ndim=2,
    optimizer='sgd',
    epochs=10,
    normalization='std',
    transform='watershed')
```
