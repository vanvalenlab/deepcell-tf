deepcell.datasets
=================
Basic datasets can be loaded according to the following example.

.. code-block:: python

    from deepcell.datasets.cytoplasm import hela

    # path can be modified to determine where the dataset is stored locally
    (X_train,y_train),(X_test,y_test) = hela.load_data(path='hela_cytoplasm.npz',test_size=0.1)

    # Details regarding dataset collection are stored in the metadata attribute
    print(hela.metadata)


Tracked datasets have a dedicated load function to handle the different data structure.

.. code-block:: python

    from deepcell.datasets.tracked import hela

    # path can be modified to determine where the dataset is stored locally
    (X_train,y_train),(X_test,y_test) = hela.load_data(path='hela_tracked.npz',test_size=0.1)

.. contents:: Contents
    :local:

deepcell.datasets.cytoplasm
---------------------------

.. automodule:: deepcell.datasets.cytoplasm
    :members:
    :undoc-members:
    :show-inheritance:

deepcell.datasets.phase
-----------------------

.. automodule:: deepcell.datasets.phase
    :members:
    :undoc-members:
    :show-inheritance:

deepcell.datasets.tracked
-------------------------

.. automodule:: deepcell.datasets.tracked
    :members:
    :undoc-members:
    :show-inheritance:


Module contents
---------------

.. automodule:: deepcell.datasets
    :members:
    :undoc-members:
    :show-inheritance:
