deepcell.applications package
=============================

.. automodule:: deepcell.applications

``deepcell.applications`` enables direct utilization of a pre-defined model with the option of loading pre-trained weights.

For example, :mod:`deepcell.applications.nuclear_segmentation` could be used as shown below.

.. code-block:: python

    # X_test is pre-loaded data

    from deepcell.applications.nuclear_segmentation import NuclearSegmentationModel
    model = NuclearSegmentationModel(input_shape=tuple(X_test.shape[1:]), backbone='resnet50', use_pretrained_weights=True)

    predictions = model.predict(X_test)

.. contents:: Contents
    :local:

deepcell.applications.cell_tracking module
------------------------------------------
.. automodule:: deepcell.applications.cell_tracking
    :members:
    :undoc-members:
    :show-inheritance:

deepcell.applications.nuclear_segmentation module
-------------------------------------------------
.. automodule:: deepcell.applications.nuclear_segmentation
    :members:
    :undoc-members:
    :show-inheritance:

deepcell.applications.label_detection module
--------------------------------------------
.. automodule:: deepcell.applications.label_detection
    :members:
    :undoc-members:
    :show-inheritance:

deepcell.applications.scale_detection module
--------------------------------------------
.. automodule:: deepcell.applications.scale_detection
    :members:
    :undoc-members:
    :show-inheritance:

deepcell.applications.phase_segmentation module
-----------------------------------------------
.. automodule:: deepcell.applications.phase_segmentation
    :members:
    :undoc-members:
    :show-inheritance:

deepcell.applications.fluorescent_cytoplasm_segmentation module
---------------------------------------------------------------
.. automodule:: deepcell.applications.fluorescent_cytoplasm_segmentation
    :members:
    :undoc-members:
    :show-inheritance: