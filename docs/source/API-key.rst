DeepCell API Key
================

DeepCell models and training datasets are licensed under a `modified Apache license <http://www.github.com/vanvalenlab/deepcell-tf/blob/master/LICENSE>`_ for non-commercial academic use only. An API key for accessing datasets and models can be obtained at https://users.deepcell.org/login/.

For more information about datasets published through DeepCell, please see :doc:`/data-gallery/index`.

API Key Usage
-------------

The token that is issued by users.deepcell.org should be added as an environment variable through one of the following methods:

1. Save the token in your shell config script (e.g. ``.bashrc``, ``.zshrc``, ``.bash_profile``, etc.)

.. code-block:: bash

    export DEEPCELL_ACCESS_TOKEN=<token-from-users.deepcell.org>

2. Save the token as an environment variable during a python session. Please be careful to avoid commiting your token to any public repositories.

.. code-block:: python

    import os

    os.environ.update({"DEEPCELL_ACCESS_TOKEN": "<token-from-users.deepcell.org>"})

