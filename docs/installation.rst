Installation
------------

.. admonition:: prerequisites
   :class: warning

	General: NumPy, SciPy, OpenMDAO

	Supporting python packages: Sphinx, PyOpt

Clone the repository at `<https://github.com/WISDEM/FloatingSE>`


To install FloatingSE, first activate the OpenMDAO environment and then install with the following command.

.. code-block:: bash

   $ plugin install

To check if installation was successful try to import the module from within an activated OpenMDAO environment:

.. code-block:: bash

    $ python

.. code-block:: python

	> import mooring
	> import tower_RNA
	> import spar

For software issues please use `<https://github.com/WISDEM/FloatingSE/issues>`_.  For functionality and theory related questions and comments please use the NWTC forum for `Systems Engineering Software Questions <https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002>`_.

.. only:: latex

    An HTML version of this documentation that contains further details and links to the source code is available at `<http://wisdem.github.io/FloatingSE>`_
