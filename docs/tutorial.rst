.. _tutorial-label:

.. currentmodule:: plant_costsse.docs.examples.example


Tutorial
--------

Tutorial for FloatingSE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an example, let us study the OC3 spar structure for a floating offshore wind turbine.

The first step is to import the relevant files and set up the component.

.. literalinclude:: examples/example.py
    :start-after: # 1 ---
    :end-before: # 1 ---

The model relies on some turbine and rotor as well as environmental input parameters that must be specified.  In this case, the tower outer diameters, length, and mass, and the rotor-nacelle-assembly (RNA) mass, center of gravity, and rotor diameter need to be specified. For plant inputs, the number of sections in the spar, water depth, wind reference speed and height, and elevations of spar sections need to be specified. User inputs regarding the material properties of the spar and the mooring system is available if there is a desire to calibrate the model to some known parameters.In addition, the user can choose to specify a stiffener, or let the tool optimize on the best stiffener dimension based on curve fits.

.. literalinclude:: examples/example.py
    :start-after: # 2 ---
    :end-before: # 2 ---

We can now evaluate the spar masses.

.. literalinclude:: examples/example.py
    :start-after: # 3 ---
    :end-before: # 3 ---

We then print out the resulting mass values.

.. literalinclude:: examples/example.py
    :start-after: # 4 ---
    :end-before: # 4 ---

The result is:

>>> total spar structure mass: 1809078 kg
>>> shell mass: 1681870
>>> bulkhead mass: 44484
>>> stiffener mass: 82724

However, the safety constraint in the last section of the spar was not met. Optimization with appropriate constraints will remedy this problem.