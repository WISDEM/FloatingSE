import unittest

import cylinder_PyU
import mapMooring_PyU
import semiPontoon_PyU
import semi_PyU
import spar_PyU
import turbine_PyU

import numpy as np
import numpy.testing as npt
import sys
import random
import itertools

def suiteAll():
    suite = unittest.TestSuite( (cylinder_PyU.suite(),
                                 mapMooring_PyU.suite(),
                                 semiPontoon_PyU.suite(),
                                 semi_PyU.suite(), 
                                 spar_PyU.suite(),
                                 turbine_PyU.suite()
    ) )
    return suite


if __name__ == '__main__' and __package__ is None:
    __package__ = 'src.test'
    unittest.TextTestRunner().run(suiteAll())
        
