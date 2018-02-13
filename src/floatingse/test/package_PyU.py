import unittest

import column_PyU
import mapMooring_PyU
import semiPontoon_PyU
import substructure_PyU

import numpy as np
import numpy.testing as npt

def suiteAll():
    suite = unittest.TestSuite( (column_PyU.suite(),
                                 mapMooring_PyU.suite(),
                                 semiPontoon_PyU.suite(),
                                 substructure_PyU.suite()
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suiteAll())
        
