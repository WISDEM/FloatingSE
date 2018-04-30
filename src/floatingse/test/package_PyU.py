import unittest

import column_PyU
import map_mooring_PyU
import floating_loading_PyU
import substructure_PyU

import numpy as np
import numpy.testing as npt

def suiteAll():
    suite = unittest.TestSuite( (column_PyU.suite(),
                                 map_mooring_PyU.suite(),
                                 floating_loading_PyU.suite(),
                                 substructure_PyU.suite()
    ) )
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suiteAll())
        
