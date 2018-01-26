import numpy as np
import numpy.testing as npt
import unittest
import floatingse.mapMooring as mapMooring
from floatingse.cylinder import CylinderGeometry

from commonse import gravity as g

def myisnumber(instr):
    try:
        float(instr)
    except:
        return False
    return True

myones = np.ones((100,))
truth='---------------------- LINE DICTIONARY ---------------------------------------\n' + \
'LineType  Diam      MassDenInAir   EA            CB   CIntDamp  Ca   Cdn    Cdt\n' + \
'(-)       (m)       (kg/m)        (N)           (-)   (Pa-s)    (-)  (-)    (-)\n' + \
'chain   0.05   28.00867   118593500.0   0.65   1.0E8   0.6   -1.0   0.05\n' + \
'---------------------- NODE PROPERTIES ---------------------------------------\n' + \
'Node Type X     Y    Z   M     V FX FY FZ\n' + \
'(-)  (-) (m)   (m)  (m) (kg) (m^3) (kN) (kN) (kN)\n' + \
'1   FIX   175.0   0.0   depth   0.0   0.0   #   #   #\n' + \
'2   VESSEL   11.0   0.0   -10.0   0.0   0.0   #   #   #\n' + \
'---------------------- LINE PROPERTIES ---------------------------------------\n' + \
'Line    LineType  UnstrLen  NodeAnch  NodeFair  Flags\n' + \
'(-)      (-)       (m)       (-)       (-)       (-)\n' + \
'1   chain   416.0   1   2\n' + \
'---------------------- SOLVER OPTIONS-----------------------------------------\n' + \
'Option\n' + \
'(-)\n' + \
'help\n' + \
' integration_dt 0\n' + \
' kb_default 3.0e6\n' + \
' cb_default 3.0e5\n' + \
' wave_kinematics\n' + \
'inner_ftol 1e-5\n' + \
'inner_gtol 1e-5\n' + \
'inner_xtol 1e-5\n' + \
'outer_tol 1e-3\n' + \
' pg_cooked 10000 1\n' + \
' outer_fd\n' + \
' outer_bd\n' + \
' outer_cd\n' + \
' inner_max_its 200\n' + \
' outer_max_its 600\n' + \
'repeat 120 240\n' + \
' krylov_accelerator 3\n' + \
' ref_position 0.0 0.0 0.0\n'

class TestMapMooring(unittest.TestCase):
    
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params['wall_thickness'] = np.array([0.5, 0.5, 0.5])
        self.params['outer_diameter'] = 2*np.array([10.0, 10.0, 10.0])
        self.params['section_height'] = np.array([20.0, 30.0])
        self.params['freeboard'] = 15.0
        self.params['fairlead'] = 10.0
        self.params['fairlead_offset_from_shell'] = 1.0

        self.params['water_density'] = 1025.0 #1e3
        self.params['water_depth'] = 218.0 #100.0

        self.params['scope_ratio'] = 2.0
        self.params['mooring_diameter'] = 0.05
        self.params['anchor_radius'] = 175.0
        self.params['number_of_mooring_lines'] = 3
        self.params['mooring_type'] = 'chain'
        self.params['anchor_type'] = 'suctionpile'
        self.params['drag_embedment_extra_length'] = 300.0
        self.params['max_offset'] = 10.0

        self.params['mooring_cost_rate'] = 1.1

        self.params['tower_base_radius'] = 4.0
        
        self.set_geometry()

        self.mymap = mapMooring.MapMooring()
        self.mymap.set_properties(self.params)
        self.mymap.set_geometry(self.params, self.unknowns)
        self.mymap.finput = open(mapMooring.FINPUTSTR, 'wb')
        
    def tearDown(self):
        self.mymap.finput.close()
        
    def set_geometry(self):
        geom = CylinderGeometry(2)
        tempUnknowns = {}
        geom.solve_nonlinear(self.params, tempUnknowns, None)
        for pairs in tempUnknowns.items():
            self.params[pairs[0]] = pairs[1]

    def read_input(self):
        self.mymap.finput.close()
        myfile = open(mapMooring.FINPUTSTR, 'rb')
        dat = myfile.read()
        myfile.close()
        return dat
        
    def testSetProperties(self):
        pass
    '''
    def testWriteLineDict(self):
        self.mymap.write_line_dictionary(self.params)
        self.mymap.finput.close()
        A = self.read_input()

    def testWriteNode(self):
        self.mymap.write_node_properties_header()
        self.mymap.write_node_properties(1, 'fix',0,0,0)
        self.mymap.write_node_properties(2, 'vessel',0,0,0)
        self.mymap.finput.close()
        A = self.read_input()

    def testWriteLine(self):
        self.mymap.write_line_properties(self.params)
        self.mymap.finput.close()
        A = self.read_input()

    def testWriteSolver(self):
        self.mymap.write_solver_options(self.params)
        self.mymap.finput.close()
        A = self.read_input()
    '''
    def testWriteInputAll(self):
        self.mymap.write_input_file(self.params)
        self.mymap.finput.close()
        actual = self.read_input().split()
        expect = truth.split()
        for k in xrange(len(actual)):
            if myisnumber(actual[k]):
                self.assertEqual( float(actual[k]), float(expect[k]) )
            else:
                self.assertEqual( actual[k], expect[k] )
            
    def testRunMap(self):
        self.mymap.runMAP(self.params, self.unknowns)

    def testCost(self):
        self.mymap.compute_cost(self.params, self.unknowns)
    
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMapMooring))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
