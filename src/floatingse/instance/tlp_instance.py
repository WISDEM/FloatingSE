from semi_instance import SemiInstance
import numpy as np

        
class TLPInstance(SemiInstance):
    def __init__(self):
        super(TLPInstance, self).__init__()

        self.params['mooring_type']        = 'nylon'
        self.params['anchor_type']         = 'suctionpile'
        self.params['mooring_line_length'] = 0.95 * self.params['water_depth']
        self.params['anchor_radius']       = 10.0
        self.params['mooring_diameter']    = 0.1
        
        # Change scalars to vectors where needed
        self.check_vectors()
        
