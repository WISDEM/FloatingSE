from floatingInstance import FloatingInstance, NSECTIONS, NPTS, vecOption
from floating import FloatingSE
from commonse import eps
import numpy as np
import time
        
class SemiInstance(FloatingInstance):
    def __init__(self):
        super(SemiInstance, self).__init__()

        # Parameters beyond those in superclass

        # Change scalars to vectors where needed
        self.check_vectors()

    def get_assembly(self): return FloatingSE(NSECTIONS)
    
    def get_design_variables(self):
        # Make a neat list of design variables, lower bound, upper bound, scalar
        desvarList = [('fairlead',0.0, 200.0, 1.0),
                      ('fairlead_offset_from_shell',0.0, 5.0, 1e2),
                      ('radius_to_auxiliary_column',0.0, 40.0, 1.0),
                      ('base_freeboard',0.0, 50.0, 1.0),
                      ('base_section_height',1e-1, 100.0, 1e1),
                      ('base_outer_diameter',1.1, 40.0, 10.0),
                      ('base_wall_thickness',5e-3, 1.0, 1e3),
                      ('auxiliary_freeboard',0.0, 50.0, 1.0),
                      ('auxiliary_section_height',1e-1, 100.0, 1e1),
                      ('auxiliary_outer_diameter',1.1, 40.0, 10.0),
                      ('auxiliary_wall_thickness',5e-3, 1.0, 1e3),
                      ('pontoon_outer_diameter', 1e-1, 3.0, 10.0),
                      ('pontoon_wall_thickness', 5e-3, 1e-1, 100.0),
                      ('base_pontoon_attach_lower',-1e2, 1e2, 1.0),
                      ('base_pontoon_attach_upper',-1e2, 1e2, 1.0),
                      ('scope_ratio', 1.0, 5.0, 1.0),
                      ('anchor_radius', 1.0, 1e3, 1e-2),
                      ('mooring_diameter', 0.05, 1.0, 1e1),
                      ('base_stiffener_web_height', 1e-2, 1.0, 1e2),
                      ('base_stiffener_web_thickness', 1e-3, 5e-1, 1e2),
                      ('base_stiffener_flange_width', 1e-2, 5.0, 1e2),
                      ('base_stiffener_flange_thickness', 1e-3, 5e-1, 1e2),
                      ('base_stiffener_spacing', 1e-1, 1e2, 1e1),
                      ('base_permanent_ballast_height', 1e-1, 50.0, 1.0),
                      ('auxiliary_stiffener_web_height', 1e-2, 1.0, 1e2),
                      ('auxiliary_stiffener_web_thickness', 1e-3, 5e-1, 1e2),
                      ('auxiliary_stiffener_flange_width', 1e-2, 5.0, 1e2),
                      ('auxiliary_stiffener_flange_thickness', 1e-3, 5e-1, 1e2),
                      ('auxiliary_stiffener_spacing', 1e-1, 1e2, 1e1),
                      ('auxiliary_permanent_ballast_height', 1e-1, 50.0, 1.0)]

        # TODO: Integer and Boolean design variables
        #prob.driver.add_desvar('number_of_auxiliary_columns', lower=1)
        #prob.driver.add_desvar('number_of_mooring_lines', lower=1)
        #prob.driver.add_desvar('mooring_type')
        #prob.driver.add_desvar('anchor_type')
        #prob.driver.add_desvar('bulkhead_nodes')
        #prob.driver.add_desvar('outer_cross_pontoons')
        #prob.driver.add_desvar('cross_attachment_pontoons')
        #prob.driver.add_desvar('lower_attachment_pontoons')
        #prob.driver.add_desvar('upper_attachment_pontoons')
        #prob.driver.add_desvar('lower_ring_pontoons')
        #prob.driver.add_desvar('upper_ring_pontoons')
        return desvarList

    def get_constraints(self):

        conlist = [
            # Ensure that draft is greater than 0 (spar length>0) and that less than water depth
            # Ensure that fairlead attaches to draft
            ['base.draft_depth_ratio', 0.0, 0.75, None],
            ['aux.draft_depth_ratio', 0.0, 0.75, None],
            ['aux.fairlead_draft_ratio', 0.0, 1.0, None],
            ['sg.base_auxiliary_spacing', 0.0, 1.0, None],
            
            # Ensure that the radius doesn't change dramatically over a section
            ['base.manufacturability', None, 0.0, None],
            ['base.weldability', None, 0.0, None],
            ['aux.manufacturability', None, 0.0, None],
            ['aux.weldability', None, 0.0, None],
            ['tow.manufacturability', None, 0.0, None],
            ['tow.weldability', None, 0.0, None],
            
            # Ensure that the spar top matches the tower base
            ['sg.transition_buffer', -1.0, 1.0, None],
            
            # Ensure max mooring line tension is less than X% of MBL: 60% for intact mooring, 80% for damanged
            ['mm.axial_unity', 0.0, 1.0, None],
            
            # Ensure there is sufficient mooring line length, MAP doesn't give an error about this
            ['mm.mooring_length_min', 1.0, None, None],
            ['mm.mooring_length_max', None, 1.0, None],
            
            # API Bulletin 2U constraints
            ['base.flange_spacing_ratio', None, 1.0, None],
            ['base.web_radius_ratio', None, 1.0, None],
            ['base.flange_compactness', 1.0, None, None],
            ['base.web_compactness', 1.0, None, None],
            ['base.axial_local_unity', None, 1.0, None],
            ['base.axial_general_unity', None, 1.0, None],
            ['base.external_local_unity', None, 1.0, None],
            ['base.external_general_unity', None, 1.0, None],
            
            ['aux.flange_spacing_ratio', None, 1.0, None],
            ['aux.web_radius_ratio', None, 1.0, None],
            ['aux.flange_compactness', 1.0, None, None],
            ['aux.web_compactness', 1.0, None, None],
            ['aux.axial_local_unity', None, 1.0, None],
            ['aux.axial_general_unity', None, 1.0, None],
            ['aux.external_local_unity', None, 1.0, None],
            ['aux.external_general_unity', None, 1.0, None],
            
            # Pontoon tube radii
            ['load.base_connection_ratio', 0.0, None, None],
            ['load.auxiliary_connection_ratio', 0.0, None, None],
            ['load.pontoon_base_attach_upper', 0.5, 1.0, None],
            ['load.pontoon_base_attach_lower', 0.0, 0.5, None],
            
            # Pontoon stress safety factor
            ['load.pontoon_stress', None, 1.0, None],
            ['load.tower_stress', None, 1.0, None],
            ['load.tower_shell_buckling', None, 1.0, None],
            ['load.tower_global_buckling', None, 1.0, None],
            
            # Achieving non-zero variable ballast height means the semi can be balanced with margin as conditions change
            ['stab.variable_ballast_height_ratio', 0.0, 1.0, None],
            ['stab.variable_ballast_mass', 0.0, None, None],
            
            # Metacentric height should be positive for static stability
            ['stab.metacentric_height', 0.1, None, None],
            
            # Center of buoyancy should be above CG (difference should be positive, None],
            #['stab.buoyancy_to_gravity', 0.1, None, None],
            
            # Surge restoring force should be greater than wave-wind forces (ratio < 1, None],
            ['stab.offset_force_ratio', None, 1.0, None],
            
            # Heel angle should be less than 6deg for ordinary operation, less than 10 for extreme conditions
            ['stab.heel_moment_ratio', None, 1.0, None]]
        return conlist


    def add_objective(self):
        # OBJECTIVE FUNCTION: Minimize total cost!
        self.prob.driver.add_objective('total_cost', scaler=1e-9)


        
    def visualize(self, fname=None):
        fig = self.init_figure()

        self.draw_ocean(fig)

        mooringMat = self.prob['mm.plot_matrix']
        self.draw_mooring(fig, mooringMat)

        pontoonMat = self.prob['load.plot_matrix']
        zcut = 1.0 + np.maximum( self.params['base_freeboard'], self.params['auxiliary_freeboard'] )
        self.draw_pontoons(fig, pontoonMat, 0.5*self.params['pontoon_outer_diameter'], zcut)

        self.draw_column(fig, [0.0, 0.0], self.params['base_freeboard'], self.params['base_section_height'],
                           0.5*self.params['base_outer_diameter'], self.params['base_stiffener_spacing'])

        R_semi  = self.params['radius_to_auxiliary_column']
        ncolumn = self.params['number_of_auxiliary_columns']
        angles = np.linspace(0, 2*np.pi, ncolumn+1)
        x = R_semi * np.cos( angles )
        y = R_semi * np.sin( angles )
        for k in xrange(ncolumn):
            self.draw_column(fig, [x[k], y[k]], self.params['auxiliary_freeboard'], self.params['auxiliary_section_height'],
                               0.5*self.params['auxiliary_outer_diameter'], self.params['auxiliary_stiffener_spacing'])

        self.draw_column(fig, [0.0, 0.0], self.params['base_freeboard']+self.params['hub_height'], self.params['tower_section_height'],
                         0.5*self.params['tower_outer_diameter'], None, (0.9,)*3)

            
        self.set_figure(fig, fname)



        


        
def example_semi():
    mysemi = SemiInstance()
    mysemi.evaluate('psqp')
    mysemi.visualize('semi-initial.jpg')
    #mysemi.run('slsqp')
    return mysemi

def optimize_semi(algo='slsqp', mysemi=None):
    if mysemi is None: mysemi = SemiInstance()
    mysemi.run(algo)
    mysemi.visualize('semi-'+algo+'.jpg')
    return mysemi

def psqp_optimal():
    mysemi = SemiInstance()
    mysemi.params['fairlead'] = 17.177323807530943
    mysemi.params['fairlead_offset_from_shell'] = 4.941895364231732
    mysemi.params['radius_to_auxiliary_column'] = 27.266399337057837
    mysemi.params['base_freeboard'] = 9.960408197770773
    mysemi.params['base_section_height'] = np.array( [7.282177926064044, 6.509388492862883, 6.129660155985191, 5.480370771116075, 5.382757297342475] )
    mysemi.params['base_outer_diameter'] = np.array( [6.545912191474823, 6.580137116128083, 6.702461164654701, 6.541518571779605, 6.06078472379382, 6.310563988634413] )
    mysemi.params['base_wall_thickness'] = np.array( [0.008028736514935459, 0.012449964898267678, 0.010273648666782207, 0.01970891677125363, 0.006037543856169967, 0.02749720537431047] )
    mysemi.params['auxiliary_freeboard'] = 5.910529536801677
    mysemi.params['auxiliary_section_height'] = np.array( [3.3331462056666994, 0.8261501300544689, 9.093696018350093, 9.190574119900322, 10.545158169430138] )
    mysemi.params['auxiliary_outer_diameter'] = np.array( [19.37923154315376, 22.277476530363643, 10.672128315236547, 8.832126822423682, 9.649278014987607, 11.772978075762646] )
    mysemi.params['auxiliary_wall_thickness'] = np.array( [0.005483291254680077, 0.008885683594236698, 0.005485807346346891, 0.00680537213783239, 0.005213663641222586, 0.006794356014331326] )
    mysemi.params['pontoon_outer_diameter'] = 1.521736449596963
    mysemi.params['pontoon_wall_thickness'] = 0.021050884370533998
    mysemi.params['base_pontoon_attach_lower'] = -19.242955772661425
    mysemi.params['base_pontoon_attach_upper'] = 9.874716098764614
    mysemi.params['scope_ratio'] = 4.681937038444982
    mysemi.params['anchor_radius'] = 837.2129303889922
    mysemi.params['mooring_diameter'] = 0.5615820760851148
    mysemi.params['base_stiffener_web_height'] = np.array( [0.054348344700015094, 0.02372637611477595, 0.4693870102231717, 0.012100794186299113, 0.01225689415048222] )
    mysemi.params['base_stiffener_web_thickness'] = np.array( [0.0023017819207248207, 0.010193261072047306, 0.022186429718501773, 0.0013588746637048608, 0.03154091789657742] )
    mysemi.params['base_stiffener_flange_width'] = np.array( [0.018415184383872213, 0.017676631632904553, 0.01620125554935165, 0.020360645704993713, 0.014594221166795688] )
    mysemi.params['base_stiffener_flange_thickness'] = np.array( [0.0230131965651039, 0.0045995740759375665, 0.1467056116656776, 0.01281598407802482, 0.02397320887428777] )
    mysemi.params['base_stiffener_spacing'] = np.array( [0.30351424821225764, 0.19082120752678203, 2.478544316637409, 0.3585901548075516, 0.5171271604589128] )
    mysemi.params['base_permanent_ballast_height'] = 0.47196748623781415
    mysemi.params['auxiliary_stiffener_web_height'] = np.array( [0.1266013437130618, 0.06632819483078813, 0.048573111230021636, 0.05506559933519377, 0.03240930541678604] )
    mysemi.params['auxiliary_stiffener_web_thickness'] = np.array( [0.007374041440190479, 0.006744067772957378, 0.004132484715422886, 0.002704640278298153, 0.007243997036273333] )
    mysemi.params['auxiliary_stiffener_flange_width'] = np.array( [0.02135406406675113, 0.01056348020854515, 0.018136997449099707, 0.011533322327725854, 0.013246111538899806] )
    mysemi.params['auxiliary_stiffener_flange_thickness'] = np.array( [0.018485765591477108, 0.018688740167702975, 0.002968255609866993, 0.005410431515293567, 0.022142071803304124] )
    mysemi.params['auxiliary_stiffener_spacing'] = np.array( [0.15824635810538867, 0.1812736633225982, 0.18984945638424194, 0.1655744185846234, 0.4566477508792847] )
    mysemi.params['auxiliary_permanent_ballast_height'] = 0.10058732188388624

    mysemi.evaluate('psqp')
    #mysemi.visualize('semi-slsqp.jpg')
    return mysemi
    
    
if __name__ == '__main__':
    #mysemi = optimize_semi('psqp')
    mysemi = psqp_optimal()
    optimize_semi('conmin', mysemi)
    #example_semi()
