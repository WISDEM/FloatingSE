from openmdao.api import Problem, ScipyOptimizer, pyOptSparseDriver
import numpy as np


NSECTIONS = 5

def nodal2sectional(x):
    """Averages nodal data to be length-1 vector of sectional data

    INPUTS:
    ----------
    x   : float vector, nodal data

    OUTPUTS:
    -------
    y   : float vector,  sectional data
    """
    return 0.5*(x[:-1] + x[1:])

def vecOption(x, in1s):
    myones = in1s if type(in1s) == type(np.array([])) else np.ones((in1s,))
    return (x*myones) if type(x)==type(0.0) or len(x) == 1 else x

class FloatingInstance(object):
    def __init__(self):
        self.prob = Problem()

    def get_assembly(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_design_variables(self):
        raise NotImplementedError("Subclasses should implement this!")
        
    def init_optimization(self, optimizer=None):
        # Establish the optimization driver
        validStr = 'Valid options are: [COBYLA, SLSQP, CONMIN, PSQP]'
        if optimizer is None:
            return
        elif optimizer.upper() in ['COBYLA','SLSQP']:
            self.prob.driver = ScipyOptimizer()
        elif optimizer.upper() in ['CONMIN', 'PSQP']:
            self.prob.driver = pyOptSparseDriver()
        elif optimizer.upper() in ['ALPSO', 'NSGA2', 'SLSQP']:
            raise ValueError('These optimizers run but jump to infeasible values. '+validStr)
        else:
            raise ValueError('Unknown or unworking optimizer. '+validStr)

        # Optimizer specific parameters
        self.prob.driver.options['optimizer'] = optimizer.upper()
        if optimizer.upper() == 'CONMIN':
            self.prob.driver.opt_settings['ITMAX'] = 1000
        elif optimizer.upper() in ['COBYLA','SLSQP']:
            self.prob.driver.options['tol'] = 1e-6
            self.prob.driver.options['maxiter'] = 100000

        # Add in design variables
        desvarList = self.get_design_variables()
        if optimizer.upper() in ['CONMIN','PSQP','ALPSO','NSGA2','SLSQP']:
            for ivar in desvarList:
                self.prob.driver.add_desvar(ivar[0], lower=ivar[1], upper=ivar[2])
        else:
            for ivar in desvarList:
                iscale=ivar[3]
                self.prob.driver.add_desvar(ivar[0], lower=iscale*ivar[1], upper=iscale*ivar[2], scaler=iscale)

    def add_constraints_objective(self):
        raise NotImplementedError("Subclasses should implement this!")

    def set_inputs(self):
        namesAssembly = self.prob.root._unknowns_dict.keys()
        for ivar in namesAssembly:
            if self.prob.root._unknowns_dict[ivar].has_key('_canset_') and self.prob.root._unknowns_dict[ivar]['_canset_']:
                selfvar = ivar.split('.')[0]
                selfval = getattr(self, selfvar, None)
                if selfval is None:
                    print 'Variable not found:', ivar, selfvar, self.prob[ivar]
                else:
                    self.prob[ivar] = selfval
        #raise NotImplementedError("Subclasses should implement this!")
    
    def init_problem(self, optimizer=None):
        self.prob = Problem()
        self.prob.root = self.get_assembly()

        self.init_optimization(optimizer=optimizer)

        self.add_constraints_objective()

        # Note this command must be done after the constraints, design variables, and objective have been set,
        # but before the initial conditions are specified (unless we use the default initial conditions )
        # After setting the intial conditions, running setup() again will revert them back to default values
        self.prob.setup()

        self.set_inputs()

        # Checks
        self.prob.check_setup()
        self.prob.pre_run_check()
        #self.prob.check_total_derivatives()

    def run(self, optimizer=None):
        self.init_problem(optimizer)
        self.prob.run()
        if not optimizer is None:
            print self.prob.driver.get_constraints()
            print self.prob.driver.get_desvars()
            print self.prob.driver.get_objectives()
        
    def evaluate(self, optimizer=None):
        self.init_problem(optimizer)
        self.prob.run_once()
        if not optimizer is None:
            print self.prob.driver.get_constraints()
            print self.prob.driver.get_desvars()
            print self.prob.driver.get_objectives()

            
