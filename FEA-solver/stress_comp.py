import numpy as np

from openmdao.api import ExplicitComponent


class StressComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('max_nn', types=int)
        self.options.declare('NEL', types=int)
        self.options.declare('max_edof', types=int)         
        self.options.declare('problem_type', types=str)

    def setup(self):
        NDOF = self.options['NDOF']
        ng = self.options['ng']
        NEL = self.options['NEL']
        max_edof = self.options['max_edof']
        problem_type = self.options['problem_type']
        
        if problem_type == 'plane_stress' or 'plane_strain':
            n_D = 3
        if problem_type == 'truss':
            n_D = 1

        self.add_input('D', shape = (n_D, n_D))
        self.add_input('strain', shape=(NEL, n_D))
        self.add_output('stress', shape=(NEL, n_D))

        self.declare_partials('strain', '*', method ='cs')

    def compute(self, inputs, outputs):
        D = inputs['D']
        strain = inputs['strain']

        stress = np.einsum('ij, kj -> ki', D, strain)

        outputs['stress'] = stress