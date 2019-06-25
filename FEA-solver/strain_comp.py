import numpy as np

from openmdao.api import ExplicitComponent
from plate_element import RectangularElement
# from rectangular_plate import RectangularElement


class StrainComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('ng', types=int)
        self.options.declare('NEL', types=int)             # defined in the problem, constant for all elements.
        self.options.declare('NDOF', types=int)
        self.options.declare('max_edof', types=int)         
        self.options.declare('problem_type', types=str)
        self.options.declare('S', types=np.ndarray)       #shape = (NEL, max_edof, NDOF)

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

        self.add_input('d', shape = NDOF)
        self.add_input('B', shape=(NEL, ng**2, n_D, max_edof))
        self.add_output('strain', shape=(NEL, n_D))

        self.declare_partials('strain', '*', method ='cs')

    def compute(self, inputs, outputs):
        S = self.options['S']
        B = inputs['B']
        d = inputs['d']
        R = RectangularElement()
        W = R.weights                            #weights defined only for a rectangular element

        strain_pre = np.einsum('ijkl, iln, n -> ijk', B, S, d)
        strain = np.einsum('ijk, j -> ik', strain_pre, W)

        outputs['strain'] = strain
