from openmdao.api import ExplicitComponent
from rectangular_plate import RectangularElement
import numpy as np

class Kel_localComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('ng', types=int)
        self.options.declare('max_edof', types=int)
        self.options.declare('NEL', types=int)
 

    def setup(self):
        ng = self.options['ng']
        max_edof = self.options['max_edof']
        NEL = self.options['NEL']

        self.add_input('B', shape=(NEL, ng, 3, max_edof))
        self.add_input('D', shape=(3, 3))
        self.add_output('Kel_local', shape=(NEL, max_edof, max_edof))
        self.declare_partials('Kel_local', '*')
        
    def compute(self, inputs, outputs):
        B = inputs['B']
        D = inputs['D']
        W = RectangularElement.gaussian_weights                         #only for a rectangular element

        Kel_local_pre = np.einsum('ijkl, ijno, kn -> ijlo', B, B, D)
        Kel_local = np.einsum('ijlo, j ->ilo',Kel_local_pre, W)

        outputs['Kel_local'] = Kel_local

    def compute_partials(self, inputs, partials):
        pass