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

        self.add_input('B', shape=(NEL, ng**2, 3, max_edof))            #3 only for a rectangular element
        self.add_input('D', shape=(3, 3))                               #(3,3) only for a rectangular element
        self.add_input('t', shape = (NEL))
        self.add_output('Kel_local', shape=(NEL, max_edof, max_edof))
        self.declare_partials('Kel_local', '*', method ='cs')
        
    def compute(self, inputs, outputs):
        B = inputs['B']
        D = inputs['D']
        t = inputs['t']
        R = RectangularElement()
        W = R.gaussian_weights()                         #only for a rectangular element
        Kel_local_pre1 = np.einsum('ijkl, ijno, kn -> ijlo', B, B, D)
        Kel_local_pre2 = np.einsum('ijlo, j ->ilo',Kel_local_pre1, W)
        Kel_local = np.einsum('ilo, i ->ilo',Kel_local_pre2, t)


        outputs['Kel_local'] = Kel_local

    # def compute_partials(self, inputs, partials):
        # partials['Kel_local', 'B'] = inputs['Kel_local'] * 2
        # partials['Kel_local', 'D'] = inputs['Kel_local'] * 2
        # pass