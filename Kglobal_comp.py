from openmdao.api import ExplicitComponent
import numpy as np

class KglobalComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('max_edof', types=int)
        self.options.declare('NDOF', types=int)
        self.options.declare('NEL', types=int)
        self.options.declare('S', types = np.ndarray)           #shape = (NEL, max_edof, NDOF)

    def setup(self):
        max_edof = self.options['max_edof']
        NDOF = self.options['NDOF']
        NEL = self.options['NEL']

        self.add_input('Kel_local', shape=(NEL, max_edof, max_edof))
        self.add_output('Kglobal', shape=( NDOF, NDOF))
        self.declare_partials( 'Kglobal', 'Kel_local', method = 'cs')
        
    def compute(self, inputs, outputs):
        S = self.options['S']
        Kel_local = inputs['Kel_local']

        Kglobal = np.einsum('ijk, imn, ijm  -> kn', S, S, Kel_local)
        outputs['Kglobal'] = Kglobal

    # def compute_partials(self, inputs, partials):
    #     # partials['Kglobal', 'Kel_local'] = inputs['Kel_local'] * 2
    #     pass