
from openmdao.api import ExplicitComponent
from mesh.py import Mesh
import numpy as np

class JacobianComp(ExplicitComponent):

    def setup(self):
        self.add_input('pN_value')
        self.add_input('EFT')
        self.add_input('Coords')
        self.add_output('Jacobian')
        self.declare_partials('Jacobian', '*')

    def compute(self, inputs, outputs):
        pN = inputs['pN_value']
        EFT = inputs['EFT']
        Coords = inputs['Coords']


        if problem_type == 'plane_stress':
            D = np.zeros((3,3))
            D[0][0] = C[0][0] - C[0][2] ** 2 / C[2][2]
            D[1][1] = C[1][1] - C[1][2] ** 2 / C[2][2]
            D[0][1] = C[0][1] - C[0][2] * C[1][2] / C[2][2]
            D[1][0] = C[0][1] - C[0][2] * C[1][2] / C[2][2]
            D[2][2] = C[3][3]

        # no sigma_3
        if problem_type == 'plane_strain':
            D = np.zeros((3,3))
            D[0][0] = C[0][0]
            D[1][1] = C[1][1]
            D[0][1] = C[0][1]
            D[1][0] = C[0][1]
            D[2][2] = C[3][3]

        if problem_type == 'truss':
            D = C[0][0]

        outputs['Jacobian'] = J

    def compute_partials(self, inputs, partials):
        C = inputs['C']

        partials['D', 'C'] = 0
