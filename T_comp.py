
from openmdao.api import ExplicitComponent
from mesh import Mesh

import numpy as np

class TComp(ExplicitComponent):
    def setup(self):
        self.add_input('mesh')
        self.add_output('T')
        self.declare_partials('C', '*')
        
    def compute(self, inputs, outputs):
        mesh = inputs['Mesh']
        ENT = mesh.ENT
        Node_Coords = mesh.Node_Coords
        NDIM = mesh.NDIM * 1
        max_edof = mesh.max_edof * 1

        if NDIM == 1:
            Tr = np.identity(max_edof)
        
        elif NDIM == 2:
            theta_lx_gx = 
            theta_lx_gy =
            theta_ly_gx =
            theta_ly_gy =

            Tr =

        elif NDIM == 2:
            theta_lx_gx = 
            theta_lx_gy =
            theta_lx_gz =
            theta_ly_gx =
            theta_ly_gy =
            theta_ly_gz =
            theta_lz_gx =
            theta_lz_gy =
            theta_lz_gz =

            Tr = 

        outputs['T'] = Tr

    def compute_partials(self, inputs, partials):
        E1 = inputs['E1']
        v23 = inputs['v23']

        partials['C', '*'] = 0