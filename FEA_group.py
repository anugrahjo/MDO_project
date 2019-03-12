import numpy as np

from mesh import Mesh
from openmdao.api import Group, ExplicitComponent
from mesh import Mesh
from jacobian_comp import JacobianComp
from B_comp import BComp
from D_comp import DComp
from Kel_local_comp import Kel_localComp
from Kglobal_comp import KglobalComp


class FEAGroup(Group):
    def initialize(self):
        self.options.declare('mesh')
        self.options.declare('C')
        self.options.declare('problem_type')
        self.options.declare('ng')

    def setup(self):
        mesh = self.options['mesh']
        C = self.options['C']
        problem_type = self.options['problem_type']
        ng = self.options['ng']

        pN = mesh.pN
        ENT = mesh.ENT
        Node_Coords = mesh.Node_Coords
        NDOF = mesh.NDOF
        NEL = mesh.NEL
        NDIM = mesh.NDIM
        max_nn = mesh.max_nn
        max_edof = mesh.max_edof
        NN = mesh.NN
        S = mesh.S

        comp = ExplicitComponent()
        comp.add_output('d', shape = (NDOF))
        # self.add_subsystem('i_comp', comp, promotes=['*'])

        comp = DComp(C = C, problem_type = problem_type)
        self.add_subsystem('D_comp', comp, promotes=['*'])

        comp = JacobianComp(ng= ng, NDIM =NDIM, NEL = NEL, pN =pN, ENT = ENT, Node_Coords = Node_Coords)
        self.add_subsystem('J_comp', comp, promotes=['*'])

        comp = BComp(ng= ng, NDIM =NDIM, max_nn = max_nn, NEL = NEL, max_edof = max_edof, pN =pN, problem_type = problem_type, ENT = ENT)
        self.add_subsystem('B_comp', comp, promotes=['*'])

        comp = Kel_localComp(ng = ng, max_edof = max_edof, NEL = NEL)
        self.add_subsystem('Kl_comp', comp, promotes=['*'])

        comp = KglobalComp(S = S, max_edof = max_edof, NEl =NEL, NDOF = NDOF)
        self.add_subsystem('Kg_comp', comp, promotes=['*'])
        


if __name__ == '__main__':
    from openmdao.api import Problem, ScipyOptimizeDriver

    prob = Problem()
    prob.model = FEAGroup()

    