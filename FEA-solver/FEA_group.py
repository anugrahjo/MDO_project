import numpy as np

from mesh import Mesh
from openmdao.api import Group, ExplicitComponent, ImplicitComponent, IndepVarComp, LinearSystemComp
from mesh import Mesh
from jacobian_comp import JacobianComp
from B_comp import BComp
from D_comp import DComp
from Kel_local_comp import Kel_localComp
from Kglobal_comp import KglobalComp
from KKT_comp import KKTComp
from displacements_comp import DisplacementsComp
from compliance_comp import ComplianceComp
from volume_comp import VolumeComp
from sparse_algebra import SparseTensor, sparse, compute_indices


class FEAGroup(Group):
    def initialize(self):
        self.options.declare('mesh', types = Mesh)
        self.options.declare('E', types = float)
        self.options.declare('v', types = float)
        self.options.declare('problem_type', types = str )
        self.options.declare('ng', types = int)
        self.options.declare('A', types = np.ndarray)
        self.options.declare('f', types = np.ndarray)
        self.options.declare('constraints', types = np.ndarray)
        self.options.declare('be', types = float)
        self.options.declare('le', types = float)

    def setup(self):
        mesh = self.options['mesh']
        E = self.options['E']
        v = self.options['v']
        problem_type = self.options['problem_type']
        ng = self.options['ng']
        be = self.options['be']
        le = self.options['le']
        A = self.options['A']
        f = self.options['f']
        constraints = self.options['constraints']

        pN = mesh.pN
        W = mesh.W
        ENT = mesh.ENT
        Elem_Coords = mesh.Elem_Coords
        NDOF = mesh.NDOF
        NEL = mesh.NEL
        NDIM = mesh.NDIM
        max_nn = mesh.max_nn
        max_ng = mesh.max_ng
        max_edof = mesh.max_edof
        NN = mesh.NN
        S = mesh.S.ind

        # comp = ExplicitComponent()
        # comp.add_output('d', shape = (NDOF))
        # self.add_subsystem('d_comp', comp, promotes=['*'])
        if problem_type == 'plane_stress' or 'plane_strain':
            n_D = 3
        if problem_type == 'truss':
            n_D = 1

        comp = IndepVarComp()
        comp.add_output('t', shape = (NEL))
        self.add_subsystem('t_comp', comp, promotes=['*'])

        comp = DComp(E = E, v = v, problem_type = problem_type)
        self.add_subsystem('D_comp', comp, promotes=['*'])

        comp = JacobianComp(pN = pN, Elem_Coords = Elem_Coords)
        self.add_subsystem('J_comp', comp, promotes=['*'])

        comp = BComp(pN = pN, problem_type = problem_type, max_edof = max_edof)
        self.add_subsystem('B_comp', comp, promotes=['*'])

        comp = Kel_localComp(W = W, max_edof = max_edof, n_D = n_D)
        self.add_subsystem('Kl_comp', comp, promotes=['*'])

        comp = KglobalComp(S = S, max_edof = max_edof, NEL = NEL, NDOF = NDOF)
        self.add_subsystem('Kg_comp', comp, promotes=['*'])

        comp = KKTComp(NDOF = NDOF, A = A , f = f, constraints = constraints)
        self.add_subsystem('KKT_comp', comp, promotes=['*'])

        self.add_subsystem('Solve_comp', LinearSystemComp(size = (NDOF+len(constraints))))

        comp = DisplacementsComp(NDOF = NDOF, constraints = constraints)
        self.add_subsystem('Displacements_comp', comp, promotes=['*'])
        
        comp = ComplianceComp(NDOF = NDOF, f = f)
        self.add_subsystem('Compliance_comp', comp, promotes=['*'])

        comp = VolumeComp(NEL = NEL, be = be, le = le )
        self.add_subsystem('Volume_comp', comp, promotes=['*'])


# if __name__ == '__main__':
#     from openmdao.api import Problem, ScipyOptimizeDriver
#
#     prob = Problem()
#     prob.model = FEAGroup()
#
