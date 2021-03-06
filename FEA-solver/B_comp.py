
from openmdao.api import ExplicitComponent
from mesh import Mesh

import numpy as np


class BComp(ExplicitComponent):

    def initialize(self):     # defined in the problem, constant for all elements.
        self.options.declare('problem_type', types=str)
        self.options.declare('pN', types=np.ndarray)
        self.options.declare('max_edof', types=int)

    def setup(self):
        problem_type = self.options['problem_type']
        pN = self.options['pN']
        max_edof = self.options['max_edof']
        (NEL, max_ng, NDIM, max_nn) = np.shape(pN)

        if problem_type == 'plane_stress' or 'plane_strain':
            n_D = 3
        if problem_type == 'truss':
            n_D = 1
        # self.add_input('J', shape=(NEL, max_ng, NDIM, NDIM))
        self.add_output('B', shape=(NEL, max_ng, n_D, max_edof))
        self.declare_partials('B', '*', method='cs')

    def compute(self, inputs, outputs):

        problem_type = self.options['problem_type']
        pN = self.options['pN']
        max_edof = self.options['max_edof']
        (NEL, max_ng, NDIM, max_nn) = pN.shape
        # same as the settings in 'mesh'

        # J = inputs['J']

        if problem_type == 'plane_stress' or 'plane_strain':
            n_D = 3
            (NEL, ng, NDIM, nn) = pN.shape

            B = np.zeros((NEL, ng, n_D, max_edof))
            R = np.zeros((NEL, ng, n_D, max_edof, NDIM, max_nn))

            L = np.zeros((n_D, max_edof, NDIM, max_nn))
            for i in range(max_nn):
                L[0][2*i][0][i] = 1
                L[1][2*i+1][1][i] = 1
                L[2][2*i][1][i] = 1
                L[2][2*i+1][0][i] = 1

            R = np.tile(L, (NEL, ng, 1, 1, 1, 1))
            B = np.einsum('ijklmn, ijmn -> ijkl', R, pN)


        if problem_type == 'truss':
            n_D = 1
            B_ele = np.array([-1/2, 0, 1/2, 0])
            B = np.tile(B_ele, (NEL, ng, n_D, 1))

        outputs['B'] = B


    # def compute_partials(self, inputs, partials):
    #     pass

#
# if __name__ == '__main__':
#     from openmdao.api import Problem
#
#     prob = Problem()
#
#     mesh = Mesh()
#
#     node_coords1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1]])
#     ent1 = np.array([[1, 2, 3, 4], [2, 5, 6, 3]])
#     elem_type1 = 2  # rectangular
#     ndof1 = 2
#
#     node_coords2 = np.array([[3, 0], [3, 1]])
#     ent2 = np.array([[3, 7, 6], [8, 7, 6]])
#     elem_type2 = 3  # triangular
#     ndof2 = 2
#
#     # node_coords = np.array([[0, 0], [1, 0], [0, 1], [-1, 0]])
#     # ent = np.array([[1, 2], [2, 3], [1, 3], [1, 4], [3, 4]])
#     # elem_type = 1  # truss
#     # ndof = 2
#
#     mesh.set_nodes(node_coords1, ndof1)
#     mesh.add_elem_group(ent1, elem_type1)
#     mesh.set_nodes(node_coords2, ndof2)
#     mesh.add_elem_group(ent2, elem_type2)
#
#     comp = BComp(problem_type='plane_strain', pN=mesh.pN, max_edof=mesh.max_edof)
#
#     prob.model = comp
#     prob.setup()
#     prob.run_model()
#     prob.model.list_outputs()
#     B = prob['B']
#     print(prob['B'])
#     prob.check_partials(compact_print=True)
