from openmdao.api import ExplicitComponent
import numpy as np
from mesh import Mesh


class Kel_localComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('W', types=np.ndarray)
        self.options.declare('max_edof', types=int)
        self.options.declare('n_D', types=int)
 

    def setup(self):
        W = self.options['W']
        max_edof = self.options['max_edof']
        n_D = self.options['n_D']
        (NEL, max_ng) = W.shape

        self.add_input('B', shape=(NEL, max_ng, n_D, max_edof))
        self.add_input('D', shape=(n_D, n_D))
        self.add_input('t', shape=(NEL))
        self.add_output('Kel_local', shape=(NEL, max_edof, max_edof))
        self.declare_partials('Kel_local', '*', method ='cs')
        
    def compute(self, inputs, outputs):
        W = self.options['W']

        B = inputs['B']
        D = inputs['D']
        t = inputs['t']
        Kel_local_pre1 = np.einsum('ijkl, kn, ijno -> ijlo', B, D, B)
        Kel_local_pre2 = np.einsum('ijlo, ij ->ilo', Kel_local_pre1, W)
        # print(Kel_local_pre2)
        Kel_local = np.einsum('ilo, i ->ilo', Kel_local_pre2, t)

        outputs['Kel_local'] = Kel_local

    # def compute_partials(self, inputs, partials):
        # partials['Kel_local', 'B'] = inputs['Kel_local'] * 2
        # partials['Kel_local', 'D'] = inputs['Kel_local'] * 2
        # pass


# if __name__ == '__main__':
#     from openmdao.api import Problem
#
#     prob = Problem()
#     mesh = Mesh()
#
#     node_coords1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1]])
#     ent1 = np.array([[1, 2, 3, 4], [2, 5, 6, 3]])
#     elem_type1 = 2  # rectangular
#     ndof1 = 2
#     n_D1 = 3
#
#     node_coords2 = np.array([[3, 0], [3, 1]])
#     ent2 = np.array([[3, 7, 6], [8, 7, 6]])
#     elem_type2 = 3  # triangular
#     ndof2 = 2
#     n_D2 = 3
#
#     # node_coords = np.array([[0, 0], [1, 0], [0, 1], [-1, 0]])
#     # ent = np.array([[1, 2], [2, 3], [1, 3], [1, 4], [3, 4]])
#     # elem_type = 1  # truss
#     # ndof = 2
#     # n_D = 1
#
#     mesh.set_nodes(node_coords1, ndof1)
#     mesh.set_nodes(node_coords2, ndof2)
#     mesh.add_elem_group(ent1, elem_type1)
#     mesh.add_elem_group(ent2, elem_type2)
#
#     comp = Kel_localComp(W = mesh.W, max_edof = mesh.max_edof, n_D = n_D1)
#
#     prob.model = comp
#     prob.setup()
#     prob.run_model()
#     prob.model.list_outputs()
#     Kel_local = prob['Kel_local']
#     print(Kel_local)
#     prob.check_partials(compact_print=True)
