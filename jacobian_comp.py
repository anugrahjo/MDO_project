from openmdao.api import ExplicitComponent
from mesh import Mesh
import numpy as np


class JacobianComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('pN', types=np.ndarray)
        self.options.declare('Elem_Coords', types=np.ndarray)

    def setup(self):
        pN = self.options['pN']
        (NEL, max_ng, NDIM, max_nn) = np.shape(pN)

        self.add_output('J', shape=(NEL, max_ng, NDIM, NDIM))
        self.add_output('pN_xy', shape=(NEL, max_ng, NDIM, NDIM))
        self.declare_partials('J', '*', method = 'cs')

    def compute(self, inputs, outputs):
        pN = self.options['pN']
        Elem_Coords = self.options['Elem_Coords']
        (NEL, max_ng, NDIM, max_nn) = np.shape(pN)

        J = np.einsum('ijkl, ilm -> ijkm', pN, Elem_Coords)

        outputs['J'] = J

    # def compute_partials(self, inputs, partials):
    #     pass

## testing
# if __name__ == '__main__':
#     from openmdao.api import Problem
#
#     prob = Problem()
#     mesh = Mesh()
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
#     comp = JacobianComp(pN=mesh.pN, Elem_Coords=mesh.Elem_Coords)
#     prob.model = comp
#     prob.setup()
#     prob.run_model()
#     prob.model.list_outputs()
#     print(prob['J'])
#     prob.check_partials(compact_print=True)
