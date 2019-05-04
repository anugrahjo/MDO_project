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
        # self.declare_partials('J', '*', method = 'cs')

    def compute(self, inputs, outputs):
        pN = self.options['pN']
        Elem_Coords = self.options['Elem_Coords']
        (NEL, max_ng, NDIM, max_nn) = np.shape(pN)

        J = np.einsum('ijkl, ilm -> ijkm', pN, Elem_Coords)

        outputs['J'] = J

    def compute_partials(self, inputs, partials):
        pass


if __name__ == '__main__':
    from openmdao.api import Problem

    prob = Problem()
    mesh = Mesh()

    node_coords = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
    ent = np.array([[1, 2, 5, 4], [2, 3, 6, 5]])
    elem_type = 2  # rectangular
    ndof = 2

    # node_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    # ent = np.array([[1, 2, 4], [3, 2, 4]])
    # elem_type = 3  # triangular
    # ndof = 2

    # node_coords = np.array([[0, 0], [1, 0], [0, 1], [-1, 0]])
    # ent = np.array([[1, 2], [2, 3], [1, 3], [1, 4], [3, 4]])
    # elem_type = 1  # truss
    # ndof = 2

    mesh.set_nodes(node_coords, ndof)
    mesh.add_elem_group(ent, elem_type)

    comp = JacobianComp(pN=mesh.pN, Elem_Coords=mesh.Elem_Coords)
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print(prob['J'])
    prob.check_partials(compact_print=True)
