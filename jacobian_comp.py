
from openmdao.api import ExplicitComponent
from mesh import Mesh
from element import Element


# mesh = Mesh()
# EFT = mesh.EFT()
# Node_Coords = mesh.Node_Coords

# ele = Element()
# pN = ele.shape_function_partial()


import numpy as np

class JacobianComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('ng', types=int)
        self.options.declare('NDIM', types=int)         # defined in the problem, constant for all elements.
        self.options.declare('max_nn', types=int)
        self.options.declare('NN', types=int)
        self.options.declare('NEL', types=int)

    def setup(self):
        ng = self.options['ng']
        NDIM = self.options['NDIM']
        max_nn = self.options['max_nn']
        NN = self.options['NN']
        NEL = self.options['NEL']

        self.add_input('pN', shape=(NEL, ng, NDIM, max_nn))
        self.add_input('EFT', shape=(NEL, max_nn))
        self.add_input('Node_Coords', shape=(NN, NDIM))

        self.add_output('J', shape=(NEL, ng, NDIM, NDIM))
        self.declare_partials('J', '*')

    def compute(self, inputs, outputs):
        ng = self.options['ng']
        NDIM = self.options['NDIM']
        max_nn = self.options['max_nn']
        NEL = self.options['NEL']

        pN = inputs['pN']
        EFT = inputs['EFT']
        Node_Coords = inputs['Node_Coords']
        J = np.zeros((NEL, ng, NDIM, NDIM))

        for i in range(NEL):
            eft_position = np.array(np.where(EFT[i]>-1))
            nn = eft_position.shape[1]
            eft = np.zeros(nn)
            for k in range(nn):
                eft[k] = EFT[i][eft_position[0][k]]
            pN_ele = pN[i][:][:][0:nn]
            coords_ele = np.zeros((nn, NDIM))
            for j in range(nn):
                position = int(eft[j])
                coords_ele[j] = Node_Coords[position]
            J[i] = np.einsum('ijk, km -> ijm', pN_ele, coords_ele)

        outputs['J'] = J

    def compute_partials(self, inputs, partials):
        pN = inputs['pN']
        EFT = inputs['EFT']
        Node_Coords = inputs['Node_Coords']

        partials['J', '*'] = 0


if __name__ == '__main__':
    from openmdao.api import Problem

    prob = Problem()

    comp = JacobianComp(ng=2, NDIM=2, max_nn=4, NN=10, NEL=3)
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print(prob['J'])
    prob.check_partials(compact_print=True)


