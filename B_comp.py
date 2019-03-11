

from openmdao.api import ExplicitComponent
from mesh import Mesh
from element import Element



import numpy as np

class BComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('ng', types=int)
        self.options.declare('NDIM', types=int)         # defined in the problem, constant for all elements.
        self.options.declare('max_nn', types=int)
        self.options.declare('NEL', types=int)
        self.options.declare('edim', types=int)
        self.options.declare('ndof', types=int)
        self.options.declare('problem_type', types=str)


    def setup(self):
        ng = self.options['ng']
        NDIM = self.options['NDIM']
        max_nn = self.options['max_nn']
        NEL = self.options['NEL']
        edim = self.options['edim']
        ndof = self.options['ndof']
        problem_type = self.options['problem_type']

        self.add_input('pN', shape=(NEL, ng, NDIM, max_nn))
        self.add_input('J', shape=(NEL, ng, NDIM, NDIM))
        self.add_output('B', shape=(NEL, ng, ndof, edim)) # or max_edof?

        self.declare_partials('B', '*', val=0)


    def compute(self, inputs, outputs):
        ng = self.options['ng']
        NDIM = self.options['NDIM']
        max_nn = self.options['max_nn']
        NEL = self.options['NEL']
        edim = self.options['edim']
        ndof = self.options['ndof']
        problem_type = self.options['problem_type']

        pN = inputs['pN']
        J = inputs['J']
        B = np.zeros((NEL, ng, ndof, edim))

        if problem_type == 'truss':
            for i in range(NEL):
                for j in range(ng):
                    B[i][j] = [-1/2, 0, 1/2, 0]

        outputs['B'] = B


    def compute_partials(self, inputs, partials):
        pass


if __name__ == '__main__':
    from openmdao.api import Problem

    prob = Problem()


    comp = BComp(ng=2, NDIM=1, max_nn=4, NEL=3, edim=4, ndof=1, problem_type='truss')

    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print(prob['B'])
    prob.check_partials(compact_print=True)


# mesh = Mesh()
# EFT = mesh.EFT()
# Node_Coords = mesh.Node_Coords

# ele = Element()
# pN = ele.shape_function_partial()
