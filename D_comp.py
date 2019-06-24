
from openmdao.api import ExplicitComponent

import numpy as np

class DComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('E', types=float)
        self.options.declare('v', types=float)
        self.options.declare('problem_type', types=str)

    def setup(self):
        problem_type = self.options['problem_type']
        if problem_type == 'plane_stress' or 'plane_strain':
            n_D = 3
        if problem_type == 'truss':
            n_D = 1
        self.add_output('D', shape=(n_D, n_D))
        self.declare_partials('D', '*', method='cs')
        
    def compute(self, inputs, outputs):
        E = self.options['E']
        v = self.options['v']
        lam = E * v / ((1 + v) * (1 - 2 * v))
        mu = E / (2 * (1 + v))
        C = np.array([[lam + 2 * mu, lam, lam, 0, 0, 0],
                      [lam, lam + 2 * mu, lam, 0, 0, 0],
                      [lam, lam, lam + 2 * mu, 0, 0, 0],
                      [0, 0, 0, mu, 0, 0],
                      [0, 0, 0, 0, mu, 0],
                      [0, 0, 0, 0, 0, mu]])

        problem_type = self.options['problem_type']
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

        outputs['D'] = D


# if __name__ == '__main__':
#     from openmdao.api import Problem
#
#     prob = Problem()
#
#     comp = DComp(E=2., v=.3, problem_type='plane_stress')
#
#     prob.model = comp
#     prob.setup()
#     prob.run_model()
#     prob.model.list_outputs()
#     D = prob['D']
#     print(prob['D'])
#     prob.check_partials(compact_print=True)


