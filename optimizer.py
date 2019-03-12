import numpy as np
from mesh import Mesh

from openmdao.api import Problem, ScipyOptimizeDriver

# from FEA_group import FEAGroup

C = 1000 / (1 - 0.3 ** 2) * np.array([[1, 0.3, 0],[0.3, 1, 0],[0, 0, 0.7/2]])

prob_type = 'plane_strain'

ng = 2

mesh = Mesh()
node_coords1 = np.array([[0,0],[0.5,0],[1,0],[0,0.5],[0.5,0.5],[1,0.5],[0,1],[0.5,1],[1,1]])
ndof1 = 2
mesh.set_nodes(node_coords1, ndof1)
ent1 = np.array([[1,2,5,4],[2,3,6,5],[4,5,8,7],[5,6,9,8]])
mesh.add_elem_group(ent1, 2)
mesh.add_elem_group_partials



# prob = Problem(model = FEAGroup(C = C, mesh = mesh))

prob = Problem()
# prob.model = FEAGroup(C=C, mesh=mesh, problem_type = prob_type, ng = ng )
# prob.model.add_design_var('d', upper=, lower=, value=)
# prob.model.add_constraint('e', equals=0, upper=, lower=)
prob.model.add_objective('f', scaler = -1)

prob.driver = ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

# prob.driver.options['tol'] = 1e-9
# prob.driver.options['obj'] = True

# prob.setup()
# # prob.run_driver()
# prob.run_model()
# prob.model.list_outputs()
# prob['']