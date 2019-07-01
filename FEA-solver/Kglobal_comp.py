from openmdao.api import ExplicitComponent
import numpy as np


from sparse_algebra import dense, sparse, SparseTensor, sparse_einsum, einsum_partial


class KglobalComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('max_edof', types=int)
        self.options.declare('NDOF', types=int)
        self.options.declare('NEL', types=int)
        self.options.declare('S', types = np.ndarray)           #shape = (NEL, max_edof, NDOF)
    

    def setup(self):
        max_edof = self.options['max_edof']
        NDOF = self.options['NDOF']
        NEL = self.options['NEL']

        self.add_input('Kel_local', shape=(NEL, max_edof, max_edof))
        self.add_input('t', shape = NEL)
        self.add_output('Kglobal', shape=(NDOF, NDOF))
        self.declare_partials('Kglobal', 't')
    
        S_ind = self.options['S']
        S_shape = np.array([NEL, max_edof, NDOF])
        S_val = np.ones(S_ind.shape[0])
        self.S = SparseTensor()
        self.S.initialize(S_shape, S_val, S_ind)
        
    # def compute(self, inputs, outputs):
    #     S = self.options['S']
    #     Kel_local = inputs['Kel_local']
    #
    #     Kglobal = np.einsum('ijk, imn, ijm  -> kn', S, S, Kel_local)
    #     outputs['Kglobal'] = Kglobal

    # def compute_partials(self, inputs, partials):
    #     # partials['Kglobal', 'Kel_local'] = inputs['Kel_local'] * 2
    #     pass


    def compute(self, inputs, outputs):
        t = inputs['t']
        Kel_local = inputs['Kel_local']

        Kel_local_sp = sparse(Kel_local)
        t_sp = sparse(t)

        print('itrn')

        self.Kglobal_sp1 = sparse_einsum([[0,1,2],[0,3,4],[0,1,3], [0,2,4]], self.S, self.S, Kel_local_sp)
        Kglobal_sp = sparse_einsum([[0, 2, 4], [0], [2, 4]], self.Kglobal_sp1, t_sp)
        # Kglobal_sp = sparse_einsum([[0,1,2],[0,3,4],[0,1,3],[2, 4]], S_sp, S_sp, Kel_local_sp)

        Kglobal = dense(Kglobal_sp)

        outputs['Kglobal'] = Kglobal

    def compute_partials(self, inputs, partials):
        t = inputs['t']
        t_sp = sparse(t)

        partial_sp = einsum_partial([[0,2,4], [0], [2, 4], [1]], self.Kglobal_sp1, t_sp)
        partial = dense(partial_sp)
        partials['Kglobal', 't'] = partial


# if __name__ == '__main__':
#     from openmdao.api import Problem, IndepVarComp
#
#     prob = Problem()
#     ivc = IndepVarComp()
#
#     comp = KglobalComp(max_edof=5, NDOF=15, NEL=3, S=np.arange(15))
#     ivc.add_output('Kel_local', val=np.random.rand(3,5,5))
#     ivc.add_output('t', val=np.random.rand(3))
#
#     prob.model.add_subsystem('ivc', ivc, promotes=['*'])
#     prob.model.add_subsystem('comp', comp, promotes=['*'])
#     prob.setup()
#     prob.run_model()
#     prob.check_partials(compact_print=True)
