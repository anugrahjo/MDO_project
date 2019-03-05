
from element import Element
from shape_functions import Shape_Function, Diff_Shape_Function
from gauss_points import Gauss_Points

import numpy as np

class Truss_Element(Element):
    
    def setup(self):
        self.nn = 2
        self.ndof = 2
        self.edof = 4
        self.element_type = 1
    
    def shape_function_value(self):
        G, W = Gauss_Points(2)
        N_value = np.zeros((2, 1, 2))
        for i in range(2):
            N_value[i] = Shape_Function(2, G[i])

        return N_value

    def shape_function_partial(self):
        G, W = Gauss_Points(2)
        pN_value = np.zeros((2, 1, 2))
        for i in range(2):
            pN_value[i] = Diff_Shape_Function(2, G[i])
        print(pN_value)
        return pN_value


        


