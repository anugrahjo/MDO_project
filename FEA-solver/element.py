
import numpy as np

# initialize the properties of element classes


class Element(object):

    def __init__(self, nn, ng):
        self.ndof = 0
        self.edof = 0
        self.nn = nn
        self.ng = ng
        self.xi = np.array([])
        self.eta = np.array([])
        self.weight = np.array([])
        self.N_ele = np.array([])     # shape function value stored at gauss points
        self.pN_ele = np.array([])     # partials of shape function value stored at gauss points

        self.setup(nn, ng)

    def setup(self):
        pass

    def gauss_points(self):
        pass

    def shape_function_value(self):
        pass

    def shape_function_partial(self):
        pass


class TrussElement(Element):

    def setup(self, nn, ng):
        self.ndof = 1
        self.edof = nn * self.ndof
        self.nn = nn
        self.ng = ng

        list_nn = [1, 2, 3]
        list_ng = [2, 3]
        if ng in list_ng:
            if nn in list_nn:
                self.gauss_points()
                self.shape_function_value()
                self.shape_function_partial()
            else:
                print('invalid number of nodes for the truss element')
        else:
            print('invalid number of Gauss points for the truss element')

    def gauss_points(self):
        ng = self.ng
        xi = np.zeros(ng)
        weight = np.zeros(ng)

        if ng == 1:
            xi = np.array([0])
            weight = np.array([2])

        if ng == 2:
            xi[0] = -.5773502691896257
            xi[1] = -1 * xi[0]
            weight[0] = 1.0
            weight[1] = 1.0

        if ng == 3:
            xi[0] = -0.7745966692414834
            xi[1] = 0.0
            xi[2] = -1 * xi[0]
            weight[0] = 0.5555555555555556
            weight[1] = 0.8888888888888888
            weight[2] = 0.5555555555555556

        self.xi = xi
        self.weight = weight

    def shape_function_value(self):
        nn = self.nn
        ng = self.ng
        xi = self.xi

        if nn == 2:
            N_value = np.transpose([1/2.*(1-xi), 1/2.*(1+xi)])
            N_value = np.reshape(N_value, (ng, 1, nn))

        if nn == 3:
            N_value = np.transpose([1/2.*xi*(xi-1), (1-xi)*(xi+1), 1/2.*xi*(xi+1)])
            N_value = np.reshape(N_value, (ng, 1, nn))

        self.N_ele = N_value

    def shape_function_partial(self):
        nn = self.nn
        ng = self.ng
        xi = self.xi

        if nn == 2:
            pN_value = np.tile([-1/2, 1/2], (ng, 1, 1))

        if nn == 3:
            pN_value = np.transpose([1/2.*(2*xi-1), -2.*xi, 1/2.*(2*xi+1)])
            pN_value = np.reshape(pN_value, (ng, 1, nn))

        self.pN_ele = pN_value


class RectangularElement(Element):

    def setup(self, nn, ng):
        self.ndof = 2
        self.edof = nn * self.ndof
        self.nn = nn
        self.ng = ng

        list_nn = [4, 6, 8, 9]
        list_ng = [1, 4, 6, 9]
        if ng in list_ng:
            if nn in list_nn:
                self.gauss_points()
                self.shape_function_value()
                self.shape_function_partial()
            else:
                print('invalid number of nodes for the rectangular element')
        else:
            print('invalid number of Gauss points for the rectangular element')

    def gauss_points(self):
        ng = self.ng
        xi = np.zeros(ng)
        eta = np.zeros(ng)
        weight = np.zeros(ng)

        if ng == 1:
            xi = np.array([0])
            eta = np.array([0])
            weight = np.array([4])

        if ng == 4:
            G = [-.5773502691896257, .5773502691896257]
            xi = np.array([G[0], G[1], G[1], G[0]])
            eta = np.array([G[0], G[0], G[1], G[1]])
            weight = np.array([1.0, 1.0, 1.0, 1.0])

        if ng == 6:
            G_1 = [-0.7745966692414834, 0, 0.7745966692414834]
            G_2 = [-.5773502691896257, .5773502691896257]
            xi = np.array([G_1[0], G_1[2], G_1[2], G_1[0], G_1[1], G_1[1]])
            eta = np.array([G_2[0], G_2[0], G_2[1], G_2[1], G_2[0], G_2[1]])
            weight = np.array([0.5555555555555556, 0.5555555555555556, 0.5555555555555556, 0.5555555555555556,
                               0.8888888888888888, 0.8888888888888888])

        if ng == 9:
            G = [-0.7745966692414834, 0, 0.7745966692414834]
            W = [0.5555555555555556, 0.8888888888888888, 0.5555555555555556]
            xi = np.array([G[0], G[2], G[2], G[0], G[1], G[2], G[1], G[0], G[1]])
            eta = np.array([G[0], G[0], G[2], G[2], G[0], G[1], G[2], G[1], G[1]])
            weight = np.array([W[0]*W[0], W[2]*W[0], W[2]*W[2], W[0]*W[2],
                               W[1]*W[0], W[2]*W[1], W[1]*W[2], W[0]*W[1], W[1]*W[1]])

        self.xi = xi
        self.eta = eta
        self.weight = weight

    def shape_function_value(self):
        nn = self.nn
        ng = self.ng
        xi = self.xi
        eta = self.eta

        if nn == 4:
            N_value = np.transpose([1/4.*(1-xi)*(1-eta), 1/4.*(1+xi)*(1-eta), 1/4.*(1+xi)*(1+eta), 1/4.*(1-xi)*(1+eta)])
            N_value = np.reshape(N_value, (ng, 1, nn))

        if nn == 6:
            N_value = np.transpose([1/4.*xi*(xi-1)*(1-eta), 1/4.*xi*(xi+1)*(1-eta), 1/4.*xi*(xi+1)*(1+eta),
                                    1/4.*xi*(xi-1)*(1+eta), 1/2.*(1-xi)*(xi+1)*(1-eta), 1/2.*(1-xi)*(xi+1)*(1+eta)])
            N_value = np.reshape(N_value, (ng, 1, nn))

        if nn == 8:
            N_value = np.transpose([1/4.*(1-xi)*(1-eta)*(-xi-eta-1), 1/4.*(1+xi)*(1-eta)*(xi-eta-1),
                       1/4.*(1+xi)*(1+eta)*(xi+eta-1), 1/4.*(1-xi)*(1+eta)*(-xi+eta-1),
                       1/2.*(1-eta)*(1+xi)*(1-xi), 1/2.*(1+xi)*(1-eta)*(1+eta),
                       1/2.*(1+eta)*(1-xi)*(1+xi), 1/2.*(1-xi)*(1-eta)*(1+eta)])
            N_value = np.reshape(N_value, (ng, 1, nn))

        if nn == 9:
            N_value = np.transpose([1/4.*xi*(xi-1)*eta*(eta-1), 1/4.*xi*(xi+1)*eta*(eta-1),
                       1/4.*xi*(xi+1)*eta*(eta+1), 1/4.*xi*(xi-1)*eta*(eta+1),
                       1/2.*(1-xi)*(xi+1)*eta*(eta-1), 1/2.*xi*(xi+1)*(1-eta)*(eta+1),
                       1/2.*(1-xi)*(xi+1)*eta*(eta+1), 1/2.*xi*(xi-1)*(1-eta)*(eta+1),
                       (xi+1)*(1-xi)*(eta+1)*(1-eta)])
            N_value = np.reshape(N_value, (ng, 1, nn))

        self.N_ele = N_value

    def shape_function_partial(self):
        nn = self.nn
        ng = self.ng
        xi = self.xi
        eta = self.eta

        pN_value = np.zeros((ng, 2, nn))

        if nn == 4:
            for i in range(ng):
                pN_value[i] = [[-1/4*(1-eta[i]), 1/4*(1-eta[i]), 1/4*(1+eta[i]), -1/4*(1+eta[i])],
                               [-1/4*(1-xi[i]), -1/4.*(1+xi[i]), 1/4.*(1+xi[i]), 1/4.*(1-xi[i])]]

        if nn == 6:
            for i in range(ng):
                N_xi = np.array([1/2*xi[i]*(xi[i]-1), (1-xi[i])*(1+xi[i]), 1/2*xi[i]*(xi[i]+1)])
                N_eta = np.array([1/2*(1-eta[i]), 1/2*(eta[i]+1)])
                pN_xi = np.array([1/4*(2*xi[i]-1), -1*xi[i], 1/4*(2*xi[i]+1)])
                pN_eta = np.array([-1/2, 1/2])

                pN_value[i] = [[pN_xi[0] * N_eta[0], pN_xi[2] * N_eta[0], pN_xi[2] * N_eta[1], pN_xi[0] * N_eta[1],
                                pN_xi[1] * N_eta[0], pN_xi[1] * N_eta[1]],
                               [N_xi[0] * pN_eta[0], N_xi[2] * pN_eta[0], N_xi[2] * pN_eta[1], N_xi[0] * pN_eta[1],
                                N_xi[1] * pN_eta[0], N_xi[1] * pN_eta[1]]]

        if nn == 8:
            for i in range(ng):
                m = xi[i]
                n = eta[i]
                pN_value[i] = [[1/4*(1-n)*(2*m+n), 1/4*(1-n)*(2*m-n), 1/4*(1+n)*(2*m+n), 1/4*(1+n)*(2*m-n),
                                -m*(1-n), 1/2*(1-n)*(1+n), -m*(1+n), -1/2*(1-n)*(1+n)],
                               [1/4*(1-m)*(2*n+m), 1/4*(1+m)*(2*n-m), 1/4*(1+m)*(2*n+m), 1/4*(1-m)*(2*n-m),
                                -1/2*(1-m)*(1+m), -n*(1+m), 1/2*(1-m)*(1+m), -n*(1-m)]]

        if nn == 9:
            for i in range(ng):
                N_xi = np.array([1/2*xi[i]*(xi[i]-1), (1-xi[i])*(1+xi[i]), 1/2*xi[i]*(xi[i]+1)])
                N_eta = np.array([1/2*eta[i]*(eta[i]-1), (1-eta[i])*(1+eta[i]), 1/2*eta[i]*(eta[i]+1)])
                pN_xi = np.array([1/4*(2*xi[i]-1), -1*xi[i], 1/4*(2*xi[i]+1)])
                pN_eta = np.array([1/4*(2*eta[i]-1), -1*eta[i], 1/4*(2*eta[i]+1)])

                pN_value[i] = [[pN_xi[0] * N_eta[0], pN_xi[2] * N_eta[0], pN_xi[2] * N_eta[2], pN_xi[0] * N_eta[2],
                                pN_xi[1] * N_eta[0], pN_xi[2] * N_eta[1], pN_xi[1] * N_eta[2], pN_xi[0] * N_eta[1],
                                pN_xi[1] * N_eta[1]],
                               [N_xi[0] * pN_eta[0], N_xi[2] * pN_eta[0], N_xi[2] * pN_eta[2], N_xi[0] * pN_eta[2],
                                N_xi[1] * pN_eta[0], N_xi[2] * pN_eta[1], N_xi[1] * pN_eta[2], N_xi[0] * pN_eta[1],
                                N_xi[1] * pN_eta[1]]]

        self.pN_ele = pN_value


class TriangularElement(Element):

    def setup(self, nn, ng):
        self.ndof = 2
        self.edof = nn * self.ndof
        self.nn = nn
        self.ng = ng
        list_ng = [1, 3, 4]
        list_nn = [3, 6]

        if ng in list_ng:
            if nn in list_nn:
                self.gauss_points()
                self.shape_function_value()
                self.shape_function_partial()
            else:
                print('invalid number of nodes for the triangular element')
        else:
            print('invalid number of Gauss points for the triangular element')

    def gauss_points(self):
        ng = self.ng
        xi = np.zeros(ng)
        eta = np.zeros(ng)
        weight = np.zeros(ng)

        if ng == 1:
            xi = np.array([1/3])
            eta = np.array([1/3])
            weight = np.array([1/2])

        if ng == 3:
            xi = np.array([0.5, 0, 0.5])
            eta = np.array([0, 0.5, 0.5])
            weight = np.array([1/6, 1/6, 1/6])

        if ng == 4:
            xi = np.array([1/3, 0.6, 0.2, 0.2])
            eta = np.array([1/3, 0.2, 0.6, 0.2])
            weight = np.array([-27/96, 25/96, 25/96, 25/96])

        self.xi = xi
        self.eta = eta
        self.weight = weight

    def shape_function_value(self):
        nn = self.nn
        ng = self.ng
        xi = self.xi
        eta = self.eta

        if nn == 3:
            N_value = np.transpose([1 - xi - eta, xi, eta])
            N_value = np.reshape(N_value, (ng, 1, nn))

        if nn == 6:

            N_value = np.transpose([2.*(1-xi-eta)*(1/2-xi-eta), 2.*xi*(xi-1/2), 2.*eta*(eta-1/2),
                                    4.*(1-xi-eta)*xi, 4.*xi*eta, 4.*(1-xi-eta)*eta])
            N_value = np.reshape(N_value, (ng, 1, nn))

        self.N_ele = N_value

    def shape_function_partial(self):
        nn = self.nn
        ng = self.ng
        xi = self.xi
        eta = self.eta
        pN_value = np.zeros((ng, 2, nn))

        if nn == 3:
            pN_value = np.tile([[-1, 1, 0], [-1, 0, 1]], (ng, 1, 1))

        if nn == 6:
            for i in range(ng):
                pN_value[i] = [[4*xi[i]+4*eta[i]-3, 4*xi[i]-1, 0, -8*xi[i]-4*eta[i]+4, 4*eta[i], -4*eta[i]],
                               [4*xi[i]+4*eta[i]-3, 0, 4*xi[i]-1, -4*xi[i], 4*xi[i], -8*eta[i]-4*xi[i]+4]]

        self.pN_ele = pN_value


class BeamElement(Element):

    def setup(self, nn, ng):
        self.ndof = 2
        self.edof = nn * self.ndof
        self.nn = nn
        self.ng = ng

        list_nn = [2, 3]
        list_ng = [1, 2]
        if ng in list_ng:
            if nn in list_nn:
                self.gauss_points()
                self.shape_function_value()
                self.shape_function_partial()
            else:
                print('invalid number of nodes for the beam element')
        else:
            print('invalid number of Gauss points for the beam element')

    def gauss_points(self):
        ng = self.ng
        xi = np.zeros(ng)
        weight = np.zeros(ng)

        if ng == 1:
            xi = np.array([0])
            weight = np.array([2])

        if ng == 2:
            xi[0] = -.5773502691896257
            xi[1] = -1 * xi[0]
            weight[0] = 1.0
            weight[1] = 1.0

        if ng == 3:
            xi[0] = -0.7745966692414834
            xi[1] = 0.0
            xi[2] = -1 * xi[0]
            weight[0] = 0.5555555555555556
            weight[1] = 0.8888888888888888
            weight[2] = 0.5555555555555556

        self.xi = xi
        self.weight = weight

    def shape_function_value(self):
        nn = self.nn
        ng = self.ng
        xi = self.xi

        if nn == 2:
            N_value = np.transpose([1/4.*(xi**3-3*xi+2), 1/8.*(xi**3-xi**2-xi+1), 1/4.*(-xi**3+3*xi+2), 1/8.*(xi**3+xi**2-xi-1)])
            N_value = np.reshape(N_value, (ng, 1, nn*2))

#        if nn == 3:
#            N_value = np.transpose([1/2.*xi*(xi-1), (1-xi)*(xi+1), 1/2.*xi*(xi+1)])
#            N_value = np.reshape(N_value, (ng, 1, nn))

        self.N_ele = N_value

    def shape_function_partial(self):
        nn = self.nn
        ng = self.ng
        xi = self.xi

        if nn == 2:
            pN_value = np.tile([-1/2, 1/2], (ng, 1, 1))

#        if nn == 3:
#            pN_value = np.transpose([1/2.*(2*xi-1), -2.*xi, 1/2.*(2*xi+1)])
#            pN_value = np.reshape(pN_value, (ng, 1, nn))

        self.pN_ele = pN_value


# truss = TrussElement(nn=2, ng=2)
# print(truss.N_ele)
# print(truss.pN_ele)
#
# rec = RectangularElement(nn=4, ng=4)
# print(rec.weight)
# print(rec.N_ele)
# print(rec.pN_ele)
#
# tri = TriangularElement(nn=3, ng=4)
