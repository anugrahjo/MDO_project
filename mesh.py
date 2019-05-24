
import numpy as np

from element import TrussElement, RectangularElement, TriangularElement

class Mesh():

    def __init__(self):
        self.NEL = 0                                #total num of elements in the mesh
        self.num_elem_types = 0                     #keeps track of num of elem types in the mesh
        self.elem_types_in_mesh = np.array([])      #keeps track of all elem types in the mesh
        self.NN = 0                                 #total num of nodes in the mesh
        self.ndof = 0                               #num of dof for each node in the mesh(Problem Dependent)
        self.max_nn = 0                             #num of nodes per element for element type with max num of nodes
        self.max_edof = 0                           #num of dof of element type with max num of dof
        self.max_ng = 0
        self.NDOF = 0                               #Global dof for the entire mesh
        self.NDIM = 0                               #num of dimensions in which the mesh is present(Problem Dependent)
        self.ENT = np.array([])                     #max. num of nodes per element is set as max_nn
        self.EFT = np.array([])                     #max. num of dof per element is set as max_edof
        self.S = np.array([])                       #max. num of dof per element is set as max_edof
        self.Node_Coords = np.array([])             #Coords of each node in the mesh
        self.Elem_Coords = np.array([])
        self.Elem_Group_Dict = np.array([])         #element type of each element in the mesh
        self.pN = np.array([])                      #partials of shape functions for each element group
        self.W = np.array([])                       #weights for gauss points

    def set_nodes(self, node_coords, ndof):
        # -----------------------------------------------------------------------------------------------
        # Set NDIM & updates for NN and NDOF
        num_new_nodes = node_coords.shape[0]
        self.NN += num_new_nodes
        if self.NDIM == 0:
            self.NDIM = node_coords.shape[1]
            self.Node_Coords = node_coords * 1
            self.ndof = ndof
            self.num_elem_type = 1
        else:
            self.Node_Coords = np.append(self.Node_Coords, node_coords, axis = 0)
        new_dof = num_new_nodes * ndof
        self.NDOF += new_dof
        

    # DOF_index and node_index starting from 1
    def add_elem_group(self, ent ,elem_type):

        # -----------------------------------------------------------------------------------------------
        # Updates for max_nn, max_edof, NEL
        ndof = self.ndof * 1
        # elem_type = elem_class.element_type
        nn = ent.shape[1]
        old_max_nn = self.max_nn *1
        if nn >= self.max_nn:
            self.max_nn = nn * 1
        max_nn = self.max_nn *1
        edof = ndof * nn
        old_max_edof = self.max_edof *1
        if edof >= self.max_edof:
            self.max_edof = edof * 1
        max_edof = self.max_edof *1
        nel = ent.shape[0]
        old_NEL = self.NEL*1
        self.NEL += nel
        NEL = self.NEL*1
        NDOF = self.NDOF*1

        # -----------------------------------------------------------------------------------------------
        # Elem_Group_Dict: element type for element groups
        elem_group_dict = np.full((nel), elem_type)
        if self.Elem_Group_Dict.size == 0: 
            self.Elem_Group_Dict = elem_group_dict * 1
        else:
            self.Elem_Group_Dict = np.append(self.Elem_Group_Dict, elem_group_dict, axis = 0 )
        self.elem_types_in_mesh = np.unique(self.Elem_Group_Dict)   #slow as traversing the entire array
        self.num_elem_types = self.elem_types_in_mesh.size

        # -----------------------------------------------------------------------------------------------
        # ENT: element-node table
        ent_temp = np.full((nel,max_nn), -1)
        ent_temp[:, 0:nn] = ent*1
        if self.ENT.size == 0: 
            self.ENT = ent_temp * 1
        else:
            ENT_temp = np.full((old_NEL,max_nn), -1)
            ENT_temp[:, 0:old_max_nn] = self.ENT * 1
            self.ENT = np.append(ENT_temp, ent_temp, axis = 0)

        # -----------------------------------------------------------------------------------------------
        # EFT: element DOF table
        dummy = np.arange(-ndof + 1, 1)
        dummy = np.tile(dummy,(nel, nn))
        eft = np.zeros((nel, edof))
        eft = np.repeat(ent, ndof, axis=1)
        eft = eft * ndof
        dummy = np.arange(-ndof + 1, 1)
        dummy = np.tile(dummy,(nel, nn))
        eft = eft + dummy
        eft_temp = np.full((nel,max_edof), -1)     
        eft_temp[:, 0:edof] = eft
        if self.EFT.size == 0: 
            self.EFT = eft_temp * 1
        else:
            EFT_temp = np.full((old_NEL, max_edof), -1)
            EFT_temp[:, 0:old_max_edof] = self.EFT * 1
            self.EFT = np.append(EFT_temp, eft_temp, axis = 0)

        elem_coords = np.full((nel, max_nn, self.NDIM), -1)
        for i in range(nel):
            for j in range(nn):
                elem_coords[i][j][:] = self.Node_Coords[ent[i][j]-1][:]
        if self.Elem_Coords.size == 0:
            self.Elem_Coords = elem_coords * 1
        else:
            Elem_coords = np.full((old_NEL, max_nn, self.NDIM), -1)
            Elem_coords[:, 0:old_max_nn, :] = self.Elem_Coords * 1
            self.Elem_Coords = np.append(Elem_coords, elem_coords, axis = 0)

        # -----------------------------------------------------------------------------------------------
        # pN: partial derivatives of the shape function evaluated at gauss points of isoparametric domain
        # W: weight for each gauss points
        self.max_ng = 4

        pN_temp = np.full((nel, self.max_ng, self.NDIM, self.max_nn), 0, dtype=float)
        w_temp = np.full((nel, self.max_ng), 0, dtype=float)

        if elem_type == 1: # truss element
            ng = 2
            Truss = TrussElement(nn, ng)
            pN_temp[:, :ng, :, :nn] = np.tile(Truss.pN_ele, (nel,1,1,1))
            w_temp[:, :ng] = np.tile(Truss.weight, (nel,1))

        elif elem_type == 2: # rectangular element
            ng = 4
            Rec = RectangularElement(nn, ng)
            pN_temp[:, :ng, :, :nn] = np.tile(Rec.pN_ele, (nel,1,1,1))
            w_temp[:, :ng] = np.tile(Rec.weight, (nel,1))

        elif elem_type == 3: # triangular element
            ng = 3
            Tri = TriangularElement(nn, ng)
            pN_temp[:, :ng, :, :nn] = np.tile(Tri.pN_ele, (nel,1,1,1))
            w_temp[:, :ng] = np.tile(Tri.weight, (nel,1))

        if self.pN.size == 0 or self.W.size == 0:
            self.pN = pN_temp * 1
            self.W = w_temp * 1
        else:
            pN = np.full((old_NEL, self.max_ng, self.NDIM, self.max_nn), 0, dtype=float) #ng_max = 4
            pN[:][:][:][0:old_max_nn] = self.pN * 1
            self.pN = np.append(pN, pN_temp, axis = 0)
            W = np.full((old_NEL, self.max_ng), 0, dtype=float)
            W = self.W * 1
            self.W = np.append(W, w_temp, axis = 0)


        # ! Selection_Matrix considering we only have one type of element: problem?? solved with 30 not tested
        # ! Also, recalculating S every time new elem. groups are added as NDOF changes : Problem?

        S = np.zeros((NEL, max_edof, NDOF))
        for i in range(NEL):
            for j in range(max_edof):
                if self.EFT[i][j] != -1:
                    dof_index = self.EFT[i][j] - 1
                    S[i][j][dof_index] = 1
        self.S = S

        # if self.S == np.array([]):
        #     self.S = S_temp * 1
        # else:
        #     self.S = np.append(self.S, S_temp, axis = 0)

    
# mesh = Mesh()
#
# ## test for multiple types of elements
# node_coords1 = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
# ent1 = np.array([[1, 2, 5, 4], [2, 3, 6, 5]])
# elem_type1 = 2  # rectangular
# ndof1 = 2
#
# node_coords2 = np.array([[3, 0], [3, 1]])
# ent2 = np.array([[3, 7, 6], [8, 7, 6]])
# elem_type2 = 3  # triangular
# ndof2 = 2
#
# mesh.set_nodes(node_coords1, ndof1)
# mesh.add_elem_group(ent1, elem_type1)
# mesh.set_nodes(node_coords2, ndof2)
# mesh.add_elem_group(ent2, elem_type2)
#


### test for truss elements
# node_coords = np.array([[0, 0], [1, 0], [0, 1], [-1, 0]])
# ent = np.array([[1, 2], [2, 3], [1, 3], [1, 4], [3, 4]])
# elem_type = 1  # truss
# ndof = 2

# mesh.set_nodes(node_coords, ndof)
# mesh.add_elem_group(ent, elem_type)
#
