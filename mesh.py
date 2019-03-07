
import numpy as np

class Mesh():

    def __init__(self):
        self.NEL = 0                                #total num of elements in the mesh
        self.NN = 0                                 #total num of nodes in the mesh
        self.ndof = 0                               #num of dof for each node in the mesh(Problem Dependent)
        self.max_nn = 0                             #num of nodes per element for element type with max num of nodes
        self.max_edof = 0                           #num of dof of element type with max num of dof
        self.NDOF = 0                               #Global dof for the entire mesh
        self.NDIM = 0                               #num of dimensions in which the mesh is present(Problem Dependent)
        self.ENT = np.array([])                     #max. num of nodes per element is set as max_nn
        self.EFT = np.array([])                     #max. num of dof per element is set as max_edof
        self.S = np.array([])                       #max. num of dof per element is set as max_edof
        self.Node_Coords = np.array([])             #Coords of each node in the mesh
        self.Elem_Group_Dict = np.array([])         #element type of each element in the mesh

    def set_nodes(self, node_coords, ndof):
        num_new_nodes = node_coords.shape[0]
        self.NN += num_new_nodes
        if self.NDIM == 0:
            self.NDIM = node_coords.shape[1]
            self.Node_Coords = node_coords * 1
            self.ndof = ndof
        else:
            self.Node_Coords = np.append(self.Node_Coords, node_coords, axis = 0)
        
        

    # DOF_index and node_index starting from 1
    def add_elem_group(self, ent ,elem_class): 
        ndof = self.ndof * 1
        elem_type = elem_class.elem_type
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
        new_dof = edof * nel
        old_NEL = self.NEL*1
        self.NEL += nel
        self.NDOF += new_dof
        NEL = self.NEL*1
        NDOF = self.NDOF*1
        elem_group_dict = np.full((nel,1), elem_type)
        if self.Elem_Group_Dict == np.array([]): 
            self.Elem_Group_Dict = elem_group_dict * 1
        else:
            self.Elem_Group_Dict = np.append(self.Elem_Group_Dict, elem_group_dict, axis = 0 )

        ent_temp = np.full((nel,max_nn), -1)
        ent_temp[:, 0:nn] = ent*1
        if self.ENT == np.array([]):
            self.ENT = ent_temp * 1
        else:
            ENT_temp = np.full((old_NEL,max_nn), -1)
            ENT_temp[:, 0:old_max_nn] = self.ENT * 1
            self.ENT = np.append(ENT_temp, ent_temp, axis = 0)


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
        if self.EFT == np.array([]):
            self.EFT = eft_temp * 1
        else:
            EFT_temp = np.full((old_NEL, max_edof), -1)
            EFT_temp[:, 0:old_max_edof] = self.EFT * 1
            self.EFT = np.append(EFT_temp, eft_temp, axis = 0)

        

        # Selection_Matrix considering we only have one type of element: problem?? solved with 30 not tested
        # Also, recalculating S every time new elem. groups are added as NDOF changes : Problem?

        S = np.zeros((NEL, max_edof, NDOF))
        for i in range(NEL):
            for j in range(max_edof):
                if self.EFT[i][j] != -1:
                    dof_index = self.EFT[i][j]
                    S[i][j][dof_index] = 1

        # if self.S == np.array([]):
        #     self.S = S_temp * 1 
        # else:
        #     self.S = np.append(self.S, S_temp, axis = 0)




        
        


        
        



    

    
