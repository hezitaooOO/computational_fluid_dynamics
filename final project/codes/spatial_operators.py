import numpy as np
import scipy.sparse as scysparse
import sys
from pdb import set_trace as keyboard
import scipy.sparse as scysparse
import scipy.sparse.linalg as spysparselinalg
import scipy.linalg as scylinalg        # non-sparse linear algebra

############################################################
############################################################

def create_DivGrad_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,boundary_conditions="Homogeneous Neumann"):
	# defaults to "Homogeneous Dirichlet"
	
     possible_boundary_conditions = ["Homogeneous Dirichlet","Homogeneous Neumann"]

     if not(boundary_conditions in possible_boundary_conditions):
         sys.exit("Boundary conditions need to be either: " +
                  repr(possible_boundary_conditions))

     # numbering with -1 means that it is not a fluid cell (i.e. either ghost cell or external)
     numbered_pressureCells = -np.ones(Xc.shape,dtype='int64')
     jj_C,ii_C = np.where(pressureCells_Mask==True)
     Np = len(jj_C)         # total number of pressure nodes, not necessarily equal to Nxc*Nyc
     numbered_pressureCells[jj_C,ii_C] = range(0,Np) # automatic numbering done via 'C' flattening
     
     inv_DxC = 1./Dxc[jj_C,ii_C]
     inv_DyC = 1./Dyc[jj_C,ii_C]

     inv_DxE = 1./(Xc[jj_C,ii_C+1]-Xc[jj_C,ii_C])
     inv_DyN = 1./(Yc[jj_C+1,ii_C]-Yc[jj_C,ii_C])

     inv_DxW = 1./(Xc[jj_C,ii_C]-Xc[jj_C,ii_C-1])
     inv_DyS = 1./(Yc[jj_C,ii_C]-Yc[jj_C-1,ii_C])
     
     DivGrad = scysparse.csr_matrix((Np,Np),dtype="float64") # initialize with all zeros
	 
     iC = numbered_pressureCells[jj_C,ii_C]
     iE = numbered_pressureCells[jj_C,ii_C+1]
     iW = numbered_pressureCells[jj_C,ii_C-1]
     iS = numbered_pressureCells[jj_C-1,ii_C]
     iN = numbered_pressureCells[jj_C+1,ii_C]

     # consider pre-multiplying all of the weights by the local value of dx*dy

     # start by creating operator assuming homogeneous Neumann

     ## if east node is inside domain
     east_node_mask = (iE!=-1)
     ii_center = iC[east_node_mask]
     ii_east   = iE[east_node_mask]
     inv_dxc_central = inv_DxC[ii_center]
     inv_dxc_east    = inv_DxE[ii_center]
     DivGrad[ii_center,ii_east]   += inv_dxc_central*inv_dxc_east
     DivGrad[ii_center,ii_center] -= inv_dxc_central*inv_dxc_east
     
     ## if west node is inside domain
     west_node_mask = (iW!=-1)
     ii_center  = iC[west_node_mask]
     ii_west    = iW[west_node_mask]
     inv_dxc_central = inv_DxC[ii_center]
     inv_dxc_west    = inv_DxW[ii_center]
     DivGrad[ii_center,ii_west]   += inv_dxc_central*inv_dxc_west
     DivGrad[ii_center,ii_center] -= inv_dxc_central*inv_dxc_west

	 ## if north node is inside domain
     north_node_mask = (iN!=-1)
     ii_center  = iC[north_node_mask]
     ii_north   = iN[north_node_mask]
     inv_dyc_central  = inv_DyC[ii_center]
     inv_dyc_north    = inv_DyN[ii_center]
     DivGrad[ii_center,ii_north]   += inv_dyc_central*inv_dyc_north
     DivGrad[ii_center,ii_center]  -= inv_dyc_central*inv_dyc_north

      ## if south node is inside domain
     south_node_mask = (iS!=-1)
     ii_center  = iC[south_node_mask]
     ii_south   = iS[south_node_mask]
     inv_dyc_central  = inv_DyC[ii_center]
     inv_dyc_south    = inv_DyS[ii_center]
     DivGrad[ii_center,ii_south]   += inv_dyc_central*inv_dyc_south
     DivGrad[ii_center,ii_center]  -= inv_dyc_central*inv_dyc_south

	 # if Dirichlet boundary conditions are requested, need to modify operator
     if boundary_conditions == "Homogeneous Dirichlet":

		# for every east node that is 'just' outside domain
		east_node_mask = (iE==-1)&(iC!=-1)
		ii_center = iC[east_node_mask]
		inv_dxc_central = inv_DxC[ii_center]
		inv_dxc_east    = inv_DxE[ii_center]
		DivGrad[ii_center,ii_center]  -= 2.*inv_dxc_central*inv_dxc_east
		
		# for every west node that is 'just' outside domain
		west_node_mask = (iW==-1)&(iC!=-1)
		ii_center = iC[west_node_mask]
		inv_dxc_central = inv_DxC[ii_center]
		inv_dxc_west    = inv_DxW[ii_center]
		DivGrad[ii_center,ii_center]  -= 2.*inv_dxc_central*inv_dxc_west

		# for every north node that is 'just' outside domain
		north_node_mask = (iN==-1)&(iC!=-1)
		ii_center = iC[north_node_mask]
		inv_dyc_central  = inv_DyC[ii_center]
		inv_dyc_north    = inv_DyN[ii_center]
		DivGrad[ii_center,ii_center]  -= 2.*inv_dyc_central*inv_dyc_north

		# for every south node that is 'just' outside domain
		south_node_mask = (iS==-1)&(iC!=-1)
		ii_center = iC[south_node_mask]
		inv_dyc_central  = inv_DyC[ii_center]
		inv_dyc_south    = inv_DyS[ii_center]
		DivGrad[ii_center,ii_center]  -= 2.*inv_dyc_central*inv_dyc_south


     return DivGrad
