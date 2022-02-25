import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Model_pro(nn.Module):
    #channels:
    # atom types 0~2 (C,N,O)
    def __init__(self, grid = 0.5, n_grid =[48,48,48] ):
        super(Model_pro, self).__init__()
        self.grid   = grid
        self.n_grid = n_grid

    def bbox(self, vec, b_size = 3, grid = 0.5, n_grid=[48,48,48]):

        min_pt_f = (vec-b_size)/grid
        max_pt_f = 2+(vec+b_size)/grid
        #min_pt:max_pt -> count min_pt ~ max_pt-1 -> add 1 to max_pt_f
        #int(max_pt) will be lesser than max_pt   -> add another 1  
    
        min_pt = [max(0, min(n_grid[i],int(min_pt_f[i]))) for i in range(3)]
        max_pt = [max(0, min(n_grid[i],int(max_pt_f[i]))) for i in range(3)]
        return min_pt,max_pt

    def forward(self, vecs): #vecs contain only one ch
        pro_keys = ['C','N','O'] 
        vdw_dic = {'C':1.7,'N':1.55,'O':1.52} 
        output = torch.FloatTensor(3,self.n_grid[0],self.n_grid[1],self.n_grid[2]).fill_(0)
        
        for ch_idx, ch in enumerate(pro_keys):
            vdw= vdw_dic[ch]
            vec_list = vecs[ch]

            for vec_idx, vec0 in enumerate(vec_list):
                vec = vec0 + np.array(self.n_grid) * 0.5 - np.array([0.5,0.5,0.5])

                min_pt,max_pt =  self.bbox(vec, b_size = 1.5*vdw, grid = self.grid, n_grid=self.n_grid)
                grid = self.grid
                r = vdw
                e2 = np.exp(2)
                r2 = r*r
                #
                x = torch.arange(min_pt[0],max_pt[0],1, out=torch.LongTensor() )
                y = torch.arange(min_pt[1],max_pt[1],1, out=torch.LongTensor() )
                z = torch.arange(min_pt[2],max_pt[2],1, out=torch.LongTensor() )
                xind,yind,zind =torch.meshgrid([x,y,z])
                #
                #
                xv = grid*xind.float()
                yv = grid*yind.float()
                zv = grid*zind.float()
                #
                vec_x = torch.full_like(xv,vec[0])
                vec_y = torch.full_like(yv,vec[1])
                vec_z = torch.full_like(zv,vec[2])
                #
                dx = vec_x-xv
                dy = vec_y-yv
                dz = vec_z-zv
                #
                dx2 = dx*dx 
                dy2 = dy*dy 
                dz2 = dz*dz 
                #
                d2 = (dx2+dy2+dz2)
                d  = torch.sqrt(d2)
                #
                f1  = torch.exp(-2.0*d2/r2) #short dist
                f2  = ((4.0*d2)/(e2*r2) - (12.0*d)/(e2*r) + 9./e2) #medium dist
                f3  = torch.full_like(d,0.) #long dist
                #torch.where: (contidion, true,false)
                f4  = torch.where(d<(1.5*r),f2,f3)
                mask  = torch.where(d<r,f1,f4)
                output[ch_idx, min_pt[0]:max_pt[0] , min_pt[1]:max_pt[1], min_pt[2]:max_pt[2] ] += mask[:,:,:]
        return output

def build_cuda(vecs,grid=0.5,n_grid=48):
    
    #prepare model
    #device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_pro = Model_pro(grid,[n_grid,n_grid,n_grid])
    model_pro.to(device)
    out_pro   =model_pro(vecs).cpu().detach().numpy()
    #out_pro   =model_pro(vecs).detach().numpy()
    
    result_pro = np.expand_dims(out_pro,axis=0)
    
    return result_pro

