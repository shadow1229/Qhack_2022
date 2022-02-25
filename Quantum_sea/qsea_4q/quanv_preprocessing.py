#https://github.com/PlanQK/TrainableQuantumConvolution
#trainable quanvolutional neural network

#Import
import os,sys,glob,copy,pickle
import random
import torch
from torch import nn
import torch.nn.functional as F
#import torchvision

import torch.optim as optim
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers

def quanv_1st_4q(vox, seed1 = 0, seed2 = 1):
    #settings
    device="default.qubit"
    wires = 4
    circuit_layers = 2
    n_rotations = 4
    stride = 2
    dev = qml.device(device, wires=wires)
    out_channels = 4
    #settings_end

    @qml.qnode(device=dev, interface="torch")
    def circuit1(inputs, weights):
        n_inputs=4
        # Encoding of 4 classical input values
        for j in range(n_inputs):
            qml.RX(min(np.pi, np.pi*inputs[3*j+2]), wires=j)  #O
            qml.RY(min(np.pi, np.pi*inputs[3*j+1]), wires=j)  #N
            qml.RZ(min(np.pi, np.pi*inputs[3*j+0]), wires=j)  #C
        # Random quantum circuit
        RandomLayers(weights, wires=list(range(wires)), seed=seed1)
        
        # Measurement producing 4 classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(out_channels)]
    
    @qml.qnode(device=dev, interface="torch")
    def circuit2(inputs, weights):
        n_inputs=4
        # Encoding of 4 classical input values
        for j in range(n_inputs):
            qml.RX(min(np.pi, np.pi*inputs[3*j+2]), wires=j)  #O
            qml.RY(min(np.pi, np.pi*inputs[3*j+1]), wires=j)  #N
            qml.RZ(min(np.pi, np.pi*inputs[3*j+0]), wires=j)  #C
        # Random quantum circuit
        RandomLayers(weights, wires=list(range(wires)), seed=seed2)
        
        # Measurement producing 4 classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(out_channels)]
    
    weight_shapes = {"weights": [circuit_layers, n_rotations]}
    circuit1l = qml.qnn.TorchLayer(circuit1, weight_shapes=weight_shapes)
    circuit2l = qml.qnn.TorchLayer(circuit2, weight_shapes=weight_shapes)
    
    b  = vox.shape[0]
    ch = vox.shape[1]
    d  = vox.shape[2]
    h  = vox.shape[3]
    w  = vox.shape[4]

    kernel_size = 2        
    d_out = (d-kernel_size) // stride + 1
    h_out = (h-kernel_size) // stride + 1
    w_out = (w-kernel_size) // stride + 1
    
    
    out = torch.zeros((b,8, d_out, h_out, w_out))
    
    # Loop over the coordinates of the top-left pixel of 2x2x2 cubes
    for b in range(b):
        for j in range(0, d_out, stride):
            for k in range(0, h_out, stride):
                for l in range(0, w_out, stride):
                    # Process a squared 2x2x2 region of the data with a quantum circuit
                    # To reduce the amount of qubit, the data is splited with 2 tetrahedrons
                    # First
                    #        x[b, :, j  , k  , l  ],
                    #        x[b, :, j+1, k+1, l  ],
                    #        x[b, :, j  , k+1, l+1],
                    #        x[b, :, j+1, k  , l+1],
                    # Second
                    #        x[b, :, j+1, k+1, l+1],
                    #        x[b, :, j  , k  , l+1],
                    #        x[b, :, j+1, k  , l  ],
                    #        x[b, :, j  , k+1, l  ],
                    
                    y1 = torch.Tensor([
                              vox[ b, 0 , 2*j  , 2*k  , 2*l  ], vox[ b, 1 , 2*j  , 2*k  , 2*l  ], vox[ b, 2 , 2*j  , 2*k  , 2*l  ],
                              vox[ b, 0 , 2*j+1, 2*k+1, 2*l  ], vox[ b, 1 , 2*j+1, 2*k+1, 2*l  ], vox[ b, 2 , 2*j+1, 2*k+1, 2*l  ],
                              vox[ b, 0 , 2*j  , 2*k+1, 2*l+1], vox[ b, 1 , 2*j  , 2*k+1, 2*l+1], vox[ b, 2 , 2*j  , 2*k+1, 2*l+1],
                              vox[ b, 0 , 2*j+1, 2*k  , 2*l+1], vox[ b, 1 , 2*j+1, 2*k  , 2*l+1], vox[ b, 2 , 2*j+1, 2*k  , 2*l+1],    ]) 

                    y2 = torch.Tensor([
                              vox[ b, 0 , 2*j+1, 2*k+1, 2*l+1], vox[ b, 1 , 2*j+1, 2*k+1, 2*l+1], vox[ b, 2 , 2*j+1, 2*k+1, 2*l+1],
                              vox[ b, 0 , 2*j  , 2*k  , 2*l+1], vox[ b, 1 , 2*j  , 2*k  , 2*l+1], vox[ b, 2 , 2*j  , 2*k  , 2*l+1],
                              vox[ b, 0 , 2*j+1, 2*k  , 2*l  ], vox[ b, 1 , 2*j+1, 2*k  , 2*l  ], vox[ b, 2 , 2*j+1, 2*k  , 2*l  ],
                              vox[ b, 0 , 2*j  , 2*k+1, 2*l  ], vox[ b, 1 , 2*j  , 2*k+1, 2*l  ], vox[ b, 2 , 2*j  , 2*k+1, 2*l  ],  ])
                    
                    q_results_1 = circuit1l(inputs=y1)
                    q_results_2 = circuit2l(inputs=y2)
                    # Assign expectation values to different channels of the output pixel (j/2, k/2)
                    for c in range(4):
                        out[b, c   , j // kernel_size, k // kernel_size, l // kernel_size] = q_results_1[c]
                        out[b, c+4 , j // kernel_size, k // kernel_size, l // kernel_size] = q_results_2[c]
    
    del(circuit1l)                
    del(circuit2l)                
    return out
def build_4q(epoch = 0, n_grid=32):
    with open('dat/data_%03d.pic'%(epoch),'rb') as f:
        dat_train = pickle.load(f)

    result = []
    for i, dat in enumerate(dat_train):
        if os.access('dat/data_%03d_4q.pic'%(epoch), 0):
            continue
        input_vox = dat[1]
        answer    = dat[0]
        quanv0 =  quanv_1st_4q(input_vox, seed1 = 0, seed2 = 1)
        quanv = quanv0.detach().numpy()
        result.append( [ answer, quanv] )
        result = [ answer, quanv] 
    
    with open('dat/data_%03d_4q.pic'%(epoch),'wb') as f:
       pickle.dump(result,f)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        epoch = int(sys.argv[1])
        build_4q(epoch,n_grid=32)
    else:
        #preprocess training set
        for i in range(100):
            build_4q(i,n_grid=32)
        #preprocess test set
        build_4q(999,n_grid=32)
    
    
