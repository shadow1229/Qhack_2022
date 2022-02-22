import os,sys,glob
import torch
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers

class QonvLayer_last(nn.Module):
    def __init__(self, stride=2, device="default.qubit", wires=4, circuit_layers=2, n_rotations=4, seed=None):
        super(QonvLayer_last, self).__init__()
        
        # init device
        self.wires = wires
        self.dev = qml.device(device, wires=self.wires)
        
        self.stride = stride
        #self.out_channels = min(out_channels, wires)
        
        if seed is None:
            seed = np.random.randint(low=0, high=10e6)
            
        print("Initializing Circuit with random seed", seed)
        
        # random circuits
        @qml.qnode(device=self.dev, interface="torch")
        def circuit1(inputs, weights):
            n_inputs=4
            # Encoding of 4 classical input values
            for j in range(n_inputs):
                qml.RY(min(np.pi, np.pi*inputs[j]), wires=j) 
            # Random quantum circuit
            RandomLayers(weights, wires=list(range(self.wires)), seed=seed)
            
            # Measurement producing 4 classical output values
            #return [qml.expval(qml.PauliZ(j)) for j in range(self.out_channels)]
            return [qml.expval(qml.PauliZ(0))]
        
        @qml.qnode(device=self.dev, interface="torch")
        def circuit2(inputs, weights):
            n_inputs=4
            # Encoding of 4 classical input values
            for j in range(n_inputs):
                qml.RY(min(np.pi, np.pi*inputs[j]), wires=j) 
            # Random quantum circuit
            RandomLayers(weights, wires=list(range(self.wires)), seed=seed)
            
            # Measurement producing 4 classical output values
            #return [qml.expval(qml.PauliZ(j)) for j in range(self.out_channels)]
            return [qml.expval(qml.PauliZ(0))]
        
        weight_shapes = {"weights": [circuit_layers, n_rotations]}
        self.circuit1 = qml.qnn.TorchLayer(circuit1, weight_shapes=weight_shapes)
        self.circuit2 = qml.qnn.TorchLayer(circuit2, weight_shapes=weight_shapes)
    
    
    def draw(self):
        # build circuit by sending dummy data through it
        _ = self.circuit1(inputs=torch.from_numpy(np.zeros(4)))
        print(self.circuit1.qnode.draw())
        self.circuit1.zero_grad()
        
        _ = self.circuit2(inputs=torch.from_numpy(np.zeros(4)))
        print(self.circuit2.qnode.draw())
        self.circuit2.zero_grad()
        
    
    def forward(self, x):
        n_batch = x.shape[0]
        ch = x.shape[1]
        d  = x.shape[2]
        h  = x.shape[3]
        w  = x.shape[4]
                        
        kernel_size = 2        
        d_out = (d-kernel_size) // self.stride + 1
        h_out = (h-kernel_size) // self.stride + 1
        w_out = (w-kernel_size) // self.stride + 1
        
        
        out = torch.zeros((n_batch, 2*ch, d_out, h_out, w_out))
        
        # Loop over the coordinates of the top-left pixel of 2x2x2 cubes
        for b in range(n_batch):
            for c in range(ch):
                for j in range(0, d_out, self.stride):
                    for k in range(0, h_out, self.stride):
                        for l in range(0, w_out, self.stride):
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
                                      x[b, c , j  , k  , l  ], 
                                      x[b, c , j+1, k+1, l  ], 
                                      x[b, c , j  , k+1, l+1], 
                                      x[b, c , j+1, k  , l+1]]) 

                            y2 = torch.Tensor([
                                      x[b, c , j+1, k+1, l+1],
                                      x[b, c , j  , k  , l+1],
                                      x[b, c , j+1, k  , l  ],
                                      x[b, c , j  , k+1, l  ]])
                            
                            q_results_1 = self.circuit1(inputs=y1)
                            q_results_2 = self.circuit2(inputs=y2)
                            # Assign expectation values to different channels of the output pixel (j/2, k/2)
                            out[b, c   , j // kernel_size, k // kernel_size, l // kernel_size] = q_results_1[0]
                            out[b, c+ch, j // kernel_size, k // kernel_size, l // kernel_size] = q_results_2[0]
                        
        return out
class Net_v2(nn.Module):
    def __init__(self):
        super(Net_v2, self).__init__()
        self.conv_i1  = nn.Conv3d(3,   8, 1,padding=0)
        self.conv1   = atr_res_12(8)
        self.conv2   = atr_res_12(8)
        self.maxpool1 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool

        self.conv_i2  = nn.Conv3d(8,   64, 1,padding=0)
        self.conv3   = atr_res_12(64)
        self.conv4   = atr_res_12(64)
        self.maxpool2 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool
        
        self.conv5   = atr_res_12(64)
        self.conv6   = atr_res_12(64)
        self.maxpool3 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool
        
        self.conv7   = atr_res_12(64)
        self.conv8   = atr_res_12(64)
        self.maxpool4 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool
        
        self.qonv_last = QonvLayer_last(stride=2, device="default.qubit",seed=0)
        self.conv9   = atr_res_12(64)
        self.conv10  = atr_res_12(64)
        self.maxpool5 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool

        self.conv_e1 = nn.Conv3d(128, 1, 1) #fc layer
        
        
    def forward(self, x_init):
        #print('init_size',x.size())
        x = F.elu(self.conv_i1(x_init))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = F.elu(self.conv_i2(x))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool3(x)
        
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.maxpool4(x)
        
        x = self.qonv_last(x)
        #x = self.conv9(x)
        #x = self.conv10(x)
        #x = self.maxpool5(x)

        result = self.conv_e1(x) 

        return result
class Net_v1(nn.Module):
    def __init__(self):
        super(Net_v1, self).__init__()
        self.conv_i1  = nn.Conv3d(3,   8, 1,padding=0)
        self.conv1   = atr_res_12(8)
        self.conv2   = atr_res_12(8)
        self.maxpool1 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool

        self.conv_i2  = nn.Conv3d(8,   64, 1,padding=0)
        self.conv3   = atr_res_12(64)
        self.conv4   = atr_res_12(64)
        self.maxpool2 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool
        
        self.conv5   = atr_res_12(64)
        self.conv6   = atr_res_12(64)
        self.maxpool3 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool
        
        self.conv7   = atr_res_12(64)
        self.conv8   = atr_res_12(64)
        self.maxpool4 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool
        
        self.conv9   = atr_res_12(64)
        self.conv10  = atr_res_12(64)
        self.maxpool5 = nn.MaxPool3d(2, stride=2) #2x2x2 maxpool

        self.conv_e1 = nn.Conv3d(64, 1, 1) #fc layer
        
        
    def forward(self, x_init):
        #print('init_size',x.size())
        x = F.elu(self.conv_i1(x_init))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        
        x = F.elu(self.conv_i2(x))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool3(x)
        
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.maxpool4(x)
        
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.maxpool5(x)

        result = self.conv_e1(x) 

        return result

class atr_res_12(nn.Module):
    '''atrous_conv, dilation 1, dilation 2 w/resnet'''
    def __init__(self, ch):
        super(atr_res_12, self).__init__()
        self.conv11 = nn.Conv3d(ch,   ch, 3,padding=1,dilation=1)
        self.conv12 = nn.Conv3d(ch,   ch, 3,padding=2,dilation=2)
        self.conv13 = nn.Conv3d(2*ch, ch, 1,padding=0)
        self.bn1 = nn.BatchNorm3d(ch)

    def forward(self, x):
        x0 = x
        y1 = F.elu(self.conv11(x))
        y2 = F.elu(self.conv12(x))
        x  = F.elu(x0 + self.bn1(self.conv13(torch.cat((y1,y2),1))))
        
        return x

