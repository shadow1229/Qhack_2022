# -*- coding: utf-8 -*-
import os,sys,glob,copy,pickle
import torch
import random
import numpy as np
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
from lib_build_torch import build_cuda

#training data preparation 

def get_rotmat(x1,x2,x3):
    #James Arvo, Fast Random Rotation Matrices (1992)
    #returns uniform distribution of rotation matrices
    # 0< x1,x2,x3 < 1
    R = np.array([ [ np.cos(2.*np.pi*x1) , np.sin(2.*np.pi*x1) , 0],
                   [-np.sin(2.*np.pi*x1) , np.cos(2.*np.pi*x1) , 0],
                   [                   0 ,                   0 , 1] ])

    v = np.array(  [ np.cos(2.*np.pi*x2)*np.sqrt(x3),
                     np.sin(2.*np.pi*x2)*np.sqrt(x3),
                     np.sqrt(1.-x3) ])[np.newaxis]
    H = np.eye(3) - 2.*np.matmul(v.T,v)

    M = -np.matmul(H,R)
    return M

def rotate(vecs,x1,x2,x3):
    #vecs:{'C':[Nx3] , 'N':..., 'O':...}
    result = {}
    rotmat = get_rotmat(x1,x2,x3)

    for ch in vecs.keys():
        if len(vecs[ch]) == 0:
            result[ch] = []
        else:
            result[ch] = np.matmul(vecs[ch],rotmat)
    return result

def build_epoch(env, epoch,  train = True, batchsize=4 ):
    # trainset / testset = [0,{'C':[vecs] , 'N':[vecs], 'O':[vecs] } ]
    #0: HOH , 1: NA. train: 400 cases HOH, NA each, test: 54 cases each
    sets0     = env['sets']
    device    = env['device']
    n_grid    = env['n_grid']

    if train:
        #random.shuffle(sets0)
        sets = []
        for s in sets0: #rotate randomly
            i = random.random()
            j = random.random()
            k = random.random()
            vecs_new = rotate(s[1],i,j,k)
            sets.append([s[0], vecs_new])
    else:
        sets = sets0

    inputs  = []
    for i in range(len(sets)):
        print(epoch, i)
        inputs.append( [sets[i][0], build_cuda(sets[i][1],n_grid=n_grid) ])

    with open('dat/data_%03d.pic'%epoch,'wb') as f:
        pickle.dump(inputs,f) 

def build(trainset=[],testset=[],n_grid=32, batchsize=4):
    # trainset / testset = [0,{'C':[vecs] , 'N':[vecs], 'O':[vecs] } ]
    #0: HOH , 1: NA. train: 400 cases HOH, NA each, test: 54 cases each
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    env_train = {'sets':trainset, 'device':device,
                 'n_grid':n_grid}
    
    env_test = {'sets':testset, 'device':device,
                 'n_grid':n_grid}
    
    #for epoch in range(100):  # loop over the dataset multiple times
    #    build_epoch(env_train, epoch,train = True, batchsize=batchsize)
    
    build_epoch(env_test, 500,train = False, batchsize=batchsize)
             

if __name__ == '__main__':

    with open('HOH_train/vs.pic','rb') as f:
        HOH_tr_vecs = pickle.load(f)
    with open('HOH_test/vs.pic','rb') as f:
        HOH_te_vecs = pickle.load(f)
    with open('NA_train/vs.pic','rb') as f:
        NA_tr_vecs = pickle.load(f)
    with open('NA_test/vs.pic','rb') as f:
        NA_te_vecs = pickle.load(f)
    trainset = []
    testset  = []

    for vecs in HOH_tr_vecs:
        trainset.append([0,vecs])
    for vecs in HOH_te_vecs:
        testset.append([0,vecs])
    
    for vecs in NA_tr_vecs:
        trainset.append([1,vecs])
    for vecs in NA_te_vecs:
        testset.append([1,vecs])

    
    build(trainset=trainset, testset=testset, n_grid=32, batchsize=4)
    
    
