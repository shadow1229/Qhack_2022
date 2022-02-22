#https://github.com/PlanQK/TrainableQuantumConvolution
#trainable quanvolutional neural network

#Import
import os,sys,glob,copy,pickle
import random
import torch
from torch import nn
import torch.nn.functional as F
import networks_quanv
#import torchvision

import torch.optim as optim
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers

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


def test_epoch(env, epoch, dat = None, batchsize=4):
    with torch.no_grad():
        run_epoch(env, epoch, train=False, dat = dat, batchsize=batchsize)

def train_epoch(env, epoch, dat = None, batchsize=4):
    run_epoch(env, epoch, train=True, dat = dat, batchsize=batchsize)
    
def run_epoch(env, epoch, train=True, dat= None, batchsize=4 ,dr=None):
    # trainset / testset = [0,{'C':[vecs] , 'N':[vecs], 'O':[vecs] } ]
    #0: HOH , 1: NA. train: 400 cases HOH, NA each, test: 54 cases each
    sets0     = env['sets']
    net       = env['net']
    device    = env['device']
    optimizer = env['optimizer']
    log       = env['log']
    n_grid    = env['n_grid']
    running_loss = 0.0
    total_loss = 0.0

    input_batch  = []
    answer_batch = []
    if dat == None:
        print('NO DATA!! - building...')
        if train:
            random.shuffle(sets0)
            sets = []
            for s in sets0: #rotate randomly
                i = random.random()
                j = random.random()
                k = random.random()
                vecs_new = rotate(s[1],i,j,k)
                sets.append([s[0], vecs_new])
        else:
            sets = sets0

        for i in range(len(sets) //batchsize):
            #print(i)
            input_batch.append([build_cuda(sets[batchsize*i+j][1],n_grid=n_grid) for j in range(batchsize)])
            answer_batch.append( np.array([ [[[[[ sets[batchsize*i + j][0] ]]]]] for j in range(batchsize)]))
    else:
        if train:
            random.shuffle(dat)
        
        for i in range(len(dat) //batchsize):
            input_batch.append([ dat[batchsize*i + j ][1] for j in range(batchsize)])
            answer_batch.append( np.array([ [[[[[ dat[batchsize*i + j][0] ]]]]] for j in range(batchsize)]))

    for i in range(len(answer_batch)):
        np_inputs  = np.vstack(input_batch[i])
        np_answers = np.vstack(answer_batch[i])
        #print(np_inputs.shape) 
        inputs  = torch.tensor(torch.FloatTensor(np_inputs),  device=device ,requires_grad=False, dtype=torch.float32)
        answers = torch.tensor(torch.FloatTensor(np_answers), device=device ,requires_grad=False, dtype=torch.float32)
        
        #print(epoch, i,'start')
        inputs  = inputs.to(device=device) #gpu
        answers = answers.to(device=device) #gpu
         
        if train:
            optimizer.zero_grad()
            outputs = net.train()(inputs) 
            lossf = nn.BCEWithLogitsLoss()
            loss = lossf(outputs, answers)
            loss.backward()
            optimizer.step()

        else:
            outputs = net.eval()(inputs) 
            lossf = nn.BCEWithLogitsLoss()
            loss = lossf(outputs, answers)

        #print(epoch, i,'end')
        # print statistics
        running_loss += loss.item()
        total_loss   += loss.item()
        if train:
            if i % 5 == 4:
                print('T[%d, %5d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss/5.))
                log.write('T[%d, %5d] loss: %.6f\n' %
                      (epoch + 1, i + 1, running_loss/5.))
    
                running_loss = 0.0

    if (not train) :
       print('E[%d, %5d] loss: %.6f' %
             (epoch + 1, i + 1, total_loss/float(len(answer_batch))))
       log.write('E[%d, %5d] loss: %.6f\n' %
             (epoch + 1, i + 1, total_loss/float(len(answer_batch))))

def training(nets,names ,trainset=[],testset=[],n_grid=32, batchsize=4):
    # trainset / testset = [0,{'C':[vecs] , 'N':[vecs], 'O':[vecs] } ]
    #0: HOH , 1: NA. train: 400 cases HOH, NA each, test: 54 cases each
    for net_idx, net in enumerate(nets):
        name = names[net_idx]
        if not os.access(name,0):
            os.mkdir(name)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        if torch.cuda.device_count() >1:
            net = nn.DataParallel(net)
        
        states = glob.glob('%s/cnn_state_*.bin'%(name))
        states.sort()
    
        if len(states) > 0:
            start_epoch = int(states[-1].split('/')[-1].split('.')[0].split('_')[-1])
            net.module.load_state_dict(torch.load(states[-1]))
        else:
            start_epoch = 0
        
        optimizer = optim.Adam(net.parameters(), lr=0.03)
        
        log = open('%s/cnn_net_v%d.log'%(name,net_idx),'a')
        env_train = {'sets':trainset, 'net':net, 'device':device,
                     'optimizer':optimizer, 'log':log,'n_grid':n_grid}
        
        env_test = {'sets':testset, 'net':net, 'device':device,
                     'optimizer':optimizer, 'log':log,'n_grid':n_grid}
        
        
        with open('dat/test.pic','rb') as f:
            dat_test = pickle.load(f)

        for epoch in range(start_epoch, 1000):  # loop over the dataset multiple times

            with open('dat/data_%03d.pic'%(epoch%100),'rb') as f:
                dat_train = pickle.load(f)
            train_epoch(env_train, epoch,dat = dat_train, batchsize=batchsize)
            test_epoch(env_test, epoch, dat = dat_test)
                 
            #if epoch%5 ==4:
            #    test_epoch(env_test, epoch, dat = dat_test)

            #if epoch%5 == 4:
            #    #save state:
            torch.save(net.state_dict(), '%s/cnn_state_%05d.bin'%(name,(epoch+1)))
    
        log.close()
        del(log)

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

    nets  = [networks_quanv.Net_v2()]
    names = ['lig_v2']
    
    training(nets,names, trainset=trainset, testset=testset, n_grid=32, batchsize=4)
    
    
