
f = open('cnn_net_v0.log','r')
lines = f.readlines()
out_t = open('train.log','w')
out_e = open('eval.log','w')

t_temp = [[] for i in range(1000)]
e_temp = [[] for i in range(1000)]

for line in lines:
    lsp = line.split()
    if line.startswith('T'):
        i    = int(lsp[0].lstrip('T[').rstrip(',')) -1
        step = int(lsp[1].rstrip(']'))
        loss = float(lsp[-1])
        t_temp[i].append((step,loss))

    if line.startswith('E'):
        i    = int(lsp[0].lstrip('E[').rstrip(',')) -1
        step = int(lsp[1].rstrip(']'))
        loss = float(lsp[-1])
        e_temp[i].append((step,loss))


for i in range(len(t_temp)):
    len_j = len(t_temp[i])
    step = i
    loss = 0

    print (t_temp[i])
    for j in range(len_j):
        loss += t_temp[i][j][1]
    loss_avg = loss / float(len(t_temp[i]))

    out_t.write('%10.4f %10.6f\n'%(step,loss_avg))

for i in range(len(e_temp)):
    len_j = len(e_temp[i])
    step = i
    loss = 0
    for j in range(len_j):
        loss += e_temp[i][j][1]
    loss_avg = loss / float(len(e_temp[i]))

    out_e.write('%10.4f %10.6f\n'%(step,loss_avg))
        
#for i in range(len(e_temp)):
#    len_j = len(e_temp[i])
#    for j in range(len_j):
#        step = i+ (1+j)/float(len_j)
#        loss = e_temp[i][j][1]
#
#        out_e.write('%10.4f %10.6f\n'%(step,loss))
