import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from matplotlib.font_manager import FontProperties #unicode
ANGSTROM = "A"
def read_dat(fpath):
    dat_t = []
    f = open(fpath,'r') 
    lines = f.readlines()
    for line in lines:
        lsp = line.split()
        if lsp[0].startswith('#'): 
            continue
        lsp_float = [float(lsp[i]) for i  in range(len(lsp))] 
        dat_t.append(lsp_float)
    dat_np_t = np.array(dat_t)
    dat_np = dat_np_t.transpose()
    return dat_np #dat_np[0]: prop [1]: acc, [2]: cov, [3]: rmsd
color = ['#FF0000',
        '#00FF00',
        '#00FFFF',
        '#0000FF',
        '#FF00FF']

dla=  read_dat('./train.log') #dbloss(2x/2x)
dlb=  read_dat('./eval.log') #dbloss(2x/2x)

plt.rc('mathtext', fontset='cm')

fig = plt.figure(figsize = (6,4),
        facecolor = 'white',
        edgecolor = 'black',
        dpi       = 300)
prop = FontProperties(size=16) #unicode
prop2 = FontProperties(size=12) #unicode
ax = fig.add_axes([0.14,0.15,0.80,0.74])
ax.grid(b=True, axis='both',linestyle='dotted',color='black')
ax.legend(bbox_to_anchor=(1.00,1.0))
#title = '$\mathrm{Loss function}$'%tts2[tt]
#ax.set_title(r'%s'%title,fontproperties=prop)
ax.set_xlim(0.0,120.0) #14
ax.set_ylim(0.2,1.0)

ax.grid(b=True, axis='both',linestyle='dotted',color='black')
ax.plot(dla[0]   ,dla[1]   ,color=color[0] ,marker='',label='Training')
ax.plot(dlb[0]   ,dlb[1]   ,color=color[1] ,marker='',label='Validation')

plt.xlabel(r'$\mathrm{Epochs}$',fontproperties=prop)
plt.ylabel(r'$\mathrm{Loss}$',fontproperties=prop)
#plt.axvline(x=750.,linestyle='dashed')
#plt.text(750.00,0.5,r'$\mathrm{Loss\ weight\ 80\%}$',fontproperties=prop2)
#plt.text(470.00,0.5,r'$\mathrm{Loss\ weight\ 100\%}$',fontproperties=prop2)
ax.legend(bbox_to_anchor=(0.3,0.2))
plt.savefig('loss.png')
    
