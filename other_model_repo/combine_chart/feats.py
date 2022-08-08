import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import pdb

def mjrFormatter(x, pos):
    if x==1:
        return '1'
    elif x==0:
        return '0'
    else:
        return ("%.1f" % x).lstrip('0')

#matplotlib.rcParams['text.usetex'] = True
    


x4 = [0.074,0.174,0.274,0.374,0.474]
x3 = [ 0.058,0.158,0.258,0.358,0.458]
x2 = [ 0.042,0.142,0.242,0.342,0.442]
x =  [ 0.026,0.126,0.226,0.326,0.426]
#,'Econ-Mahindas','BN']
for name in ['Email_univ', 'Econ-Mahindas', 'BN']:
    if(name == 'Email_univ'):
        y = [0.9050, 0.7832, 0.6729, 0.5394, 0.3466]
        # z = [0.9682,0.9487,0.9098,0.8304,0.6281]
        z = [0.8929,0.7633,0.6685,0.4928,0.2848]
        t = [0.8265,0.7793,0.6985,0.5871,0.4877]
        l = [0.8067,0.7049,0.6048,0.4831,0.3481] 
        # l = [0.8310	,0.7384	,0.6729	,0.5533,0.4730]
    if(name == 'Econ-Mahindas'):
        y = [0.9027, 0.7756, 0.6773, 0.5303, 0.4121]
        z = [0.9936,0.9921,0.9587,0.8704,0.6010] 
        t = [0.7742,0.6988,0.5994,0.5233,0.4515] 
        l = [0.7838,0.6787,0.5234,0.4695,0.3248]
        # l = [0.5825	,0.5137,0.4819	,0.4025	,0.3451]
    if(name == 'BN'):
        y = [0.8509, 0.7182, 0.5730, 0.4663, 0.3256]
        # z = [0.9336,0.8684,0.7702,0.7144,0.5414] 
        z = [0.8551,0.7154,0.5646,0.4396,0.2757]
        t = [0.8663,0.7618,0.6685,0.5975,0.4834] 
        l = [0.6725,0.5872,0.4757,0.3527,0.2669]
    # l = [0.4494	,0.4291	,0.3633	,0.3140	,0.2444]
# if name == 'PPI':
#     y = [0.9314,	0.8639,	0.7950,	0.7144,	0.6281]
#     z = [0.4886,	0.3680,	0.2937,	0.2355,	0.1936]
#     t = [0.7869,	0.7376,	0.7060,	0.6322,	0.5740]
#     l = [0.0332,	0.02,	0.02,	0.01,	0.01]
    matplotlib.rcParams.update({'font.size': 20})
    ax = plt.subplot(111)
    b1 = ax.bar(x, y,width=0.016,color='black', align='center',  hatch="",edgecolor='#000000')
    b2 = ax.bar(x2, z,width=0.016,color = 'none', align='center',  hatch="///",edgecolor='#000000')
    b3 = ax.bar(x3, t,width=0.016,color = 'none', align='center',  hatch="...",edgecolor='#000000')
    b4 = ax.bar(x4, l,width=0.016,color = 'none', align='center',  hatch="",edgecolor='#000000')
    ax.set_xlabel('Attribute noise ratio',fontsize=22)
    ax.set_ylabel('Success@1',fontsize=22)
    xticks([ 0.05,0.15,0.25,0.35,0.45], ["0.1", "0.2", "0.3", "0.4", "0.5"], fontsize = 22);
    #ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))
    xlim(-0.05,0.55);
    yticks(np.arange(0,1.1,0.2),fontsize=22);
    ylim(0,1.15)

    plt.gca().yaxis.grid(True)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height*0.8])
    ax.legend( (b1, b2, b3,b4), ('NAME', 'GAlign', 'REGAL', 'FINAL'), fontsize=14.5, frameon=False, labelspacing=0, borderpad = 0.01,
            columnspacing= 0.45, handletextpad=0.3, ncol=4, loc='upper left')

    dir = "./chart_output/"
    savefig(dir + "ch_feats_{}.png".format(name), bbox_inches='tight')
    close()
