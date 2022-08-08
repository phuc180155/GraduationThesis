import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import pdb
import math
import argparse

# parser = argparse.ArgumentParser(description="Network alignment")
# parser.add_argument('--language', default="zh_en")
# param = parser.parse_args()

def mjrFormatter(x, pos):
    if x==1:
        return '1'
    elif x==0:
        return '0'
    else:
        return ("%.1f" % x).lstrip('0')

#matplotlib.rcParams.update({'font.size': 20})
#matplotlib.rcParams['text.usetex'] = True
    

# x7 = [1.3,2.3,3.3]
x6 = [ 1.25,2.25,3.25]
x5 = [ 1.15,2.15,3.15]
x4 = [1.05,2.05,3.05]
x3 = [ 0.95,1.95,2.95]
x2 = [ 0.85,1.85,2.85]
x =  [ 0.75,1.75,2.75]

#if param.language == 'all':
y1 = [652,5647,20959] 
y2 = [26,336,62] 
y3 = [68,1680,5228] 
y4 = [14,76,422]
y5 = [25,323,1876]
y6 = [198,353,2624]
# y7 = [3689,3538,4195]

# if param.language == 'zh_en':
#     y1 = [1175,1175,1175] 
#     y2 = [30748,30264,31523] 
#     y3 = [11484,11040,10569] 
#     y4 = [128,91,81]
#     y5 = [5932,7433,8291]
#     y6 = [8601,6788,6027]
#     y7 = [3689,3838,4295]
# if param.language == 'ja_en':
#     y1 = [1185,1185,1185] 
#     y2 = [30929,30472,31245] 
#     y3 = [11640,11027,10504] 
#     y4 = [127,92,80]
#     y5 = [6596,8152,9033]
#     y6 = [7527,5931,5422]
#     y7 = [3538,3782,4174]
# if param.language == 'fr_en':
#     y1 = [1196,1196,1196] 
#     y2 = [30523,30726,30217] 
#     y3 = [12879,12284,11629] 
#     y4 = [142,102,87]
#     y5 = [7555,8379,9420]
#     y6 = [7825,6012,5523]
#     y7 = [4195,4482,4902]

ax = plt.subplot(111)
b1 = ax.bar(x, y1,width=0.1,color='black', align='center',  hatch="",edgecolor='#000000')
b2 = ax.bar(x2, y2,width=0.1,color = 'none', align='center',  hatch="///",edgecolor='#000000')
b3 = ax.bar(x3, y3,width=0.1,color = 'none', align='center',  hatch="...",edgecolor='#000000')
b4 = ax.bar(x4, y4,width=0.1,color = 'none', align='center',  hatch="\\",edgecolor='#000000')
b5 = ax.bar(x5, y5,width=0.1,color = 'none', align='center',  hatch="--",edgecolor='#000000')
b6 = ax.bar(x6, y6,width=0.1,color = 'none', align='center',  hatch="",edgecolor='#000000')
# b7 = ax.bar(x7, y7,width=0.1,color = 'none', align='center',  hatch="++",edgecolor='#000000')

ax.set_ylabel('Time (sec)',fontsize=22)
xticks([ 1,2,3], ["Douban", "Allmovie-Imdb", "PPI"], fontsize = 18)
xlim(0.5,3.5)
# yticks(fontsize=22);
ax.set_yscale('log')
yticks([1,10,100,1000,10000,100000],fontsize=18)
ylim(0,250000)

plt.gca().yaxis.grid(True)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height*0.8])
ax.legend( (b1, b2, b3,b4,b5,b6), ('NAME', 'GAlign', 'PALE', 'REGAL', 'IsoRank', 'FINAL'), fontsize=14, frameon=False, labelspacing=0, borderpad = 0.01,
           columnspacing= 0.45, handletextpad=0.3, ncol=4, loc='upper left')
dir = "./chart_output/"
savefig(dir + "Time.png", bbox_inches='tight')
