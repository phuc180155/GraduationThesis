import matplotlib.pyplot as plt  
import numpy as np 
from pylab import *

with open("data/time_nawal.txt", 'r', encoding='utf-8') as file:
    data = []
    for line in file:
        data_line = line.split()
        data.append([float(ele) for ele in data_line])
    # data = np.array(data)

# print(data)

models = ['NAWAL', 'UAGA', 'IsoRank', 'FINAL', 'REGAL', 'IONE', 'PALE', 'DeepLink', 'NAWAL']
xtick = ['1k', '5k', '10k', '20k', '100k', '1M']

x0 = [ele + 0.65 for ele in list(range(6))] 
x1 = [ele + 0.75 for ele in list(range(6))] 
x2 = [ele + 0.85 for ele in list(range(6))] 
x3 = [ele + 0.95 for ele in list(range(6))] 
x4 = [ele + 1.05 for ele in list(range(6))] 
x5 = [ele + 1.15 for ele in list(range(6))] 
x6 = [ele + 1.25 for ele in list(range(6))] 
x7 = [ele + 1.35 for ele in list(range(6))] 

y0 = data[0]
y1 = data[1]
y2 = data[2]
y3 = data[3]
y4 = data[4]
y5 = data[5]
y6 = data[6]
y7 = data[7]

ax = plt.subplot(111)
b0 = ax.bar(x0, y0,width=0.1,color= 'black', align='center',  hatch="",edgecolor='#000000')
b1 = ax.bar(x1, y1,width=0.1,color= 'none', align='center',  hatch=".",edgecolor='#000000')
b2 = ax.bar(x2, y2,width=0.1,color = 'none', align='center',  hatch="///",edgecolor='#000000')
b3 = ax.bar(x3, y3,width=0.1,color = 'none', align='center',  hatch="...",edgecolor='#000000')
b4 = ax.bar(x4, y4,width=0.1,color = 'none', align='center',  hatch="\\",edgecolor='#000000')
b5 = ax.bar(x5, y5,width=0.1,color = 'none', align='center',  hatch="--",edgecolor='#000000')
b6 = ax.bar(x6, y6,width=0.1,color = 'none', align='center',  hatch="",edgecolor='#000000')
b7 = ax.bar(x7, y7,width=0.1,color = 'none', align='center',  hatch="o",edgecolor='#000000')


ax.set_ylabel('Time (sec)',fontsize=35)
ax.set_xlabel('Number of nodes',fontsize=35)
xticks([ele + 1 for ele in list(range(6))], xtick, fontsize = 30)
# xlim(0.5,3.5)
yticks(fontsize=30)

ax.set_yscale('log')
# ax.set_yticks([0, 10, 1e2, 1e3, 1e4, 1e5, 1e6])
# yticks([1,10,100,1000,10000,100000],fontsize=18)
# ylim(0,250000)
# ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.gca().yaxis.grid(True)

box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width , box.height*0.8])
ax.legend((b0, b1, b2, b3, b4, b5, b6, b7), models, fontsize=30, frameon=False, labelspacing=0, borderpad = 0.01,
           columnspacing= 0.45, handletextpad=0.3, ncol=4, loc='upper left')
dir = "./chart_output/"
# savefig(dir + "Time_nawal.png", bbox_inches='tight')
plt.show()

