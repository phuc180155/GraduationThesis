import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from pylab import *
import json
import os
import pdb
from pylab import *

def get_data(file_path):
    text = open(file_path,'r').read().split()
    return np.asarray(text, dtype=np.float64).reshape(3,5)

def line_chart(models, data_matrix, x_label, y_label, title, xpoints, higher_models = [], name=None, maxx=1.1):
    
    barwith = 0.1
    #ax1 = plt.subplot(111)
    ax1 = plt.subplots(figsize=(12, 7))[1]
    #pdb.set_trace()
    num_models = data_matrix.shape[0]
    num_x_levels = data_matrix.shape[1]
        
    assert num_models == len(models), "Number of model must equal to data matrix shape 0"


    lns1 = []
    for i, model in enumerate(models):
        if model not in higher_models:
            line = data_matrix[i]
            x = np.arange(num_x_levels)
            
            if i > 3:
                fillstyle = 'full'
            lni, = ax1.plot(x, line, marker='o', markersize=14, label=models[i], markevery=1,linewidth=4,fillstyle = 'full')
            print('ok')
            lns1.append(lni)

    ax1.set_xlabel(x_label, fontsize = 30)
    ax1.set_ylabel(y_label, fontsize = 30)

    plt.xticks(np.arange(len(xpoints)), xpoints, fontsize = 30)
    plt.yticks(np.arange(0, 1.09, step=0.2), fontsize = 30)
    ax1.set_xlim(-0.3, len(xpoints) + .3 - 1)
    ax1.set_ylim(-0.05, maxx )
    # ypoints = np.arange(0, 1.1, 0.2)
    # plt.yticks(np.arange(len(ypoints)), ypoints, fontsize = 16)

    plt.gca().yaxis.grid(True,linewidth=1)
    box = ax1.get_position()
    ax1.set_position([box.x0+0.02, box.y0+0.04, box.width, box.height])
    plt.legend(ncol = 3,fontsize = 25,loc = 4)
    plt.savefig(name)
    plt.close()




if __name__ == "__main__":
    models = ['Allmovie-Imdb','Douban']
    # modes = ['del_edges', 'del_nodes']
    # datasets = ['BN','Econ-mahindas','Email-univ']
    # xpoint_del_nodes = {"xp": ["0.1", "0.2", "0.3", "0.4", "0.5"], "xlabel": "Nodes removal ratio"}
    # xpoint_del_edges = {"xp": ["0.1", "0.2", "0.3", "0.4", "0.5"], "xlabel": "Edges removal ratio"}
    # xpoints = {'del_edges': xpoint_del_edges, 'del_nodes': xpoint_del_nodes}
    ylabel = "Success@1"
    maxx = 1.05
    data = np.array([[[0.8737,0.8773,0.8763,0.8709,0.8670],
            [0.8703,0.8773,0.8645,0.8543,0.8317],
            [0.8558,0.8773,0.8725,0.8633,0.8527],
            [0.8717,0.8785,0.8749,0.8655,0.8671]],
            [[0.5202,0.5251,0.4827,0.4816,0.4771],
            [0.4804,0.5285,0.5274,0.5117,0.5106],
            [0.4637,0.4793,0.5256,0.5050,0.5184],
            [0.5095,0.5173,0.5129,0.5006,0.5061]]])
    file_names = ['Negative sample size', 'Number of GCN layer', 'GCN Embedding  dimension', 'Global  community-aware  embedding  dimension']
    for i, f in enumerate(file_names):
        accs = np.asarray(data[:,i]).reshape(2,5)
        if f == 'Negative sample size':
            xticks = ['5', '10', '15', '20', '25']
        elif f == 'Number of GCN layer':
            xticks = ['1', '2', '3', '4', '5']
        elif f == 'GCN Embedding  dimension':
            xticks = ['50', '100', '200',' 300', '500']
        else:
            xticks = ['5', '10', '15', '20', '25']
        xlabel = 'Embedding dimention'
        print(accs)
        line_chart(models = models, data_matrix=accs, \
            x_label=f, y_label=ylabel, title='name', xpoints=xticks, higher_models=[], name='chart_output/{}.png'.format('_'.join(f.split())), maxx=maxx)
