import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from pylab import *
import json
import os
import pdb
import re

def get_acc(file_path):
    output = []
    with open(file_path,'r') as f:
        
        for line in f.readlines():
            k = []
            line = line.split()
            for j in line:
                k.append(float(j))
            output.append(k)
    return np.asarray(output)

def get_info_special_text(file_path):
    text = open(file_path,'r').read()
    accs = re.findall(r"Accuracy: .+?[\"\n]", text)
    accs = list(map(lambda x:x.replace("Accuracy: ","").strip(),accs))
    def to_float(x): return np.array(list(map(float, x)))
    accs = to_float(accs)
    accs1 = np.zeros(49)
    for i in range(56):
    	if(i<49):
    		accs1[i]=accs[i]
    	if(i>=14):
    		accs1[i-7] = accs[i];

    #accs1 = text.split()
    accs = np.array(accs1, dtype=np.float64).reshape(7,-1)
    return accs


def line_chart_old(data, xpoints, xtitle, ytitle, filename, models, yticks=None, add_Legend=False):
    """

    :param data: np array shape (n_models, n_xpoints)
    :param xpoints:
    :param xtitle:
    :param ytitle:
    :return:
    """

    styles = ['o-', '>--', 's-', 'd-.', '^:', 'x-', '*-', 'v-']

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111)
    for i in range(len(models)):
        ax.plot(np.arange(len(xpoints)), data[i], styles[i], label=models[i], markersize=10, fillstyle=None, mfc='none')
    # plot(x,h1, , marker="^",ls='--',label='GNE',fillstyle='none')

    plt.xticks(np.arange(len(xpoints)), xpoints)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    ax.set_ylim(0, 1.1)

    if add_Legend:
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, bbox_to_anchor=(1.,1))
        fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        print(filename)
        fig.savefig(filename, bbox_inches='tight')
    plt.close()



def line_chart(models, data_matrix, x_label, y_label, title, xpoints, higher_models = [], name=None, maxx=1.2):
    styles = ['o', 's', 'd', '^','x', '*']
    #styles = ['o', 's', 'v', '*']
    line_styles = ['-', '--', '-', '-.', ':','-','--']
    # styles = ['o-', '>--', 's-', 'd-.', '^:', 'x-', '*-', 'v-']
    
    colors = ['#003300', '#009933', '#33cc33', '#66ff66', '#99ff99', '#ffffff','#0033ff']
    barwith = 0.1
    ax1 = plt.subplot(111)

    num_models = data_matrix.shape[0]
    num_x_levels = data_matrix.shape[1]\
        
    assert num_models == len(models), "Number of model must equal to data matrix shape 0"


    lns1 = []
    for i, model in enumerate(models):
        if model not in higher_models:
            line = data_matrix[i]
            x = np.arange(num_x_levels)
            fillstyle = 'none'
            if False:
                fillstyle = 'full'
            lni, = ax1.plot(x, line, marker=styles[i % len(styles)], markersize=10, color='k', label=models[i], markevery=1, fillstyle=fillstyle, linestyle = line_styles[i])
            lns1.append(lni)

    ax1.set_xlabel(x_label, fontsize = 15,fontweight='bold')
    ax1.set_ylabel(y_label, fontsize = 15,fontweight='bold')
    ax2 = None
    lns2 = []
    count = 0

    for i, model in enumerate(models):
        if model in higher_models:
            line = data_matrix[i]
            x = np.arange(num_x_levels)
            if ax2 is None:
                ax2 = ax1.twinx()
            ln2 = ax2.bar(x + count * barwith, line, width=barwith, color = colors[count],  edgecolor='k', label=model, alpha = 0.5)
            lns2.append(ln2)
            count += 1


    plt.xticks(np.arange(len(xpoints)), xpoints, fontsize = 16)
    plt.yticks(np.arange(0, 1.1, step=0.2), fontsize = 16)
    ax1.set_xlim(-0.3, len(xpoints) + .3 - 1)
    ax1.set_ylim(-0.05, maxx + .22)
    # ypoints = np.arange(0, 1.1, 0.2)
    # plt.yticks(np.arange(len(ypoints)), ypoints, fontsize = 16)

    if ax2 is not None:
        ax2.set_xlim(-0.5, len(xpoints) + .5 - 1)
        ax2.set_ylim(0, 0.7)
        ax2.set_yticks(np.arange(0, 0.6, 0.1))
        ax2.tick_params('y', colors='green')

    ax1.grid(True)
    box = ax1.get_position()
    ax1.set_position([box.x0 + 0.02, box.y0 + 0.04, box.width, box.height])

    # plt.legend(ncol = 4,borderaxespad = 0.3, fontsize=10.7)
    plt.legend(ncol = 3, fontsize = 11.5, loc = 1, columnspacing = 5.1)
    plt.savefig(name)
    plt.close()




if __name__ == "__main__":
    models = ['NAME', 'GAlign', 'PALE', 'REGAL', 'IsoRank', 'FINAL']
    modes = ['del_edges', 'del_nodes']
    datasets = ['econ', 'bn', 'email', 'ppi']
    xpoint_del_nodes = {"xp": ["0.1", "0.2", "0.3", "0.4", '0.5'], "xlabel": "Nodes removal ratio"}
    xpoint_del_edges = {"xp": ["0.1", "0.2", "0.3", "0.4", '0.5'], "xlabel": "Edges removal ratio"}
    xpoints = {'del_edges': xpoint_del_edges, 'del_nodes': xpoint_del_nodes}
    ylabel = "Success@1"
    maxx = 1.05
    for dataset in datasets:
        for mode in modes:
            file_name = "data/COMBINE/{}_{}".format(dataset,mode)
            name = mode + '-' + dataset
            accs = get_acc(file_name)
            #accs = accs.reshape((len(models), -1))
            #accs = np.delete(accs, 1, axis=0)
            xticks = xpoints[mode]["xp"]
            xlabel = xpoints[mode]["xlabel"]
            line_chart(models = models, data_matrix=accs, \
                x_label=xlabel, y_label=ylabel, title=name, xpoints=xticks, higher_models=[], name='chart_output/{}.png'.format(name), maxx=maxx)
            

            #line_chart_old(data=accs, xpoints=xticks, xtitle=xlabel, ytitle=ylabel, filename='{}.png'.format(name), models=models, yticks=None, add_Legend=True)
