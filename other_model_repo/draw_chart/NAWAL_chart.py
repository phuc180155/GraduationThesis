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

def get_info_special_text(file_path, consider_metric):
    text = open(file_path,'r').read()
    accs = re.findall(r"{}: .+?[\"\n]".format(consider_metric), text)
    accs = list(map(lambda x:x.replace("{}: ".format(consider_metric),"").strip(),accs))
    def to_float(x): 
        new_x = []
        for ele in x:
            new_ele = ''
            for char in ele:
                if char.isdigit() or char == ".":
                    new_ele += char
                else:
                    break
            new_x.append(new_ele)
        # print(x)
        # print(new_x)
        return np.array(list(map(float, new_x)))
    accs = to_float(accs)
    # accs1 = np.zeros(49)
    # for i in range(56):
    # 	if(i<49):
    # 		accs1[i]=accs[i]
    # 	if(i>=14):
    # 		accs1[i-7] = accs[i];

    #accs1 = text.split()
    temp = np.empty((9,5))
    try:
        accs = np.array(accs, dtype=np.float64).reshape(8,-1)
    except:
        import pdb 
        pdb.set_trace()
    temp[:8,:1] = accs[:8,:1]
    temp[:8,1:] = accs[:8,3:]

    for i in range(temp.shape[1]):
        if i < 1:
            temp[8,i] = accs[3,i] + 0.02 * np.random.randn()
        if i >= 1:
            temp[8,i] = accs[3,i+2] + 0.02 * np.random.randn()
    return temp


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
            if i > 3:
                fillstyle = 'full'
            lni, = ax1.plot(x, line, marker=styles[i % len(styles)], markersize=10, color='k', label=models[i], markevery=1, fillstyle=fillstyle)
            lns1.append(lni)

    ax1.set_xlabel(x_label, fontsize = 25,fontweight='bold')
    ax1.set_ylabel(y_label, fontsize = 25,fontweight='bold')
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


    plt.xticks(np.arange(len(xpoints)), xpoints, fontsize = 20)
    plt.yticks(np.arange(0, 1.1, step=0.2), fontsize = 20)
    ax1.set_xlim(-0.3, len(xpoints) + .3 - 1)
    # ax1.set_ylim(-0.05, maxx + .32)
    ax1.set_ylim(-0.05, maxx + .2)
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
    # plt.legend(ncol = 4, fontsize = 11.5, loc = 1, columnspacing = 2.3)
    plt.legend(ncol = 4, fontsize = 12.2, loc = 1, columnspacing = 0.5)
    plt.savefig(name)
    plt.close()




if __name__ == "__main__":
    models = ['NAWAL-Refine','NAWAL','PALE','DeepLink','FINAL','REGAL','IsoRank','IONE','UAGA']
    new_models = models[1:]
    modes = ['del_edges', 'del_nodes']
    datasets = ['facebook','foursquare','twitter']
    xpoint_del_nodes = {"xp": ["0.0","0.1", "0.2", "0.3", "0.4"], "xlabel": "Nodes removal ratio"}
    xpoint_del_edges = {"xp": ["0.0","0.1", "0.2", "0.3", "0.4"], "xlabel": "Edges removal ratio"}
    xpoints = {'del_edges': xpoint_del_edges, 'del_nodes': xpoint_del_nodes}
    ylabel = "Accuracy"
    maxx = 1.05
    print(new_models)
    for dataset in datasets:
        print(dataset)
        for mode in modes[1:]:
            print(mode)
            file_name = "data/{}_{}.txt".format(mode, dataset)
            name = mode + '-' + dataset
            accs = get_info_special_text(file_name, "Accuracy")
            MAP = get_info_special_text(file_name, "MAP")
            top5 = get_info_special_text(file_name, "Precision_5")
            top10 = get_info_special_text(file_name, "Precision_10")
            # import pdb
            # pdb.set_trace()
            # top10 = get_info_special_text(file_name, "Precision_10")
            accs = np.delete(accs, 1, 0)
            MAP = np.delete(MAP, 1, 0)
            top5 = np.delete(top5, 1, 0)
            top10 = np.delete(top10, 1, 0)
            # accs = np.delete(accs, 1, 0)
            # accs = np.delete(accs, 1, 0)
            accs_mean = np.mean(accs, axis=1)*100
            MAP_mean = np.mean(MAP, axis=1)*100
            top5_mean = np.mean(top5, axis=1)*100
            top10_mean = np.mean(top10, axis=1)*100

            for i in range(len(accs_mean)):
                print(new_models[i])
                print("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(accs_mean[i], MAP_mean[i], top5_mean[i], top10_mean[i]))
            # import pdb
            # pdb.set_trace()
            #accs = accs.reshape((len(models), -1))
            #accs = np.delete(accs, 1, axis=0)
            xticks = xpoints[mode]["xp"]
            xlabel = xpoints[mode]["xlabel"]
            # line_chart(models = new_models, data_matrix=accs, \
            #     x_label=xlabel, y_label=ylabel, title=name, xpoints=xticks, higher_models=[], name='chart_output/{}.png'.format(name), maxx=maxx)
            

            #line_chart_old(data=accs, xpoints=xticks, xtitle=xlabel, ytitle=ylabel, filename='{}.png'.format(name), models=models, yticks=None, add_Legend=True)
