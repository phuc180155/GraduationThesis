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


def f1(p, r):
    return 2 * p * r / (p + r)


def get_info(typee):
    path = "LBSNdata/abl_friend_{}.txt".format(typee)
    ab_type = typee.split("_")[-1]
    print(typee)
    styles = ['o', 's', 'd', '^','x', '*', '1', 'P', '>', '<', '.', '+']
    file = open(path, 'r')
    data_precision = []
    data_recall = []
    for line in file:
        # print(line.split())
        data_line = line.split()
        for ele in data_line:
            try:
                data_precision.append(float(ele.split("|")[0]))
            except:
                # print(line)
                data_precision.append(float(ele.split("|")[0][1:]))
            try:
                data_recall.append(float(ele.split("|")[1]))
            except:
                # print(line)
                data_recall.append(float(ele.split("|")[1][:-1]))
    
    x_labels = x_labels_all[ab_type]
    data_precision = np.array(data_precision).reshape((-1, 5)).T
    data_recall = np.array(data_recall).reshape((-1, 5)).T
    # data_recall = np.a
    mmm = ["Precision 10", "Precision 20", "Precision 30", "Precision 40", "Precision 50"]
    mmm2 = ["Recall 10", "Recall 20", "Recall 30", "Recall 40", "Recall 50"]
    print(data_precision)

    plt.figure(figsize=(5,4), constrained_layout=True)

    for i in range(len(data_precision)):
        plt.plot(data_precision[i], label=mmm[i], marker=styles[i])
        # plt.plot(data_recall[i], label=mmm2[i], marker=styles[i])
    # plt.show()
    smt = smts[ab_type]

    plt.grid()
    # plt.text(3.40, np.max(this_precision), s=dataset, fontweight='bold', fontsize=35)
    plt.legend(fontsize=10, framealpha=1)
    plt.xticks(np.arange(len(x_labels)), x_labels, fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')
    plt.xlabel(smt, fontsize=10, fontweight='bold')
    # plt.ylabel(mode, fontsize=30, fontweight='bold')

    # h.set_rotation(0)
    # plt.ylim(-0.001, np.max(this_precision) + np.max(this_precision) / 10)
    plt.savefig('LBSNout/alb{}_precision.png'.format(typee))
    plt.close()
    # plt.show()




    plt.figure(figsize=(5,4), constrained_layout=True)

    for i in range(len(data_precision)):
        # plt.plot(data_precision[i], label=mmm[i], marker=styles[i])
        plt.plot(data_recall[i], label=mmm2[i], marker=styles[i])
    # plt.show()


    plt.grid()
    # plt.text(3.40, np.max(this_precision), s=dataset, fontweight='bold', fontsize=35)
    plt.legend(fontsize=10, framealpha=1)
    plt.xticks(np.arange(len(x_labels)), x_labels, fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')
    plt.xlabel(smt, fontsize=10, fontweight='bold')
    # plt.ylabel(mode, fontsize=30, fontweight='bold')

    # h.set_rotation(0)
    # plt.ylim(-0.001, np.max(this_precision) + np.max(this_precision) / 10)
    plt.savefig('LBSNout/alb{}_recall.png'.format(typee))
    plt.close()



def get_info2(path):
    models = ['OUR', 'DeepWalk S', 'DeepWalk M', 'DeepWalk SM', 'Node2Vec S', 'Node2Vec M', 'Node2Vec SM', 'Line S', 'Line M', "Line SM", "DHNE M", "LBSN2Vec"]
    datas = ["IST", "SP", "KL", "TKY", "USA", "NYC", "JK"]
    x_labels = ["10", "20", "50", "100", "200"]
    styles = ['o', 's', 'd', '^','x', '*', '1', 'P', '>', '<', '.', '+']
    file = open(path, 'r')
    cur_index = 0
    data = dict()

    for line in file:
        if len(line.split()) < 3:
            cur_index += 1
            continue
        print(line)
        print(models[cur_index])
        if models[cur_index] not in data:
            data[models[cur_index]] = {'precision':[], 'recall':[], 'f1':[]}
        data_line = line.strip().split()
        this_precision = []
        this_recall = []
        this_f1 = []
        for ele in data_line:
            precision = float(ele.split("|")[0])
            recall = float(ele.split("|")[1])
            f1_score = f1(precision, recall)
            this_precision.append(precision)
            this_recall.append(recall)
            this_f1.append(f1_score)
        data[models[cur_index]]['precision'].append(this_precision)
        data[models[cur_index]]['recall'].append(this_recall)
        data[models[cur_index]]['f1'].append(this_f1)
    
    # new_data = {'precision':[], 'recall': [], 'f1': []}
    new_data = dict()

    for index in range(len(datas)):
        cur = datas[index]
        this_info = {'precision': [], 'recall': [], 'f1': []}
        for j in range(len(models)):
            model_j = models[j]
            model_j_precision = data[model_j]['precision'][index]
            model_j_recall = data[model_j]['recall'][index]
            model_j_f1 = data[model_j]['f1'][index]

            this_info['precision'].append(model_j_precision)
            this_info['recall'].append(model_j_recall)
            this_info['f1'].append(model_j_f1)
        new_data[cur] = this_info


    def draw(mode, this_precision):
        # plt.figure(figsize=(6.5,5), constrained_layout=True)
        print(len(this_precision), len(models), len(styles))
        for j in range(len(this_precision)):
            plt.plot(this_precision[j], label=models[j], marker=styles[j])
        plt.grid()
        plt.text(3.40, np.max(this_precision), s=dataset, fontweight='bold', fontsize=35)
        plt.legend(fontsize=20, framealpha=1)
        plt.xticks(np.arange(len(x_labels)), x_labels, fontsize=20, fontweight='bold')
        plt.yticks(fontsize=20, fontweight='bold')
        plt.xlabel('K', fontsize=30, fontweight='bold')
        plt.ylabel(mode, fontsize=30, fontweight='bold')

        # h.set_rotation(0)
        plt.ylim(-0.001, np.max(this_precision) + np.max(this_precision) / 10)
        plt.savefig('LBSNout/{}_{}.png'.format(dataset, mode))
        plt.show()
        exit()
        # plt.close()


    for dataset in new_data:
        this_precision = new_data[dataset]['precision']
        # print(this_precision)
        for i in range(len(this_precision)):
            for j in range(4):
                if this_precision[i][j] < this_precision[i][j+1]:
                    print(dataset, models[i])
        this_recall = new_data[dataset]['recall']
        for i in range(len(this_precision)):
            for j in range(4):
                if this_recall[i][j] > this_recall[i][j+1]:
                    print(dataset, models[i])
        this_f1 = new_data[dataset]['f1']
        draw('Precision@K', this_precision)
        draw('Recall@K', this_recall)
        draw('F1@K', this_f1)
        

    return new_data


if __name__ == "__main__":
    x_labels_p = ["0.2", "0.4", "0.6", "0.8", "1"]
    x_labels_q = x_labels_p
    x_labels_wl = ["10", "20", "50", "80", "100"]
    x_labels_ed = ["32", "64", "128", "256", "512"]
    x_labels_mr = ["0.1", "0.3", "0.5", "0.7", "0.9"]
    x_label_nw = ["2", "5", "7", "10", "15", "20"]

    x_labels_all = {"p": x_labels_p, "q": x_labels_q, "wl": x_labels_wl, "ed": x_labels_ed, "mr": x_labels_mr, "nw": x_label_nw}
    # data = get_info('LBSNdata/lbsn.txt')
    smts = {"p": "p", "q": "q", "wl": "Walk length", "ed": "Embedding size", "mr": "Mobility ratio", "nw": "Number of walks"}
    # for t in [""]
    for key in x_labels_all.keys():
        get_info("hongzhi_{}".format(key))
    print("DONE!")

# def line_chart(models, data_matrix, x_label, y_label, title, xpoints, higher_models = [], name=None, maxx=1.2):
#     styles = ['o', 's', 'd', '^','x', '*']
#     #styles = ['o', 's', 'v', '*']
#     line_styles = ['-', '--', '-', '-.', ':','-','--']
#     # styles = ['o-', '>--', 's-', 'd-.', '^:', 'x-', '*-', 'v-']
    
#     colors = ['#003300', '#009933', '#33cc33', '#66ff66', '#99ff99', '#ffffff','#0033ff']
#     barwith = 0.1
#     ax1 = plt.subplot(111)

#     num_models = data_matrix.shape[0]
#     num_x_levels = data_matrix.shape[1]\
        
#     assert num_models == len(models), "Number of model must equal to data matrix shape 0"


#     lns1 = []
#     for i, model in enumerate(models):
#         if model not in higher_models:
#             line = data_matrix[i]
#             x = np.arange(num_x_levels)
#             fillstyle = 'none'
#             if i > 3:
#                 fillstyle = 'full'
#             lni, = ax1.plot(x, line, marker=styles[i % len(styles)], markersize=10, color='k', label=models[i], markevery=1, fillstyle=fillstyle)
#             lns1.append(lni)

#     ax1.set_xlabel(x_label, fontsize = 25,fontweight='bold')
#     ax1.set_ylabel(y_label, fontsize = 25,fontweight='bold')
#     ax2 = None
#     lns2 = []
#     count = 0

#     for i, model in enumerate(models):
#         if model in higher_models:
#             line = data_matrix[i]
#             x = np.arange(num_x_levels)
#             if ax2 is None:
#                 ax2 = ax1.twinx()
#             ln2 = ax2.bar(x + count * barwith, line, width=barwith, color = colors[count],  edgecolor='k', label=model, alpha = 0.5)
#             lns2.append(ln2)
#             count += 1


#     plt.xticks(np.arange(len(xpoints)), xpoints, fontsize = 20)
#     plt.yticks(np.arange(0, 1.1, step=0.2), fontsize = 20)
#     ax1.set_xlim(-0.3, len(xpoints) + .3 - 1)
#     # ax1.set_ylim(-0.05, maxx + .32)
#     ax1.set_ylim(-0.05, maxx + .2)
#     # ypoints = np.arange(0, 1.1, 0.2)
#     # plt.yticks(np.arange(len(ypoints)), ypoints, fontsize = 16)

#     if ax2 is not None:
#         ax2.set_xlim(-0.5, len(xpoints) + .5 - 1)
#         ax2.set_ylim(0, 0.7)
#         ax2.set_yticks(np.arange(0, 0.6, 0.1))
#         ax2.tick_params('y', colors='green')

#     ax1.grid(True)
#     box = ax1.get_position()
#     ax1.set_position([box.x0 + 0.02, box.y0 + 0.04, box.width, box.height])

#     # plt.legend(ncol = 4,borderaxespad = 0.3, fontsize=10.7)
#     # plt.legend(ncol = 4, fontsize = 11.5, loc = 1, columnspacing = 2.3)
#     plt.legend(ncol = 4, fontsize = 12.2, loc = 1, columnspacing = 0.5)
#     plt.savefig(name)
#     plt.close()




# if __name__ == "__main__":
#     models = ['NAWAL-Refine','NAWAL','PALE','DeepLink','FINAL','REGAL','IsoRank','IONE','UAGA']
#     new_models = models[1:]
#     modes = ['del_edges', 'del_nodes']
#     datasets = ['facebook','foursquare','twitter']
#     xpoint_del_nodes = {"xp": ["0.0","0.1", "0.2", "0.3", "0.4"], "xlabel": "Nodes removal ratio"}
#     xpoint_del_edges = {"xp": ["0.0","0.1", "0.2", "0.3", "0.4"], "xlabel": "Edges removal ratio"}
#     xpoints = {'del_edges': xpoint_del_edges, 'del_nodes': xpoint_del_nodes}
#     ylabel = "Accuracy"
#     maxx = 1.05
#     print(new_models)
#     for dataset in datasets:
#         print(dataset)
#         for mode in modes[1:]:
#             print(mode)
#             file_name = "data/{}_{}.txt".format(mode, dataset)
#             name = mode + '-' + dataset
#             accs = get_info_special_text(file_name, "Accuracy")
#             MAP = get_info_special_text(file_name, "MAP")
#             top5 = get_info_special_text(file_name, "Precision_5")
#             top10 = get_info_special_text(file_name, "Precision_10")
#             # import pdb
#             # pdb.set_trace()
#             # top10 = get_info_special_text(file_name, "Precision_10")
#             accs = np.delete(accs, 1, 0)
#             MAP = np.delete(MAP, 1, 0)
#             top5 = np.delete(top5, 1, 0)
#             top10 = np.delete(top10, 1, 0)
#             # accs = np.delete(accs, 1, 0)
#             # accs = np.delete(accs, 1, 0)
#             accs_mean = np.mean(accs, axis=1)*100
#             MAP_mean = np.mean(MAP, axis=1)*100
#             top5_mean = np.mean(top5, axis=1)*100
#             top10_mean = np.mean(top10, axis=1)*100

#             for i in range(len(accs_mean)):
#                 print(new_models[i])
#                 print("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(accs_mean[i], MAP_mean[i], top5_mean[i], top10_mean[i]))
#             # import pdb
#             # pdb.set_trace()
#             #accs = accs.reshape((len(models), -1))
#             #accs = np.delete(accs, 1, axis=0)
#             xticks = xpoints[mode]["xp"]
#             xlabel = xpoints[mode]["xlabel"]
#             # line_chart(models = new_models, data_matrix=accs, \
#             #     x_label=xlabel, y_label=ylabel, title=name, xpoints=xticks, higher_models=[], name='chart_output/{}.png'.format(name), maxx=maxx)
            

#             #line_chart_old(data=accs, xpoints=xticks, xtitle=xlabel, ytitle=ylabel, filename='{}.png'.format(name), models=models, yticks=None, add_Legend=True)
