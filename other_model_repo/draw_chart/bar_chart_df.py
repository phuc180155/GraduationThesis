import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def read_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.split()
            data_line = [float(ele) / 100.0 for ele in data_line]
            data.append(data_line)
    return data



# fake_gen_techs = ["FaceSwap-2D",	"FaceSwap-3D",	"3DMM",	"Deepfake",	"StarGAN",	"ReenactGAN",	"MonkeyNet"]
# detech_techs = ["FaceSwap-2D",	"FaceSwap-3D",	"3DMM",	"Deepfake",	"StarGAN",	"ReenactGAN",	"MonkeyNet"]
# fake_gen_techs = ["Meso4", "Capsule", "XceptionNet", "GAN-fingerprint", "Spectrum1D", "HPBD", "Visual-Artifacts"]
# fake_gen_techs = ["Meso4", "Capsule", "XceptionNet", "GAN-fingerprint", "Spectrum1D", "HPBD", "Visual-Artifacts","Efficient-Frequency","WADD"]
fake_gen_techs = ["Spectrum", "Visual Artifact", "Meso4", "Capsulenet", "XceptionNet", "Efficient-Frequency","Proposal Model"]


# detech_techs = ["Meso4", "Capsule", "XceptionNet", "GAN-fingerprint", "FDBD", "HPBD", "Visual-Artifacts"]
detech_techs = ["0.5", "0.75", "1", "1.5", "2"]
# detech_techs = ["0.0", "0.05", "0.1", "0.15", "0.2","0.25","0.3"]
# detech_techs = ["0", "0.1", "0.2", "0.3", "0.4",'0.5']
# detech_techs = ["16","32","64","128","256"]
# detech_techs = ["50","60","70","80","90","100"]
# detech_techs = ["Meso4", "Capsule", "XceptionNet", "GAN-fingerprint", "Spectrum1D", "HPBD", "Visual-Artifacts","Efficient-Frequency","WADD"]

# detech_techs = ["256","128","64","32","16"]
# fake_gen_techs = ["0.5", "0.75", "1", "1.5", "2"]

# data_path = 'data/c_data'
# data_path = 'data/miss_data'
# data_path = 'data/resize_data'
data_path = 'data_contrast.csv'
# data_path = 'data/c_data_ne

data = read_data(data_path)

new_data = []
this_bag = []
for i in range(len(data)):
    if i % len(detech_techs) == 0:
        if len(this_bag):
            new_data.append(this_bag)
        this_bag = [data[i]]
    else:
        this_bag.append(data[i])
    if i == len(data) - 1:
        new_data.append(this_bag)

# colorss = {0: '#DDE3FD',1: '#889AF7',2: '#3250E7',3: '#162A8D',4: '#000A3D'}
# colorss = {0: '#000000',1: '#0016cc',2: '#3349ff',3: '#99a4ff',4: '#99a4ff',5:'ffffff'}
# colorss = {0: '#0014dc',1: '#1950ff',2: '#649bff',3: '#96cdff',4: '#b4ebff'}
# colorss = {0: '#0014dc',1: '#192df5',2: '#649bff',3: '#96cdff',4: '#b4ebff', 5 :'#e1f5ff',6:'#ffffff'}
# colorss = {0: '#ffffff',1: '#e1f5ff',2: '#b4ebff',3: '#96cdff',4: '#649bff', 5 :'#192df5',6:'#0014dc'}
# colorss = {0: '#e1f5ff',1: '#b4ebff',2: '#96cdff',3: '#649bff', 4 :'#192df5',5:'#0014dc'}
# colorss = {0: '#000000',1: '#004faa',2: '#55a4fe',3: '#ffffff'}
# print(new_data)
# c
colorss = {0: '#f0bdd4',1: '#ed7eb0',2: '#f24492',3: '#ff0073', 4 :'#a6034c',5:'#0014dc'}
# b
# colorss = {0: '#FFE0F9',1: '#EEB3C7',2: '#DD8695',3: '#B75A64', 4 :'#9F2E33',5:'#870101'}
# categories = ['Deepfake', '3DMM', 'FaceSwap-2D', 'FaceSwap-3D','MonkeyNet','ReenactGAN','StarGAN']
categories = ['Deepfake', '3DMM', 'FaceSwap-2D', 'FaceSwap-3D','MonkeyNet','ReenactGAN','StarGAN','X2Face']
# hatches = ["//", "o", "\\\\", "O", ".", "*", "xx"]

fig = plt.figure(figsize=(18,16))
# fig = plt.figure(figsize=(8,10))
# fig = plt.figure(figsize=(13,20))
for i in range(len(categories)):
    current_cate = categories[i]
    current_data = new_data[i]
    #
    print(current_data)
    print(i)
    ax = plt.subplot(8,1,i+1)
    # fig, ax = plt.subplots(figsize=(10, 3.5))
    x = np.arange(len(fake_gen_techs))  # the label locations
    # import pdb
    # pdb.set_trace()
    width = 0.08  # the width of the bars

    for j in range(len(current_data)):
        lol = x - (2 - j) * 0.09
        print(j)
        print(colorss[j])
        ax.bar(lol, current_data[j], width, label=detech_techs[j], edgecolor='k',color = colorss[j])


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(current_cate, fontweight='bold', fontsize=13)
    # ax.set_title('Scores by group and gender')
    # ax.set_xticks(x)
    # ax.set_yticks(np.arange(0, 1.2, 0.2), fontsize=10)
    # ax.set_xticklabels(fake_gen_techs, fontsize=18)
    # ax.legend(ncol=4, fontsize=14)
    ax.set_ylim(0, 1.25)
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    # plt.legend(ncol = 7)
    ax.get_xaxis().set_visible(False)
    plt.yticks(fontsize=12)
    # def autolabel(rects):
    #     """Attach a text label above each bar in *rects*, displaying its height."""
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate('{}'.format(height),
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(0, 3),  # 3 points vertical offset
    #                     textcoords="offset points",
    #                     ha='center', va='bottom')

    # handles, labels = ax.get_legend_handles_labels()
    # fig_legend = plt.figure(figsize=(2, 2))
    # fig_legend.show()
    # plt.legend(ncol=7)
    # autolabel(rects1)
    # autolabel(rects2)

# fig.tight_layout()
    plt.grid()
# plt.savefig('chart_output/{}.png'.format("resize_bar_py"),dpi=fig.dpi)
ax.get_xaxis().set_visible(True)
ax.set_xticks(x)
ax.set_xticklabels(fake_gen_techs, fontsize=12)

plt.subplots_adjust(hspace = 0.00002)

# handles,labels = ax.get_legend_handles_labels()
# fig_legend = plt.figure(figsize=(2,2))
# fig_legend.show()
# plt.legend(ncol = 7)
# plt.margins(x=0.1, y=0.1)

# plt.show()
# plt.draw()
# exit()
# plt.savefig('chart_output/{}.png'.format("resize_bar_py"),dpi=150, bbox_inches='tight')
plt.savefig("b_bar_thesis.png",dpi=150, bbox_inches='tight')
# plt.savefig("c_bar_thesis.png",dpi=150, bbox_inches='tight')

# plt.close()
# exit()
