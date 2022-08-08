import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def read_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.split()
            data_line = [float(ele) for ele in data_line]
            data.append(data_line)
    return data 



# fake_gen_techs = ["FaceSwap-2D",	"FaceSwap-3D",	"3DMM",	"Deepfake",	"StarGAN",	"ReenactGAN",	"MonkeyNet"]
# detech_techs = ["FaceSwap-2D",	"FaceSwap-3D",	"3DMM",	"Deepfake",	"StarGAN",	"ReenactGAN",	"MonkeyNet"]
fake_gen_techs = ["Meso4", "Capsule", "XceptionNet", "GAN-fingerprint", "Spectrum1D"]


# detech_techs = ["Meso4", "Capsule", "XceptionNet", "GAN-fingerprint", "FDBD", "HPBD", "Visual-Artifacts"]
detech_techs = ["0.5", "0.75", "1", "1.5", "2"]
# fake_gen_techs = ["0.5", "0.75", "1", "1.5", "2"]

data_path = 'data/g_data'

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
colorss = {0: '#000000',1: '#003b7f',2: '#0076ff',3: '#7fbaff',4: '#ffffff'}
# print(new_data)

categories = ['Deepfake', '3DMM', 'FaceSwap-2D', 'FaceSwap-3D','MonkeyNet','ReenactGAN','StarGAN']
# hatches = ["//", "o", "\\\\", "O", ".", "*", "xx"]
fig = plt.figure(figsize=(17,5))
for i in range(len(categories)):
    current_cate = categories[i]
    current_data = new_data[i]
    # fig, ax = plt.subplots(figsize=(5, 8))
    ax = plt.subplot(1, 7, i + 1)
    x = np.arange(len(fake_gen_techs))  # the label locations
    # import pdb
    # pdb.set_trace()
    width = 0.09  # the width of the bars

    for j in range(len(current_data)):
        lol = x - (3 - j) * 0.11
        print(j)
        print(colorss[j])
        ax.barh(lol, current_data[j], width, label=detech_techs[j], edgecolor='k',color = colorss[j])


    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax.set_xlabel(current_cate, fontweight='bold', fontsize=18)
    # ax.set_title('Scores by group and gender')
    if i ==0:
        ax.set_yticks(x)
        # ax.set_yticks(np.arange(0, 1.2, 0.2), fontsize=10)
        ax.set_yticklabels(fake_gen_techs, fontsize=11, fontweight='bold')
    else:
        ax.get_yaxis().set_visible(False)
    # ax.legend(ncol=4, fontsize=14)
    ax.set_xlim(0, 1.25)
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    # plt.legend(ncol = 7)
    plt.xticks(fontsize=13)
    # def autolabel(rects):
    #     """Attach a text label above each bar in *rects*, displaying its height."""
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate('{}'.format(height),
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(0, 3),  # 3 points vertical offset
    #                     textcoords="offset points",
    #                     ha='center', va='bottom')


    # autolabel(rects1)
    # autolabel(rects2)

    # fig.tight_layout()
    plt.grid()
plt.savefig('chart_output/{}.png'.format("test"))
# plt.show()
# exit()
# plt.close()
# exit()