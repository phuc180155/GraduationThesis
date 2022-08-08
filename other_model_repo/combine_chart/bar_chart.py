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



fake_gen_techs = ["3DMM",	"DeepFake",	"FaceSwap 2D",	"FaceSwap 3D",	"MonkeyNet", "ReenactGAN",	"StarGAN", "X2Face"]
# detech_techs = ["Meso4", "Capsule", "XceptionNet", "GAN-fingerprint", "Spectrum1D", "HPBD", "Visual-Artifacts"]

detech_techs = ["Spectrum", "Visual Artifact", "MesoNet4", "CapsuleNet", "XceptionNet", "Efficient Frequency", "Proposal"]

data_path = 'data/deepfake_chart.csv'

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


# print(new_data)

categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
hatches = ["o", "-", "/", "\\",  "//", ".",  "+", "x", "*"]
# colors = ["#FBEEE6", "#EDBB99", "#DC7633", "#D35400", "#BA4A00", "#A04000", "#6E2C00"]
# colors = ["#FEF9E7", "#F9E79F", "#F7DC6F", "#F4D03F", "#F1C40F", "#B7950B", "#7D6608"]
colors = ["#fc5a03", "#fcdf03", "#9dfc03", "#03fc41", "#03fceb", "#2403fc", "#f003fc"]
# colors2 = ["#E9F7EF", "#A9DFBF", "#52BE80", "#27AE60", "#1E8449", "#196F3D", "#145A32"]
# colors = ["#00FF00", "#F0FF00", "#00FFFD", "#FFAD00", "#FF001B", "#8800FF", "#FDFEFE"]

for i in range(len(categories)):
    current_cate = categories[i]
    current_data = new_data[i]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(fake_gen_techs))  # the label locations
    # import pdb
    # pdb.set_trace()
    width = 0.08  # the width of the bars

    for j in range(len(current_data)):
        lol = x - (4 - j) * 0.1
        ax.bar(lol, current_data[j], width, label=detech_techs[j], hatch=hatches[j], edgecolor='k', color=colors[j])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(current_cate, fontweight='bold', fontsize=17)
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    # ax.set_yticks(np.arange(0, 1.2, 0.2), fontsize=10)
    ax.set_xticklabels(fake_gen_techs, fontsize=13, fontweight='bold')
    plt.setp(ax.get_xticklabels(),ha="center", rotation=30)

    ax.legend(ncol=4, fontsize=14)
    ax.set_ylim(0, 1.5)
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    # plt.legend(ncol = 7)
    plt.yticks(fontsize=13)
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

    fig.tight_layout()
    plt.grid()
    plt.savefig('chart_output/{}.png'.format(current_cate))
    # plt.show()
    # exit()
    # plt.close()
    # exit()