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


model = ["MSC-LBSN","Line M","node2vec M","deepwalk M","DHNE","LBSN2Vec"]
# detech_techs = ["Meso4", "Capsule", "XceptionNet", "GAN-fingerprint", "Spectrum1D", "HPBD", "Visual-Artifacts"]


# dataset = ["NYC",	"hongzhi",	"TKY",	"Jakarta",	"KualaLampur",	"SaoPaulo",	"Istanbul"]
dataset = ["IST",	"SP",	"KL",	"USA",	"JK"]

data_path = 'data/time.txt'

data = read_data(data_path)

# new_data = []
# this_bag = []
# for i in range(len(data)):
#     if i % len(detech_techs) == 0:
#         if len(this_bag):
#             new_data.append(this_bag)
#         this_bag = [data[i]]
#     else:
#         this_bag.append(data[i])
#     if i == len(data) - 1:
#         new_data.append(this_bag)
# print(data)
# exit()

# print(new_data)

category = 'time'
# hatches = ["/", "o", "\\", ",", ".", "*", "//"]
# colors = ["#FBEEE6", "#EDBB99", "#DC7633", "#D35400", "#BA4A00", "#A04000", "#6E2C00"]
# colors = ["#FEF9E7", "#F9E79F", "#F7DC6F", "#F4D03F", "#F1C40F", "#B7950B", "#7D6608"]
colors = ["#ff0000", "#ffce00", "#ff8100", "#0039ff", "#00ffe5", "#ff00df", "#1dff00"]
# colors2 = ["#E9F7EF", "#A9DFBF", "#52BE80", "#27AE60", "#1E8449", "#196F3D", "#145A32"]
# colors = ["#00FF00", "#F0FF00", "#00FFFD", "#FFAD00", "#FF001B", "#8800FF", "#FDFEFE"]

current_cate = category
current_data = data
fig, ax = plt.subplots(figsize=(25, 10))
x = np.arange(len(dataset))  # the label locations
# import pdb
# pdb.set_trace()
width = 0.11  # the width of the bars

j = 0
lol = x - (3 - j) * 0.12
cmap = matplotlib.cm.get_cmap('tab20c')

# ax.bar(lol, current_data[len(current_data)-1], width, label="split time", edgecolor='k', color=cmap(0.1))
ax.bar(lol, current_data[j], width, label=model[j], edgecolor='k', color=cmap(0.1))
# ax.bar(lol, current_data[j], width, label=model[j], bottom = current_data[len(current_data)-1], edgecolor='k', color=cmap(0.9))

for j in range(1,len(current_data)):
    lol = x - (3 - j) * 0.12

    ax.bar(lol, current_data[j], width, label=model[j],  edgecolor='k', color=cmap(0.1*(j+1)))


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Time (seconds)", fontweight='bold', fontsize=35)
ax.set_xlabel("Dataet", fontweight='bold', fontsize=35)
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
# ax.set_yticks(np.arange(0, 1.2, 0.2), fontsize=10)
ax.set_xticklabels(dataset, fontsize=35 )
# plt.setp(ax.get_xticklabels(),ha="center", rotation=30)
plt.setp(ax.get_xticklabels(),ha="center",fontweight='bold', fontsize=35)

ax.legend(ncol=4, fontsize=25)
# ax.set_ylim(0, 1.25)
# ax.set_yticks(np.arange(0, 1.2, 0.2))
plt.yticks(fontsize=35,fontweight='bold')


fig.tight_layout()
plt.grid()
plt.savefig('chart_output/lbsn_time.png')
# plt.show()
# exit()
plt.close()
# exit()