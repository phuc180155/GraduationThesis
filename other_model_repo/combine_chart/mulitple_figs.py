
import matplotlib.pyplot as plt 
import numpy as np
# import matplotlib.gridspec as gridspec


def read_info(path):
    data = []
    count = 0
    Block_LEN = 7
    with open(path, 'r', encoding='utf-8') as file:
        this_block = []
        for line in file:
            if count % Block_LEN == 0:
                this_block = []
            line = line.replace(',', '.')
            data_line = line.split()
            if 'x' in line:
                if (count + 1) % 7 == 0:
                    data.append(this_block)
                count += 1
                continue
            else:
                data_line = [float(ele) for ele in data_line]
            this_block.append(data_line)
            if (count + 1) % 7 == 0:
                data.append(this_block)
                # print(count)
            count += 1
            # print(count)
    return data

import pdb

# OK
data = read_info('data/deepbenchmark.txt')
# pdb.set_trace()


gen_techniques = ['deepfake', '3dmm', 'swap2d', 'swap3d', 'monkey', 'reenact', 'stargan']
dis_techniques = ["meso4", "capsule", "xception_torch", "gan", "1dfft", "headpose", "visual"]
noisetype = ["b", "c", "g", "size", "miss"]
noisetype_value = {"b": [0.5, 0.75, 1, 1.5, 2], "c": [0.5, 0.75, 1, 1.5, 2], "g": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3], "size": [16, 32, 64, 128, 256], "miss": [0.1, 0.2, 0.3, 0.4, 0.5]}
data_cleaned = {"deepfake": {"b": [], "c": [], "g": [], "size": [], "miss": []}, "3dmm": {"b": [], "c": [], "g": [], "size": [], "miss": []}, "swap2d": {"b": [], "c": [], "g": [], "size": [], "miss": []}, \
    "swap3d": {"b": [], "c": [], "g": [], "size": [], "miss": []}, "monkey": {"b": [], "c": [], "g": [], "size": [], "miss": []}, "reenact": {"b": [], "c": [], "g": [], "size": [], "miss": []}, "stargan": {"b": [], "c": [], "g": [], "size": [], "miss": []}}


gen_type = -1
for i in range(len(data)):
    if i % 5 == 0:
        gen_type += 1
    for j in range(len(data[i])):
        g_tech = gen_techniques[gen_type]
        n_type = noisetype[i % 5]
        data_cleaned[g_tech][n_type].append(data[i][j])
import pdb
# pdb.set_trace()
# print(data_cleaned)


# print(data)


styles = ['o', 's', 'd', '^','x', '*']
#styles = ['o', 's', 'v', '*']
line_styles = ['-', '--', '-', '-.', ':','-','--']
# styles = ['o-', '>--', 's-', 'd-.', '^:', 'x-', '*-', 'v-']

colors = ['#003300', '#009933', '#33cc33', '#66ff66', '#99ff99', '#ffffff','#0033ff']



# current_g_tech = -1
# for i in range(len(axs)):
#     if i % 3 == 0:
#         current_g_tech += 1
#     g_tech = gen_techniques[current_g_tech]
#     cur_noise = noisetype[i % 3]
#     data_to_visual = np.array(data_cleaned[g_tech][cur_noise])

#     for j, model in enumerate(dis_techniques):
#         # if model not in higher_models:
#         line = data_to_visual[j]
#         x = np.arange(5)
#         fillstyle = 'none'
#         if j > 3:
#             fillstyle = 'full'
#         axs[i].plot(x, line) #, marker=styles[j % len(styles)], markersize=10, color='k', label=model, markevery=1, fillstyle=fillstyle)
#         # lns1.append(lni)
for vinh_suhi in range(5):
    
    # fig = plt.figure(figsize=(7, 10))
    fig = plt.figure(figsize=(17, 5))

    # gs = gridspec.GridSpec(7, 3)
    # gs = fig.add_gridspec(5, 7, hspace=0.2, wspace=0)
    gs = fig.add_gridspec(1, 7, wspace=0) #, hspace=, wspace=0)
    # gs = fig.add_gridspec(3, 7) #, hspace=0, wspace=0)
    # (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15), (ax16, ax17, ax18), (ax19, ax20, ax21) = gs.subplots(sharex='col', sharey='row')
    # (ax1, ax2, ax3, ax4, ax5, ax6, ax7), (ax8, ax9, ax10, ax11, ax12, ax13, ax14), (ax15, ax16, ax17, ax18, ax19, ax20, ax21), (ax22, ax23, ax24, ax25, ax26, ax27, ax28), (ax29, ax30, ax31, ax32, ax33, ax34, ax35) = gs.subplots(sharey='row')
    ax1, ax2, ax3, ax4, ax5, ax6, ax7 = gs.subplots(sharey='row')
    # ax1, ax2, ax3, ax4, ax5 = gs.subplots(sharey='row')

    # axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24, ax25, ax26, ax27, ax28, ax29, ax30, ax31, ax32, ax33, ax34, ax35]
    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7] #, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24, ax25, ax26, ax27, ax28, ax29, ax30, ax31, ax32, ax33, ax34, ax35]
    # axs = [ax1, ax2, ax3, ax4, ax5] #, ax6, ax7] #, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24, ax25, ax26, ax27, ax28, ax29, ax30, ax31, ax32, ax33, ax34, ax35]



    current_noise = -1
    current_noise += vinh_suhi
    for i in range(7):
        if i % 7 == 0:
            current_noise += 1
        cur_noise = noisetype[current_noise]
        # print(cur_noise)
        cur_tech = gen_techniques[i % 7]
        data_to_visual = np.array(data_cleaned[cur_tech][cur_noise])

        for j, model in enumerate(dis_techniques):
            # if model not in higher_models:
            if j == len(data_to_visual):
                break
            line = data_to_visual[j]
            if cur_noise == 'size':
                line = [ele for ele in reversed(line)]
            # x = noisetype_value[cur_noise]
            x = list(range(len(noisetype_value[cur_noise])))
            fillstyle = 'none'
            if j > 3:
                fillstyle = 'full'
            try:
                axs[i].plot(x, line, label=model) #, marker=styles[j % len(styles)], markersize=10, color='k', label=model, markevery=1, fillstyle=fillstyle)
                axs[i].set_xticks(x)
                # fontdict = {'fontsize': 20}
                # print("lol")
                axs[i].set_xticklabels(noisetype_value[cur_noise], rotation=50)
                # print(noisetype_value[cur_noise])
                axs[i].tick_params(axis='both', which='major', labelsize=15)
                axs[i].set_xlim(-0.5, len(x)-0.5)
            except:
                import pdb
                pdb.set_trace()
            # lns1.append(lni)
    # plt.legend()
    # plt.xlabel('Fake images generators')
    # ax1.set_title(gen_techniques[0])

    for i in range(7):
        axs[i].set_title(gen_techniques[i], fontsize=20)

    # axs[0].set_ylabel('Brightness', fontsize=20)
    # axs[7].set_ylabel('Contrast', fontsize=20)
    # axs[14].set_ylabel('G', fontsize=20)
    # axs[21].set_ylabel('size', fontsize=20)
    # axs[28].set_ylabel('missing data', fontsize=20)
    # axs[35].set_ylabel('G', fontsize=20)

    # ax1.set_xticks((0.1, 0.3))

    # plt.legend(ncol=7, framealpha=1, fontsize=10)
    axs[0].set_ylabel('Accuracy', fontsize=20)

    # plt.show()
    plt.tight_layout()
    plt.savefig('chart_output/{}.png'.format(cur_noise))
    plt.close()
  