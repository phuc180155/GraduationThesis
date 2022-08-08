import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", size_scale1 = 1, size_scale2 = 1, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap

    im = ax.imshow(data, **kwargs)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]) * size_scale1)
    LOL = np.arange(data.shape[0]) * grid_size_2
    ax.set_yticks(LOL, minor=False)
    ax.set_yticks(LOL + size_scale2/2, minor=True)
    
    # print(LOL)
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels, fontweight='bold', fontsize=11)
    # ax.set_xlabel('vinhsuhi')
    # ax.set_title(title)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="center",
             rotation_mode="anchor", fontsize=12)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.grid(color="w", linestyle='-', linewidth=2, which='minor', axis='x')
    ax.grid(color="w", linestyle='-', linewidth=2, which='minor', axis='y')
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = None
    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.
    # print(threshold)
    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i * grid_size_2, valfmt(data[i, j], None), weight='bold', fontsize=11, **kw)
            texts.append(text)

    return texts






data_path = "data/finaldeepfake3"


final_data = []
count = 0
new_data = None
with open(data_path, 'r', encoding='utf-8') as file:
    for line in file:
        count += 1
        # print(count)
        # print(count)
        if (count - 1) % 7 == 0:
            if new_data is not None:
                final_data.append(new_data)
            new_data = []
        data_line = line.split()
        data_line = [float(ele) for ele in data_line]
        new_data.append(data_line)
        if count == 49:
            final_data.append(new_data)
        # print(new_data)

# print(final_data)

gen_teches = ['Deepfake', '3DMM', 'FaceSwap-2D', 'FaceSwap-3D', 'MonkeyNet', 'ReenactGAN', 'StarGAN']
# detech_teches = 
detech_techs = ["Meso4", "Capsule", "XceptionNet", "GAN-fingerprint", "FDBD", "HPBD", "Visual-Artifacts"]
# noises = ["0.5", "0.75", "1", "1.5", "2"]
noises = ["50", "60", "70", "80", "90", "100"]

for i in range(3):
    # noises += ["0.5", "0.75", "1", "1.5", "2"]
    noises += ["50", "60", "70", "80", "90", "100"]

grid_size_1 = 1
grid_size_2 = 0.6

# for i in range(len(gen_teches)):
#     gentech = gen_teches[i]
#     data_block = np.array(final_data[i])
final_data = final_data[:4]
data_block = np.concatenate([np.array(ele) for ele in final_data], axis=1)


print(data_block)
leng_1 = data_block.shape[1]
leng_2 = data_block.shape[0]

import pdb
# pdb.set_trace()

fig, ax = plt.subplots()
im, cbar = heatmap(data_block, detech_techs, noises,  ax=ax,
                cmap="YlGn", cbarlabel="harvest [t/year]", 
                size_scale1=grid_size_1, size_scale2=grid_size_2, 
                extent=[-0.5, -0.5 + leng_1 * grid_size_1, (leng_2 - 0.5) * grid_size_2, -0.5 * grid_size_2])
texts = annotate_heatmap(im, valfmt="{x:.2f}")

fig.tight_layout()
# plt.show()
plt.savefig('chart_output/corr.png')