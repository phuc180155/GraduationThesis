import matplotlib.pyplot as plt
x = [0.94,0.96,0.99,0.90,0.92,0.50,0.5]
y = [0.92,0.96,0.99,0.93,0.90,0.5,0.51]
s = [x[i]/2000+y[i]/10 for i in range(len(x))]
c = ['r','g','b','c','m','y','tab:orange','tab:pink']
la = ['Mesonet','Capsule','XceptionNet','GAN-fp','FDBD','HPBD','VA']

fig, ax = plt.subplots()

for i in range(len(x)):
    scatter = ax.scatter(x[i],y[i],color=c[i],label=la[i])
# plt.title('Number deepfake')
plt.xlabel('Accuracy')
plt.ylabel('Precision')
# plt.xlim(0,1.7e6)
# plt.ylim(0,40000)
# ax.legend()
# legend1 = ax.legend(la,
#                     loc="upper left", title="Ranking")
# ax.add_artist(legend1)
# plt.legend(scatterpoints=1,markerscale=0.5,)
# Plot legend.

lgnd = plt.legend(numpoints=1, fontsize=10,loc="lower right")
#
# #change the marker size manually for both lines
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
lgnd.legendHandles[2]._sizes = [30]
lgnd.legendHandles[3]._sizes = [30]
lgnd.legendHandles[4]._sizes = [30]
lgnd.legendHandles[5]._sizes = [30]
lgnd.legendHandles[6]._sizes = [30]
plt.savefig('acc_pre.png')

plt.show()