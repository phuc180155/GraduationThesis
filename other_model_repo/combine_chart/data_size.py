import matplotlib.pyplot as plt
x = [900000,500000,20000,500000,100000,800000,1500000]
y = [707,6000,100,5000,430,5000,30173]
s = [x[i]/2000+y[i]/10 for i in range(len(x))]
c = ['r','g','b','c','m','y','tab:orange']
la = ['df_in_the_wild','Celeb-DF','UADFV','FF+','DF-TIMIT','DFDC','Our_dataset']

fig, ax = plt.subplots()

for i in range(len(x)):
    scatter = ax.scatter(x[i],y[i],s=s[i],color=c[i],label=la[i])
# plt.title('Number deepfake')
plt.xlabel('Number images')
plt.ylabel('Number videos')
plt.xlim(0,1.7e6)
plt.ylim(0,40000)
# ax.legend()
# legend1 = ax.legend(la,
#                     loc="upper left", title="Ranking")
# ax.add_artist(legend1)
# plt.legend(scatterpoints=1,markerscale=0.5,)
# Plot legend.

lgnd = plt.legend(numpoints=1, fontsize=10)
#
# #change the marker size manually for both lines
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
lgnd.legendHandles[2]._sizes = [30]
lgnd.legendHandles[3]._sizes = [30]
lgnd.legendHandles[4]._sizes = [30]
lgnd.legendHandles[5]._sizes = [30]
lgnd.legendHandles[6]._sizes = [30]
plt.savefig('dataset_size.png')

# plt.show()