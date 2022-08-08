import pylab
fig = pylab.figure()

figlegend = pylab.figure(figsize=(3,2))

ax = fig.add_subplot(111)

lines = ax.plot(range(10), pylab.randn(10), range(10), pylab.randn(10),range(10), pylab.randn(10),range(10), pylab.randn(10),range(10), pylab.randn(10),range(10), pylab.randn(10))
figlegend.legend(lines, ("0", "0.1", "0.2", "0.3", "0.4",'0.5'), 'center')
fig.show()
figlegend.show()
# figlegend.savefig('legend.png')