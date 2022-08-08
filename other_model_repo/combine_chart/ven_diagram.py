import matplotlib.pyplot as plt
import numpy as np
figure, axes = plt.subplots()


draw_circle = plt.Circle((0.5, 0.5), 0.2,fill=False)
axes.set_aspect(1)
axes.add_artist(draw_circle)
axes.annotate("cpicpi", xy=(0.5, 0.5), fontsize=10)
plt.show()