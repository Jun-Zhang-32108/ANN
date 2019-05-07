# Use: A code snippet for drawing 3D images
# Author: Jun Zhang <junzha@kth.se>
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d.axes3d import Axes3D
x = []
y = []
targets = []
for i in range(-50, 50, 5):
    for j in range(-50, 50, 5):
        x.append(i/10.0)
        y.append(j/10.0)
        targets.append(np.exp(-1*(((i/10.0)**2+(j/10.0)**2)/10)) - 0.5)
x = np.array(x)
y = np.array(y)
targets = np.array(targets)
samples = np.column_stack([x,y])
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-5, 5, 0.5)
Y = np.arange(-5, 5, 0.5)
X, Y = np.meshgrid(X, Y)
T = np.exp(-1*(X**2+Y**2)/10)-0.5
print(T)
# p = ax.plot_surface(X, Y, T)
p = ax.plot(x, y, targets)
targets = np.reshape(targets, (400, 1))
shuffled_index = np.random.permutation(200)
samples = samples[shuffled_index].T
targets = targets[shuffled_index].T
plt.show()