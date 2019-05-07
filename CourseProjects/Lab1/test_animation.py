# A code snippet for drawing annimation
# Author: Jun Zhang <junzha@kth.se>
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import pandas as pd

#This is the weights of each epoch
W = pd.read_csv('W_output_Dataset_HasBias_NonlinearlySepartable_.csv') # w_x*x+w_y*y+w_bias = 0 so y = (-1*w_bias-w_x*x)/w_y
w_1 = np.array(W.iloc[0,0:3])
print(w_1)
w_rest = np.array(W.iloc[1:,0:3])

# This is the original dataset
Data = pd.read_csv('Dataset_Unshuffled_NonlinearlySepartable.csv',header = None)
Data = np.array(Data)

x_d_1 = []
y_d_1 = []
x_d_2 = []
y_d_2 = []
for i in range(200):
	if (Data[i][3] == 1):
		x_d_1.append(Data[i][0])
		y_d_1.append(Data[i][1])
	else:
		x_d_2.append(Data[i][0])
		y_d_2.append(Data[i][1])
fig, ax = plt.subplots()
fig.set_tight_layout(True)
plt.title('Annimation_Delta_Batch')
plt.grid()
print('fig size: {0} DPI, size in inches {1}'.format(
    fig.get_dpi(), fig.get_size_inches()))


x = np.arange(-2.5, 2.5, 0.1)
#Draw the scatter fig of the original dataset 
ax.scatter(x_d_1, y_d_1)
ax.scatter(x_d_2, y_d_2)
#Draw the first pic of the boundary line
line, = ax.plot(x, (-1*w_1[2]-w_1[0]*x)/w_1[1], 'r-', linewidth=1) #with bias

# line, = ax.plot(x, (-w_1[0]*x)/w_1[1], 'r-', linewidth=1) #without bias


def update(i):
    label = 'Epoch {0}'.format(i)
    print(label)

    #with bias
    line.set_ydata((-1*w_rest[i][2]-w_rest[i][0]*x)/w_rest[i][1])
    
    #without bias
    # line.set_ydata((-w_rest[i][0]*x)/w_rest[i][1])
    ax.set_xlabel(label)
    return line, ax

if __name__ == '__main__':
    anim = FuncAnimation(fig, update, frames=np.arange(0, 199), interval=100)
    # if you want to save the file, input the parameter "save" when executing the program
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        anim.save(str(sys.argv[2])+'.html')
    else:
        plt.show()