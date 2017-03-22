# -*- coding: utf-8 -*-

import numpy as np
from numpy import *
from numpy import genfromtxt

# === Matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
# 3D plot
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
# Draw arrows
from matplotlib.patches import FancyArrowPatch


from sophus import *

import sys

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def create_arrow(position, direction, size=.1, color="r"):
    """Creates a 3D arrow"""
    s= position
    e = position + size * direction;
    a = Arrow3D([s[0],e[0]],[s[1],e[1]],[s[2],e[2]],color=color, mutation_scale=20, lw=1, arrowstyle="-|>")
    return a

def plot_transform(ax, T, color='r'):
    p =  T[0:3, 3]
    rx = T[0:3, 0]
    ry = T[0:3, 1]
    rz = T[0:3, 2]
    print p
    a1 = create_arrow(p, rx, color=color) 
    a2 = create_arrow(p, ry, color=color) 
    a3 = create_arrow(p, rz, color=color) 
    ax.add_artist(a1)
    ax.add_artist(a2)
    ax.add_artist(a3)


### Display trajectories
if len(sys.argv) < 1:
    print "Usage: display.py data.csv"

# Read lie elements from CSV
my_data = genfromtxt(sys.argv[1], delimiter=';')
print my_data


# Prepare figure
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

# Plot data
min_x=min_y=min_z=0
max_x=max_y=max_z=0
for row in my_data:
    x1 = row[0:6]
    X1 = SE3.exp(x1)
    print "X1:"
    print X1
    plot_transform(ax, X1.matrix(), color='r')
    x2 = row[6:12]
    X2 = SE3.exp(x2)
    plot_transform(ax, X2.matrix(), color='g')
    x3 = row[12:18]
    X3 = SE3.exp(x3)
    plot_transform(ax, X3.matrix(), color='b')
    # Compute axes limits
    min_x = min(min_x, min([X1.matrix()[0,3],X2.matrix()[0,3],X3.matrix()[0,3]]))
    min_y = min(min_y, min([X1.matrix()[1,3],X2.matrix()[1,3],X3.matrix()[1,3]]))
    min_z = min(min_z, min([X1.matrix()[2,3],X2.matrix()[2,3],X3.matrix()[2,3]]))
    max_x = .1 + max(max_x, max([X1.matrix()[0,3],X2.matrix()[0,3],X3.matrix()[0,3]]))
    max_y = .1 + max(max_y, max([X1.matrix()[1,3],X2.matrix()[1,3],X3.matrix()[1,3]]))
    max_z = .1 + max(max_z, max([X1.matrix()[2,3],X2.matrix()[2,3],X3.matrix()[2,3]]))

# Finalize plot
ax.set_xlim([min_x, max_x])
ax.set_ylim([min_y, max_y])
ax.set_zlim([min_z, max_z])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
 

red_patch = mpatches.Patch(color='red', label='x_pred')
green_patch = mpatches.Patch(color='green', label='x_mes')
blue_patch = mpatches.Patch(color='blue', label='x_ekf')
handles=[red_patch, green_patch, blue_patch] 
labels=[h.get_label() for h in handles]
ax.legend(handles, labels)

plt.title('Kalman')
plt.draw()
plt.show()

# 
# a = create_arrow(np.array([0,0,0]), np.array([1,1,1]))
# ax.add_artist(a)
# 
# T = np.array([[1,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
# print T
# plot_transform(ax, T)
# 
# ax.set_xlabel('x_values')
# ax.set_ylabel('y_values')
# ax.set_zlabel('z_values')
# 
# plt.title('Eigenvectors')
# 
# plt.draw()
# plt.show()
