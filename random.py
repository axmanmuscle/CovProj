import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


np.setseed(20190604)

mu_2 = np.array([[0, 0]])
cov_2 = np.array([[5, 1], [1, 7]])

x2, y2 = np.random.multivariate_normal(mu_2, cov_2, 500)

np.setseed(20190604)

mu_3 = np.array([[0, 0, 0]])
cov_3 = np.array([[5, 1, 2], [1, 7, 3], [2, 3, 10]])

x3, y3, z3 = np.random.multivariate_normal(mu_3, cov_3, 500)

fig2d = plt.figure()
ax2d = fig2d.add_subplot(111)
ax2d.title('2D')
ax2d.scatter(x2, y2, 'r.')

fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.title('3D')
ax3d.scatter(x3, y3, z3, 'r.')
