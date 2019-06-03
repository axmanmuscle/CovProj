import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2

np.random.seed(20190601)
n = 500
mu = np.array([0, 0])
cova = np.array([[5, 3], [3, 10]])

x, y = np.random.multivariate_normal(mu, cova, n).T

fig2d = plt.figure()
ax2d = fig2d.add_subplot(111, aspect='equal')
ax2d.set_title('2D / XY Scatter')


perc_cont = 0.95
df = 2

scale = chi2.ppf(perc_cont, df)

w, v = np.linalg.eig(cova)

ind = np.argmin(w)

x_eig = v[0, ind]
y_eig = v[1, ind]

alpha = np.arctan(y_eig/x_eig)

ell = Ellipse(xy=(mu[0], mu[1]), width=np.sqrt(w[0]*scale)*2, height=np.sqrt(w[1]*scale)*2,
              angle=alpha* 180/np.pi, color='black')

print(np.sqrt(w[0]*scale)*2)
print(np.sqrt(w[1]*scale)*2)
print(alpha)
ell.set_facecolor('none')
ax2d.add_artist(ell)

ax2d.set_xlim(-20, 20)
ax2d.set_ylim(-20, 20)
ax2d.scatter(x, y, marker='.', color='b')
plt.show()