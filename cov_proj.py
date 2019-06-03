import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.patches import Ellipse
from scipy.stats import chi2

def cov_2d(cov, perc_cont):

    if cov.shape[0] != cov.shape[1]:
        return -1

    scale = chi2.ppf(perc_cont, 2)
    
    w, v = np.linalg.eig(cov)

    ind = np.argmin(w)
    
    x_eig = v[0, ind]
    y_eig = v[1, ind]

    alpha = np.arctan(y_eig/x_eig)

    ell = Ellipse(xy=(mu[0], mu[1]), width=np.sqrt(w[0]*scale)*2, height=np.sqrt(w[1]*scale)*2,
                      angle=alpha* 180/np.pi, color='black')

    ell.set_facecolor('none')
    return ell
    
def cov_3d(cov, perc_cont):
    assert cov.shape==(3,3)

    # Find and sort eigenvalues to correspond to the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.sum(cov,axis=0).argsort()
    eigvals_temp = eigvals[idx]
    idx = eigvals_temp.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    # Set of all spherical angles to draw our ellipsoid
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)
    
    nstd = chi2.ppf(perc_cont, 3)

    # Width, height and depth of ellipsoid
    rx, ry, rz = np.sqrt(nstd) * np.sqrt(eigvals)

    # Get the xyz points for plotting
    # Cartesian coordinates that correspond to the spherical angles:
    X = rx * np.outer(np.cos(theta), np.sin(phi))
    Y = ry * np.outer(np.sin(theta), np.sin(phi))
    Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    # Rotate ellipsoid for off axis alignment
    old_shape = X.shape
    # Flatten to vectorise rotation
    X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
    X,Y,Z = np.matmul(eigvecs, np.array([X,Y,Z]))
    X,Y,Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)
   
    # Add in offsets for the mean
    X = X + mu[0]
    Y = Y + mu[1]
    Z = Z + mu[2]
    
    return X,Y,Z


np.random.seed(20190601)
n = 500
mu = np.array([0, 0, 0])
cova = np.array([[5, 2, 2], [2, 10, 3], [2, 3, 8]])

x, y, z = np.random.multivariate_normal(mu, cova, n).T

fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.set_title('Full 3D')
ax3d.set_xlabel('x')
ax3d.set_ylabel('y')
ax3d.set_zlabel('z')
ellX, ellY, ellZ = cov_3d(cova, .95)
ax3d.plot_wireframe(ellX, ellY, ellZ, color='b', alpha=0.1)
ax3d.scatter(x, y, z)

fig2d = plt.figure()
ax2d = fig2d.add_subplot(111)
ax2d.set_title('2D / XY Scatter')
ell_2d = cov_2d(cova[:2, :2], 0.95)
ax2d.add_artist(ell_2d)
ax2d.scatter(x, y)

figle = plt.figure()
axle = figle.add_subplot(111)
axle.set_title('Linear / Vertical')
axle.plot(z, 'r.')
plt.show()

## To do:
# Plot 3d 1sigma/2sigma/3sigma/95% containment ellipses
# same for 2d
# project ellipse to 2d
# intersect ellipse to 2d
# compare to containment factors kurt sent