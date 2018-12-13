from py21cmmc_fg.c_wrapper import stitch_and_coarsen_sky
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 23)
X, Y = np.meshgrid(x,x)

z = X**2 + Y**2
z = np.atleast_3d(z)
z = np.tile(z ,(1,1,6))

print(z.shape)

plt.imshow(z[:,:,0])
plt.savefig("derp1.png")

z = np.atleast_3d(z)
out = stitch_and_coarsen_sky(z, 1, 2.3, 27)
plt.clf()

print(out)

plt.imshow(out[:,:,0])
plt.savefig("derp2.png")
