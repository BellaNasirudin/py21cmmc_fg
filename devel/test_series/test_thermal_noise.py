from gridded_noise_nofg import core_eor, core_instr, likelihood, model_name
from py21cmmc.mcmc.mcmc import build_computation_chain
import pickle
import numpy as np
import matplotlib.pyplot as plt


nrealisations = 100

# Build the chain and run setup()
chain = build_computation_chain([core_eor, core_instr], likelihood)

num_mean, num_cov = likelihood.numerical_covariance(nrealisations=nrealisations, nthreads=4)

num_var = np.array([np.diag(c) for c in num_cov])

anl_mean, anl_cov = likelihood.noise['mean'], likelihood.noise['covariance']

anl_var = np.array([np.diag(c) for c in anl_cov])

with open("thermal_noise_data.pkl", 'wb') as f:
    pickle.dump({"num_mean":num_mean, "num_var":num_var, "anl_mean":anl_mean, "anl_cov":anl_var}, f)


# Make a plot
fig, ax = plt.subplots(2,3, sharex=True, sharey=True,
                       subplot_kw={"xscale":'log', 'yscale':'log'},
                       figsize=(12,8)
                      )
fig.suptitle("Thermal Noise Power")

extent=(likelihood.u.min(), likelihood.u.max(),likelihood.eta.min(),likelihood.eta.max())

im = ax[0,0].imshow(num_mean.T, origin='lower', extent=extent)
plt.colorbar(im, ax=ax[0,0])
ax[0,0].set_title("Numerical mean")


im = ax[0,1].imshow(anl_mean.T, origin='lower', extent=extent)
plt.colorbar(im, ax=ax[0, 1])
ax[0,1].set_title("Analytic Mean")

im = ax[0,2].imshow((num_mean/anl_mean).T, origin='lower', extent=extent)
plt.colorbar(im, ax=ax[0,2])
ax[0,2].set_title("Num/Anl Mean")

im = ax[1,0].imshow(num_var.T, origin='lower', extent=extent)
plt.colorbar(im, ax=ax[1, 0])
ax[1,0].set_title("Numerical Var.")

im = ax[1,1].imshow(anl_var.T, origin='lower', extent=extent)
plt.colorbar(im, ax=ax[1,1])
ax[1,1].set_title("Analytic var.")

im = ax[1,2].imshow((num_var/anl_var).T, origin='lower', extent=extent)
plt.colorbar(im, ax=ax[1,2])
ax[1,2].set_title("Num./Anl. Var")

fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Perpendicular Scale, $u$", labelpad=15, fontsize=15)
plt.ylabel("Line-of-Sight Scale, $\eta$", labelpad=15, fontsize=15)

plt.savefig("thermal_noise_test.png")
