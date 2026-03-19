import os
import pandas as pd
import numpy as np
from astropy.cosmology import Flatw0waCDM
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as  plt
import matplotlib.colors as cols
from matplotlib.ticker import AutoMinorLocator

hd = pd.read_csv('DES-Dovekie_HD.csv')
meta = pd.read_csv('DES-Dovekie_Metadata.csv')

for x in meta.index:
    if meta.loc[x, 'PROB_SNNV19'] <0.999:
        meta.drop(x, inplace=True)

# 1. Define your cosmology (DES often uses Planck-like parameters)
cosmo = Flatw0waCDM(H0=70, Om0=0.473, w0=-0.497, wa=-7.46)
cosmo2 = FlatLambdaCDM(H0=70, Om0=0.330)

# 2. Create a redshift array (covering DES SN range ~0.01 to 1.2)
z = np.linspace(0.05, 1.1, 1000)

# 3. Compute distance modulus μ(z)
mumod = cosmo.distmod(z).value  # in magnitudes

z2 = np.linspace(0.05, 1.1, 1000)
mumodlambda = cosmo2.distmod(z2).value

# =============================================
# SALT3 Parameters
# =============================================

alpha=0.169
alphaerr=0.0003
beta=3.14
betaerr=0.04
MOavg=-29.96210

# =============================================
# Column Extraction
# =============================================

mB=meta['mB']
mBerr=meta['mBERR']
x1=meta['x1']
x1err=meta['x1ERR']
c=meta['c']
cerr=meta['cERR']
mubiascor=meta['biasCor_mu']
gamma = 0.033
gammaerr = 0.008
x0=meta['x0']
x0err=meta['x0ERR']
logmass = meta['HOST_LOGMASS']
prob = meta['PROB_SNNV19']

# ============================================
# Distance modulus calculation 
# ============================================

#np.where(condition, value_if_true, value_if_false)
mass_step = np.where(logmass < 10, +gamma/2, -gamma/2)

meta['MUcalc'] = (
    -5*np.log10(x0)/2
    + alpha*x1
    - beta*c
    #- mubiascor
    - MOavg
   # - mass_step
)
# ============================================
# Distance modulus error calculation
# ============================================

#meta['MUcalcERR'] = np.absolute(x0err/x0)+np.absolute(x1err/x1)+(alphaerr/alpha)+(betaerr/beta)+np.absolute(cerr/c)
meta['MUcalcERR'] = np.sqrt(
    ((5 / (2 * np.log(10))) * (x0err / x0))**2 +
    (alpha * x1err)**2 +
    (beta * cerr)**2 +
    (x1 * alphaerr)**2 +
    (c * betaerr)**2
)


zHD = meta['zHD']
mu = meta['MUcalc']
xfit = z
yfit = mumod
x2fit = z2
y2fit = mumodlambda
zHDerr = meta['zHDERR']
MUerr = meta['MUcalcERR']


plt.figure(figsize=(8, 5))

# Create scatter with colormap
sc = plt.scatter(
    zHD, mu,
    c=prob,
    cmap='cividis',   # you can change this (plasma, inferno, etc.)
    s=10,
    edgecolor='none'
)

# Optional: add faint error bars
plt.errorbar(
    x=zHD, y=mu,
    yerr=MUerr, xerr=zHDerr,
    fmt='none',
    ecolor='gray',
    elinewidth=0.5,
    alpha=0.3
)

# Add colorbar
cbar = plt.colorbar(sc)
cbar.set_label('Supernova Probability (PROB-SNNV19)', fontsize=12)

# Model line
plt.plot(xfit, yfit, linewidth=1, color='red', zorder=20, label='$w_0w_aCDM model$')

# Labels
plt.xlabel(r'$Redshift(z)$', fontsize=25, color='grey')
plt.ylabel('Distance Modulus' r'$(\mathrm{\mu})$', fontsize=25, color='grey')
plt.legend(frameon=False, loc='upper left', fontsize=15)

# Axis styling (unchanged)
ax = plt.gca()
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.tick_params(axis='x', which='major', length=6, labelsize=15)
ax.tick_params(axis='x', which='minor', length=3)
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.tick_params(axis='y', which='major', length=6, labelsize=15)
ax.tick_params(axis='y', which='minor', length=3)

plt.show()