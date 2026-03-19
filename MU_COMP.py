# HR Logmass

import os
import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as  plt
import matplotlib.colors as cols
from matplotlib.ticker import AutoMinorLocator

# =============================================
# LOAD DATA
# =============================================

hd = pd.read_csv('DES-Dovekie_HD.csv')
meta = pd.read_csv('DES-Dovekie_Metadata.csv')

# =============================================
# Type 1a probabiltity cut at 99.9%
# =============================================

for x in meta.index:
    if meta.loc[x, 'PROB_SNNV19'] <0.999:
        meta.drop(x, inplace=True)

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

# ============================================
# Distance modulus calculation 
# ============================================

#np.where(condition, value_if_true, value_if_false)
mass_step = np.where(logmass < 10, +gamma/2, -gamma/2)

meta['MUcalc'] = (
    -5*np.log10(x0)/2
    + alpha*x1
    - beta*c
    - mubiascor
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

# =============================================
# Distance modulus error cut off
# =============================================

for x in meta.index:
    if meta.loc[x, 'MUcalcERR'] >0.1:
        meta.drop(x, inplace=True)
MUcalcerr = np.array(meta['MUcalcERR'])

# =============================================
# Hubble resiudal calculation and Host Galaxy Properties
# =============================================


meta['MU_comp'] = meta['MUcalc'] - meta['MU'] #test against DES
meta['HRmod'] = meta['MUcalc'] - meta['MUMODEL'] #our calculated hr values

mucomp = meta['MU_comp']
hrmod = np.array(meta['HRmod'])


logmass = np.array(meta['HOST_LOGMASS'])
logmasserr = meta['HOST_LOGMASS_ERR']

logSFR = meta['HOST_LOGSFR']
logSFRerr= meta['HOST_LOGMASS_ERR']

logsSFR = meta['HOST_LOGsSFR']
logsSFRerr = meta['HOST_LOGsSFR_ERR']

color = meta['HOST_COLOR']
colorerr= np.absolute(meta['HOST_COLOR_ERR'])

plt.figure(figsize=(8, 5))

# Scatter with errors
plt.errorbar(
    x=logmass,
    y=mucomp,
    yerr=MUcalcerr,
    xerr=logmasserr,
    fmt='o',
    elinewidth=0.1,
    markersize=2,
    color='grey',
    zorder = 1
)

# Zero line
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.7)

# mass step
step = 10
plt.axvline(x=step, color='black', linestyle='--', linewidth=0.7)
# ---- define x-limits for the mean lines ----
xmin = np.min(logmass)
xmax = np.max(logmass)

# ---- weighted mean segments ----
plt.hlines(
    y=gamma/2,
    xmin=xmin,
    xmax=step,
    colors='red',
    linestyles='-',
    linewidth=1,
    zorder = 2
)

plt.hlines(
    y=-gamma/2,
    xmin=step,
    xmax=xmax,
    colors='red',
    linestyles='-',
    linewidth=1,
    zorder = 2
)

plt.xlabel(r'$\log(\mathrm{M_*/M_\odot})$', fontsize='25', color='grey')
plt.ylabel('MU difference', fontsize='25', color='grey')
plt.legend(frameon=False)
ax = plt.gca()
#ax.set_xlim(8,12)
#ax.set_ylim(-0.3,0.3)
ax.xaxis.set_minor_locator(AutoMinorLocator(5))  # 0.1 dex minor ticks
ax.tick_params(axis='x', which='major', length=6, labelsize=15)
ax.tick_params(axis='x', which='minor', length=3)
ax.tick_params(axis='y', labelsize=15)


# Get current automatic ticks
yticks = list(ax.get_yticks())

# Add ±gamma/2 explicitly
yticks.extend([gamma/2, -gamma/2])

# Sort and remove duplicates
yticks = np.unique(yticks)

# Apply ticks
ax.set_yticks(yticks)

# Now label them
yticklabels = []
for y in yticks:
    if np.isclose(y,  gamma/2):
        yticklabels.append(r'$+\gamma/2$')
    elif np.isclose(y, -gamma/2):
        yticklabels.append(r'$-\gamma/2$')
    else:
        yticklabels.append(f'{y:.2f}')

ax.set_yticklabels(yticklabels)

plt.show()
