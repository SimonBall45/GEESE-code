# HR Logmass

import os
import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as  plt
import matplotlib.colors as cols
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit

# =============================================
# LOAD DATA
# =============================================

hd = pd.read_csv('DES-Dovekie_HD.csv')
meta = pd.read_csv('DES-Dovekie_Metadataa.csv', delimiter=r'\s+')
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

# =============================================
# Distance modulus error cut off
# =============================================

for x in meta.index:
    if meta.loc[x, 'MUcalcERR'] >1:
        meta.drop(x, inplace=True)
MUcalcerr = np.array(meta['MUcalcERR'])

# =============================================
# Hubble resiudal calculation and Host Galaxy Properties
# =============================================


meta['MUcomp'] = meta['MUcalc'] - meta['MU'] #test against DES
meta['HRmod'] = meta['MUcalc'] - meta['MUMODEL'] #our calculated hr values

mucomp = meta['MUcomp']
hrmod = np.array(meta['HRmod'])


logmass = np.array(meta['HOST_LOGMASS'])
logmasserr = meta['HOST_LOGMASS_ERR']

logSFR = meta['HOST_LOGSFR']
logSFRerr= meta['HOST_LOGMASS_ERR']

logsSFR = meta['HOST_LOGsSFR']
logsSFRerr = meta['HOST_LOGsSFR_ERR']

color = meta['HOST_COLOR']
colorerr= np.absolute(meta['HOST_COLOR_ERR'])



# =============================================
# Weighted mean mass
# =============================================

mass_below = logmass < 10
mass_above = logmass > 10

#print(mass_below)

def rms(x,N):
    return np.sqrt(np.sum((x - np.mean(x))**2)/N)


def weighted_mean(x, w):
    return np.sum(x * w) / np.sum(w)

def weighted_mean_err(x, w, N):
    return np.sqrt((np.sum(w*(x-hrmod)**2))/((np.sum(w)*(N-1))))


# ---- compute RMS scatter for each subsample ----
rms_below = rms(hrmod[mass_below], 434)
rms_above = rms(hrmod[mass_above], 836)


# ---- total variance = measurement + intrinsic scatter ----
sigma2_below = MUcalcerr[mass_below]**2 + rms_below**2
sigma2_above = MUcalcerr[mass_above]**2 + rms_above**2


# ---- inverse-variance weights ----
weights_below = 1.0 / sigma2_below
weights_above = 1.0 / sigma2_above


# ---- weighted means ----
below_mean = weighted_mean(
    hrmod[mass_below],
    weights_below
)

above_mean = weighted_mean(
    hrmod[mass_above],
    weights_above
)

print("Weighted mean HR (logM < 10):", below_mean)
print("Weighted mean HR (logM > 10):", above_mean)

# =============================================
# Chi squared calc
# =============================================
step = np.where(logmass < 10, below_mean, above_mean)
def chi(x, w, N):
    return ((np.sum(w*(x-hrmod)**2))/(N-1))

rmschi = rms(hrmod, 1270)
sigma2chi = MUcalcerr**2 + rmschi**2
weightschi = 1.0 / sigma2chi

chisq_step = chi(step, weightschi, 1270)

print("reduced chi squared step", chisq_step)


# ---- uncertainty on weighted means ----
below_mean_err1 = 1.0 / np.sqrt(np.sum(weights_below))
above_mean_err1 = 1.0 / np.sqrt(np.sum(weights_above))

below_mean_err2 = (np.sqrt(chisq_step))*below_mean_err1
above_mean_err2 = (np.sqrt(chisq_step))*below_mean_err2

below_mean_err3 = np.sqrt(((below_mean_err1)**2)+((below_mean_err2)**2))
above_mean_err3 = np.sqrt(((above_mean_err1)**2)+((above_mean_err2)**2))
print("Weighted mean error (logM < 10):", below_mean_err3)
print("Weighted mean error (logM > 10):", above_mean_err3)
# ---- mass step ----
mass_step = below_mean - above_mean
mass_step_err = np.sqrt(below_mean_err3**2 + above_mean_err3**2)

print("Mass step ΔHR:", mass_step)
print("Mass Step error", mass_step_err)
print("Significance:", mass_step / mass_step_err)



# =============================================
# Sigmoid fit
# =============================================

def sigmoid(M, A, delta, M0, w):
    return ((delta)/(1+np.exp(-(M-M0)/w))) -A

p0 = [-0.005, mass_step, 10, 1]

values, covs = curve_fit(sigmoid, logmass, hrmod, p0,)
sigmoidData = sigmoid(logmass, *values)


print(covs)
print(np.diag(covs))
print(np.sqrt(np.diag(covs)))
m = np.linspace(7, 12,1000)
plt.plot(m, sigmoid(m,*values))

print(values)

chisq_sigmoid=chi(sigmoidData, weightschi, 1270)
print("reduced chi squared sigmoid", chisq_sigmoid)



# =============================================
# Graph production
# =============================================


plt.figure(figsize=(8, 5))

# Scatter with errors
plt.errorbar(
    x=logmass,
    y=hrmod,
    yerr=MUcalcerr,
    xerr=logmasserr,
    fmt='o',
    elinewidth=0.1,
    markersize=2,
    ecolor='grey',
    color='grey',
    zorder = 1
)

#sigmoid fit
m = np.linspace(7, 12,1000)
#plt.plot(m, sigmoid(m,*values))


# Zero line
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.7)

# mass cut
cut = 10
plt.axvline(x=cut, color='black', linestyle='--', linewidth=0.7)

# ---- define x-limits for the mean lines ----
xmin = np.min(logmass)
xmax = np.max(logmass)

# ---- weighted mean segments ----
plt.hlines(
    y=below_mean,
    xmin=8.5,
    xmax=cut,
    colors='red',
    linestyles='-',
    linewidth=2,
    zorder =2
)

plt.hlines(
    y=above_mean,
    xmin=cut,
    xmax=11.5,
    colors='red',
    linestyles='-',
    linewidth=2,
    zorder = 2
)

plt.xlabel(r'$\log(\mathrm{M_*/M_\odot})$', fontsize = '30', color='grey')
plt.ylabel('Hubble Residual', fontsize = '30', color='grey')
plt.legend(frameon=False)
ax = plt.gca()
ax.set_xlim(8,12)
ax.xaxis.set_minor_locator(AutoMinorLocator(5))  # 0.1 dex minor ticks
ax.tick_params(axis='x', which='major', length=6, labelsize='20')
ax.tick_params(axis='x', which='minor', length=3)
ax.tick_params(axis='y', labelsize = '20')
plt.show()

