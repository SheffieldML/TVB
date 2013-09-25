import numpy as np
import matplotlib
matplotlib.use('pdf')
import pylab as pb
import os
from scipy import stats

fnames = [e for e in os.listdir('.') if 'raw_res' in e]

data = [np.loadtxt(fn) for fn in fnames]

means = np.vstack([stats.nanmean(e, 0) for e in data])
stds = np.vstack([stats.nanstd(e, 0) for e in data])

x = np.arange(len(data))
width=0.35

pb.figure()
error_kw = {'linewidth':1.2, 'color':'k'}
pb.bar(x, means[:,1], yerr=stds[:,1], width=width, color='b', label='var_EP', error_kw=error_kw)
pb.bar(x+width, means[:,5], yerr=stds[:,5], color='r', width=width, label='EP', error_kw=error_kw)
pb.legend()
pb.xticks(x+width,[fn.split('raw')[0] for fn in fnames])
pb.savefig('nlps.pdf')

