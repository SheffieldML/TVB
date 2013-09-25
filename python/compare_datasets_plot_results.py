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
error_kw = {'elinewidth':1.2, 'ecolor':'k','mew':1.2}
rects1 = pb.bar(x, means[:,1], yerr=stds[:,1], width=width, color='b', label='var_EP', error_kw=error_kw)
rects2 = pb.bar(x+width, means[:,5], yerr=stds[:,5], color='r', width=width, label='EP', error_kw=error_kw)
pb.xticks(x+width,[fn.split('raw')[0] for fn in fnames])

for xx, m, s in zip(x, means[:,1], stds[:,1]):
    pb.text(xx+0.5*width, 1.0*(m+s), '%.3f'%m,ha='center', va='bottom')
for xx, m, s in zip(x+width, means[:,5], stds[:,5]):
    pb.text(xx+0.5*width, 1.0*(m+s), '%.3f'%m,ha='center', va='bottom')

#pb.legend(loc=0)
pb.ylim(0,0.7)

pb.savefig('nlps.pdf')
