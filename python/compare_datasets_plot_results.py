import numpy as np
import matplotlib
#matplotlib.use('pdf')
import pylab as pb
pb.ion()
import os
from scipy import stats

fnames = [e for e in os.listdir('.') if 'raw_res' in e]

data = [np.loadtxt(fn) for fn in fnames]

means = np.vstack([stats.nanmedian(e, 0) for e in data])
stds = np.vstack([stats.nanstd(e, 0) for e in data])

x = np.arange(len(data))
width=0.35

def do_plots(i1, i2, lab1="", lab2=""):
    pb.figure(figsize=(10,4))
    error_kw = {'elinewidth':1.2, 'ecolor':'k', 'capsize':5, 'mew':1.2}
    pb.bar(x, means[:,i1], yerr=stds[:,i1], width=width, color='b', label=lab1, error_kw=error_kw)
    pb.bar(x+width, means[:,i2], yerr=stds[:,i2], color='r', width=width, label=lab2, error_kw=error_kw)
    pb.xticks(x+width,[fn.split('raw')[0] for fn in fnames], rotation=45)

    for xx, m, s in zip(x, means[:,i1], stds[:,i1]):
        pb.text(xx+0.5*width, 1.0*(m+s), '%.3f'%m,ha='center', va='bottom', fontsize='small')
    for xx, m, s in zip(x+width, means[:,i2], stds[:,i2]):
        pb.text(xx+0.5*width, 1.0*(m+s), '%.3f'%m,ha='center', va='bottom', fontsize='small')

    #pb.legend(loc=0)
    pb.ylim(0,1.05*np.max(means[:,[1,5]].flatten() + stds[:,[1,5]].flatten()))
    pb.ylabel(r'$-\log\, p(y_\star)$')
    pb.subplots_adjust(bottom=0.2)
    pb.legend(loc=0)

do_plots(1,5, "varEP", "EP")
pb.savefig('nlps.pdf')

do_plots(7,3, "varEP", "EP")
pb.savefig('cross_compare.pdf')

do_plots(1,7, "varEP (vaEP params)", "varEP (EP params)")
pb.savefig('varEP_param_compare.pdf')


