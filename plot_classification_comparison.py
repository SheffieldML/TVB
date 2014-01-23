# Copyright (c) 2014, James Hensman, Max Zwiessele
# Distributed under the terms of the GNU General public License, see LICENSE.txt

import numpy as np
import matplotlib
#matplotlib.use('pdf')
import pylab as pb
#pb.ion(); pb.close('all')
import os
from scipy import stats

dirname = 'raw_results_classification2'
fnames = [e for e in os.listdir('raw_results_classification2') if e[-11:]=='raw_results']

data = [np.loadtxt(os.path.join(dirname,fn)) for fn in fnames]

means = np.vstack([stats.nanmean(e, 0) for e in data])
stds = np.vstack([stats.nanstd(e, 0) for e in data])

x = np.arange(len(data))
width=0.35

def do_plots(i1, i2, lab1="", lab2=""):
    pb.figure(figsize=(10,4))
    error_kw = {'elinewidth':1.2, 'ecolor':'k', 'capsize':5, 'mew':1.2}
    pb.bar(x, means[:,i1], yerr=2*stds[:,i1], width=width, color='b', label=lab1, error_kw=error_kw)
    pb.bar(x+width, means[:,i2], yerr=2*stds[:,i2], color='r', width=width, label=lab2, error_kw=error_kw)
    pb.xticks(x+width,[fn.split('raw')[0] for fn in fnames], rotation=45)

    for xx, m, s in zip(x, means[:,i1], 2*stds[:,i1]):
        pb.text(xx+0.5*width, 1.0*(m+s), '%.3f'%m,ha='center', va='bottom', fontsize='small')
    for xx, m, s in zip(x+width, means[:,i2], 2*stds[:,i2]):
        pb.text(xx+0.5*width, 1.0*(m+s), '%.3f'%m,ha='center', va='bottom', fontsize='small')

    #pb.legend(loc=0)
    pb.ylim(0,1.05*np.max(means[:,[1,5]].flatten() + 2*stds[:,[1,5]].flatten()))
    pb.ylabel(r'$-\log\, p(y_\star)$')
    pb.subplots_adjust(bottom=0.2)
    pb.legend(loc=0)

#do_plots(1,5, "varEP", "EP")
#pb.savefig('nlps.pdf')
#
#do_plots(7,3, "varEP (EP params)", "EP (varEP params)")
#pb.savefig('cross_compare.pdf')
#
#do_plots(1,7, "varEP (varEP params)", "varEP (EP params)")
#pb.savefig('varEP_param_compare.pdf')
#
#do_plots(0,4, "varEP", "EP")
#pb.title('hold out error')
#pb.savefig('errors.pdf')

def whiskers(i1, i2, lab1="", lab2=""):
    width = 0.35
    l1 = pb.boxplot([d[:, i1] for d in data] , positions=np.arange(len(data))-1.03*width/2., widths=width) 
    l2 = pb.boxplot([d[:, i2] for d in data] , positions=np.arange(len(data))+1.03*width/2., widths=width) 
    pb.xticks(np.arange(len(data)),[fn.split('raw')[0].replace('_',' ') for fn in fnames], rotation=45)
    pb.xlim(-1.2*width, len(data)-1+1.2*width)
        
    for key, lines in l1.iteritems():
        pb.setp(lines, lw=1)
        if key == "boxes":
            pb.setp(lines, color='b', lw=1.4)
        if key == 'whiskers':
            pb.setp(lines, color='b')
        if key == 'fliers':
            pb.setp(lines, color='b')
        if key == 'medians':
            pb.setp(lines, color='k', lw=1.4)
    for key, lines in l2.iteritems():
        pb.setp(lines, lw=1.2)
        if key == "boxes":
            pb.setp(lines, color='g', lw=1.4)
        if key == 'whiskers':
            pb.setp(lines, color='g')
        if key == 'fliers':
            pb.setp(lines, color='g')
        if key == 'medians':
            pb.setp(lines, color='k', lw=1.4)
    #pb.setp(l2['boxes'], color='g')
    #pb.setp(l1['medians'], color='b')
    #pb.setp(l2['medians'], color='g')
    #pb.setp(l1['whiskers'], color='b')
    #pb.setp(l2['whiskers'], color='g')

#os.makedirs('classification_plots')
import matplotlib as mpl; mpl.rcParams['text.usetex'] = False
pb.close('all')
pb.figure('holdout', figsize=(8,3))
pb.ylabel(u'Fraction error')
whiskers(0,4, "varEP", "EP")
pb.tight_layout()
pb.savefig('/home/maxz/Documents/publications/varEP/aistats2014/classification_plots/holdout.pgf')

pb.close('all')
pb.figure('crossparameters', figsize=(8,3))
pb.ylabel(u'Fraction error')
whiskers(6,2, "varEP", "EP")
pb.tight_layout()
pb.savefig('/home/maxz/Documents/publications/varEP/aistats2014/classification_plots/crossparams.pgf')

pb.figure('negprob', figsize=(8,3))
whiskers(1,5, "varEP", "EP")
pb.ylabel(u'$-\log{p}(\mathbf{y}^{\star})$')
pb.tight_layout()
pb.savefig('/home/maxz/Documents/publications/varEP/aistats2014/classification_plots/negprob.pgf')
mpl.rcParams['text.usetex'] = True
    







