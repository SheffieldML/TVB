# Copyright (c) 2014, James Hensman
# Distributed under the terms of the GNU General public License, see LICENSE.txt

import numpy as np
import pylab as pb
from scipy.io import loadmat
import GPy
from classification2 import classification
import sys
import os

#set up parallel stuff
from IPython import parallel
try:
    c = parallel.Client()
    dv = c.direct_view()
    dv.execute('import GPy', block=True)
    dv.execute('import numpy as np', block=True)
    dv.execute('from classification1 import classification', block=True)
    DO_PARALLEL = True
except:
    DO_PARALLEL = False
    print "parallel init failed"

def par_map(f, *seq):
    if DO_PARALLEL:
        return dv.map(f, *seq, block=True)
    else:
        print "parallel diabled: this will be slow!"
        return map(f,*seq)

d = loadmat('benchmarks.mat')
dataset_names = d.keys()

#sort alphabetically
dataset_names = np.sort(np.array(dataset_names, dtype=np.str))

#define a function that can compare the methods
def compare(Xtrain, Ytrain, Xtest, Ytest):

    m = classification(Xtrain, Ytrain)
    try:
        m.optimize('bfgs')#, messages=0, bfgs_factor=1e8)
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    predictions = m.predict(Xtest)
    predictions = np.clip(predictions, 1e-6, 1.-1e-6)
    truth = Ytest.flatten()
    my_error = 1. - np.mean(truth==(predictions>0.5))
    my_nlp = - np.mean(truth*np.log(predictions) + (1-truth)*np.log(1-predictions))

    ##build an EP model, with the same link and kernel parameters
    link = GPy.likelihoods.noise_models.gp_transformations.Heaviside()
    lik = GPy.likelihoods.binomial(link)
    m_ep1 = GPy.models.GPClassification(Xtrain,likelihood=GPy.likelihoods.EP(Ytrain.reshape(-1,1), lik), kernel = m.kern.copy())
    try:
        m_ep1.update_likelihood_approximation()
        predictions = m_ep1.predict(Xtest)[0].flatten()
        predictions = np.clip(predictions, 1e-6, 1.-1e-6)
        EP_errorX = 1. - np.mean(truth==(predictions>0.5))
        EP_nlpX = - np.mean(truth*np.log(predictions) + (1-truth)*np.log(1-predictions))
    except:
        EP_errorX = np.nan
        EP_nlpX = np.nan

    #now optimize against the ep approximation to the marg. lik.
    m_ep2 = GPy.models.GPClassification(Xtrain,likelihood=GPy.likelihoods.EP(Ytrain.reshape(-1,1), lik), kernel = GPy.kern.rbf(Xtrain.shape[1]) + GPy.kern.white(Xtrain.shape[1]))
    try:
        m_ep2.pseudo_EM()
    except:
        #psuedo_EM failed...
        return my_error, my_nlp, EP_errorX, EP_nlpX, np.nan, np.nan, np.nan, np.nan
    predictions = m_ep2.predict(Xtest)[0].flatten()
    predictions = np.clip(predictions, 1e-6, 1.-1e-6)
    EP_error = 1. - np.mean(truth==(predictions>0.5))
    EP_nlp = - np.mean(truth*np.log(predictions) + (1-truth)*np.log(1-predictions))

    #now build varEP with the kern fixed to the EP solution
    k = m_ep2.kern.copy()
    k.constrain_fixed('')
    m2 = classification(Xtrain, Ytrain, k)
    m2.no_K_grads_please = True # don;t compute the gradient wrt kern. params to save time
    m2.optimize('bfgs', messages=0, bfgs_factor=1e7)

    predictions = m2.predict(Xtest)
    predictions = np.clip(predictions, 1e-6, 1.-1e-6)
    var_EP_errorX = 1. - np.mean(truth==(predictions>0.5))
    var_EP_nlpX = - np.mean(truth*np.log(predictions) + (1-truth)*np.log(1-predictions))

    return my_error, my_nlp, EP_errorX, EP_nlpX, EP_error, EP_nlp, var_EP_errorX, var_EP_nlpX


#loop through all the data...
for dn in dataset_names:
    if dn[:2]=='__':continue # first 3 keys are meta-data from the mat file
    if dn=='benchmarks': continue
    if dn=='image': continue # image takes forever
    if dn+'raw_results' in os.listdir('.'):
        print dn, 'is done already'
        continue
    print 'doing', dn

    #extract the data matrices from the structure. The extra [0,0] is a mystery to me!
    X = d[dn]['x'][0,0]
    Y = d[dn]['t'][0,0]
    Y = np.where(Y==1,1,0).flatten() # we use 1/0, not 1/-1, in a flat array
    train_inds = d[dn]['train'][0,0] -1 # subtract 1 to get offset (proper!) indexing
    test_inds = d[dn]['test'][0,0] -1 # subtract 1 to get offset (proper!) indexing

    #cut duplicates from the training set...
    train_inds = [np.unique(ti) for ti in train_inds]

    ##onyl do 20 folds, not 100 (for now) TODO: remove
    #train_inds = train_inds[:16]
    #test_inds = test_inds[:16]

    results = par_map(compare,
            [X[tr_i] for tr_i in train_inds],
            [Y[tr_i] for tr_i in train_inds],
            [X[te_i] for te_i in test_inds],
            [Y[te_i] for te_i in test_inds])

    results = np.array(results)
    np.savetxt(dn+'raw_results', results)

