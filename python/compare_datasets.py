import numpy as np
import pylab as pb
from scipy.io import loadmat
import GPy
from classification1 import classification
import sys

#set up parallel stuff
from IPython import parallel
try:
    c = parallel.Client()
    dv = c.direct_view()
    dv.execute('import GPy')
    dv.execute('from classifiaction1 import parallel')
    DO_PARALLEL = True
except:
    DO_PARALLEL = False

DO_PARALLEL = False

def par_map(f, *seq):
    if DO_PARALLEL:
        return dv.map(f, *seq, block=True)
    else:
        return map(f,*seq)

d = loadmat('benchmarks.mat')
dataset_names = d.keys()

#sort alphabetically
dataset_names = np.sort(np.array(dataset_names, dtype=np.str))

#define a function that can compare the methods
def compare(Xtrain, Ytrain, Xtest, Ytest):
    m = classification(Xtrain, Ytrain)
    m.optimize('bfgs')#, messages=0, bfgs_factor=1e8)

    predictions = m.predict(Xtest)
    truth = Ytest.flatten()
    my_error = 1. - np.mean(truth==(predictions>0.5))
    my_nlp = - np.mean(truth*np.log(predictions) + (1-truth)*np.log(1-predictions))

    ##build an EP model, with the same link and kernel parameters
    link = GPy.likelihoods.noise_models.gp_transformations.Heaviside()
    lik = GPy.likelihoods.binomial(link)
    m_ep1 = GPy.models.GPClassification(X[tr_i],likelihood=GPy.likelihoods.EP(Y[tr_i].reshape(-1,1), lik), kernel = m.kern.copy())
    m_ep1.update_likelihood_approximation()
    predictions = m_ep1.predict(X[te_i])[0].flatten()
    EP1_error = 1. - np.mean(truth==(predictions>0.5))
    EP1_nlp = - np.mean(truth*np.log(predictions) + (1-truth)*np.log(1-predictions))

    #now optimize against the ep approximation to the marg. lik.
    m_ep2 = GPy.models.GPClassification(X[tr_i],likelihood=GPy.likelihoods.EP(Y[tr_i].reshape(-1,1), lik), kernel = GPy.kern.rbf(X.shape[1]) + GPy.kern.white(X.shape[1]))
    try:
        m_ep2.pseudo_EM()
    except:
        #psuedo_EM failed...
        return my_error, my_nlp, EP1_error, EP1_nlp, np.nan, np.nan, np.nan, np.nan
    predictions = m_ep2.predict(X[te_i])[0].flatten()
    EP_error = 1. - np.mean(truth==(predictions>0.5))
    EP_nlp = - np.mean(truth*np.log(predictions) + (1-truth)*np.log(1-predictions))

    #now build varEP with the kren fixed to the EP solution
    k = m_ep2.kern.copy()
    k.constrain_fixed('')
    m2 = classification(X[tr_i], Y[tr_i], k)
    m2.no_K_grads_please = True # don;t compute the gradient wrt kern. params to save time
    m2.optimize('bfgs', messages=0, bfgs_factor=1e7)

    predictions = m2.predict(X[te_i])
    var_EP1_error = 1. - np.mean(truth==(predictions>0.5))
    var_EP1_nlp = - np.mean(truth*np.log(predictions) + (1-truth)*np.log(1-predictions))

    return my_error, my_nlp, EP1_error, EP1_nlp, EP_error, EP_nlp, var_EP1_error, var_EP1_nlp


#loop through all the data...
for dn in dataset_names[3:]: # first 3 keys are meta-data from the mat file
    if dn=='benchmarks': continue
    if not (dn=='banana'): continue
    print dn
    failures = 0

    #extract the data matrices from the structure. The extra [0,0] is a mystery to me!
    X = d[dn]['x'][0,0]
    Y = d[dn]['t'][0,0]
    Y = np.where(Y==1,1,0).flatten() # we use 1/0, not 1/-1, in a flat array
    train_inds = d[dn]['train'][0,0] -1 # subtract 1 to get offset (proper!) indexing
    test_inds = d[dn]['test'][0,0] -1 # subtract 1 to get offset (proper!) indexing

    results = par_map(compare,
            [X[tr_i] for tr_i in train_inds],
            [Y[tr_i] for tr_i in train_inds],
            [X[te_i] for te_i in test_inds],
            [Y[te_i] for te_i in test_inds])

    results = np.array(results)
    np.savetxt(dn+'raw_results', results)
    #write the results to a file
    #f = file(dn+'_results.txt','w')
    #f.write('errors:\n')
    #f.write('varEP:' + str(np.mean(my_errors)) +' +/- ' + str(np.std(my_errors)) + '\n')
    #f.write('EP (var_EP kern):' + str(np.mean(EP_errors)) +' +/- ' + str(np.std(EP_errors)) + '\n')
    #f.write('EP (pseudo-EM):' + str(np.mean(EPopt_errors)) +' +/- ' + str(np.std(EPopt_errors)) + '\n')
    #f.write('varEP (EP kern):' + str(np.mean(my_errors2)) +' +/- ' + str(np.std(my_errors2)) + '\n')
    #f.write('\nnlps:\n')
    #f.write('varEP:' + str(np.mean(my_nlps)) +' +/- ' + str(np.std(my_nlps)) + '\n')
    #f.write('EP (var_EP kern):' + str(np.mean(EP_nlps)) +' +/- ' + str(np.std(EP_nlps)) + '\n')
    #f.write('EP (pseudo-EM):' + str(np.mean(EPopt_nlps)) +' +/- ' + str(np.std(EPopt_nlps)) + '\n')
    #f.write('varEP (EP kern):' + str(np.mean(my_nlps2)) +' +/- ' + str(np.std(my_nlps2)) + '\n')
    #f.write('\nfailures: ' + str(failures))
    #f.close()

