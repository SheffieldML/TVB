import numpy as np
import pylab as pb
from scipy.io import loadmat
import GPy
from classification1 import classification
import sys

d = loadmat('benchmarks.mat')
dataset_names = d.keys()

#sort alphabetically
dataset_names = np.sort(np.array(dataset_names, dtype=np.str))

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

    #build a model for each of the train/test sets
    my_errors = []
    my_nlps = []
    EP_errors = []
    EP_nlps = []
    EPopt_errors = []
    EPopt_nlps = []
    my_errors2 = []
    my_nlps2 = []

    for tr_i, te_i in zip(train_inds, test_inds):
        m = classification(X[tr_i], Y[tr_i])
        m.optimize('bfgs')#, messages=0, bfgs_factor=1e8)

        predictions = m.predict(X[te_i])
        truth = Y[te_i]
        error = 1. - np.mean(truth==(predictions>0.5))
        nlp = - np.mean(truth*np.log(predictions) + (1-truth)*np.log(1-predictions))
        print 'error', error
        print 'nlp', nlp
        my_errors.append(error)
        my_nlps.append(nlp)

        ##build an EP model, with the same link and kernel parameters
        link = GPy.likelihoods.noise_models.gp_transformations.Heaviside()
        lik = GPy.likelihoods.binomial(link)
        m_ep1 = GPy.models.GPClassification(X[tr_i],likelihood=GPy.likelihoods.EP(Y[tr_i].reshape(-1,1), lik), kernel = m.kern.copy())
        m_ep1.update_likelihood_approximation()
        predictions = m_ep1.predict(X[te_i])[0].flatten()
        error = 1. - np.mean(truth==(predictions>0.5))
        nlp = - np.mean(truth*np.log(predictions) + (1-truth)*np.log(1-predictions))
        print 'EP error', error
        print 'EP nlp', nlp
        sys.stdout.flush()

        EP_errors.append(error)
        EP_nlps.append(nlp)

        #now optimize against the ep approximation to the marg. lik.
        m_ep2 = GPy.models.GPClassification(X[tr_i],likelihood=GPy.likelihoods.EP(Y[tr_i].reshape(-1,1), lik), kernel = GPy.kern.rbf(X.shape[1]) + GPy.kern.white(X.shape[1]))
        try:
            m_ep2.pseudo_EM()
        except:
            print 'psuedo_EM failed'
            failures += 1
            continue
        predictions = m_ep2.predict(X[te_i])[0].flatten()
        error = 1. - np.mean(truth==(predictions>0.5))
        nlp = - np.mean(truth*np.log(predictions) + (1-truth)*np.log(1-predictions))
        print 'EPopt error', error
        print 'EPopt nlp', nlp
        sys.stdout.flush()

        EPopt_errors.append(error)
        EPopt_nlps.append(nlp)

        #now build varEP with the kren fixed to the EP solution
        k = m_ep2.kern.copy()
        k.constrain_fixed('')
        m2 = classification(X[tr_i], Y[tr_i], k)
        m2.no_K_grads_please = True # don;t compute the gradient wrt kern. params to save time
        m2.optimize('bfgs', messages=0, bfgs_factor=1e7)

        predictions = m2.predict(X[te_i])
        error = 1. - np.mean(truth==(predictions>0.5))
        nlp = - np.mean(truth*np.log(predictions) + (1-truth)*np.log(1-predictions))
        print 'my2 error', error
        print 'my2 nlp', nlp
        print ""
        sys.stdout.flush()

        my_errors2.append(error)
        my_nlps2.append(nlp)

    #write the results to a file
    f = file(dn+'_results.txt','w')
    f.write('errors:\n')
    f.write('varEP:' + str(np.mean(my_errors)) +' +/- ' + str(np.std(my_errors)) + '\n')
    f.write('EP (var_EP kern):' + str(np.mean(EP_errors)) +' +/- ' + str(np.std(EP_errors)) + '\n')
    f.write('EP (pseudo-EM):' + str(np.mean(EPopt_errors)) +' +/- ' + str(np.std(EPopt_errors)) + '\n')
    f.write('varEP (EP kern):' + str(np.mean(my_errors2)) +' +/- ' + str(np.std(my_errors2)) + '\n')
    f.write('\nnlps:\n')
    f.write('varEP:' + str(np.mean(my_nlps)) +' +/- ' + str(np.std(my_nlps)) + '\n')
    f.write('EP (var_EP kern):' + str(np.mean(EP_nlps)) +' +/- ' + str(np.std(EP_nlps)) + '\n')
    f.write('EP (pseudo-EM):' + str(np.mean(EPopt_nlps)) +' +/- ' + str(np.std(EPopt_nlps)) + '\n')
    f.write('varEP (EP kern):' + str(np.mean(my_nlps2)) +' +/- ' + str(np.std(my_nlps2)) + '\n')
    f.write('\nfailures: ' + str(failures))
    f.close()

