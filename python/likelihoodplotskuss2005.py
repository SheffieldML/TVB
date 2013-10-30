'''
Created on 12 Sep 2013

@author: maxz, James Hensman
'''
import numpy as np
from classification1 import classification as class1
from classification2 import classification as class2
import GPy, sys
from sklearn.cross_validation import train_test_split
import pylab as pb
import os
from pandas.core.frame import DataFrame
from scipy.io.matlab.mio import loadmat


def load_stats(dirname):
    data_train = None
    data_test = None
    for dirname,_,fnames in os.walk(dirname):
        for f in fnames:
            if f.endswith('npz'):
                d = np.load(os.path.join(dirname,f))
                seed = str(d['seed'])
                if data_train is None:
                    data_train = DataFrame(data=d['Z_EP'].flatten(),columns=[seed+"Z_EP"])
                else:
                    data_train[seed+"Z_EP"] = d['Z_EP'].flatten()
                if data_test is None:
                    data_test = DataFrame(data=d['pred_EP'].flatten(),columns=[seed+"pred_EP"])            
                else:
                    data_test[seed+"pred_EP"] = d['pred_EP'].flatten()
                data_train[seed+"Z_tVB"] = d['Z_tVB'].flatten()
                data_test[seed+"pred_tVB"] = d['pred_tVB'].flatten()
    #trs, tes = np.sqrt(data_train.shape[0]), np.sqrt(data_test.shape[0])
    return data_train, data_test

def plot_contour(name, l, a, Z, ax=None):
    if ax is None:
        fig = pb.figure(name)
        ax = fig.add_subplot(111)
    extent = [l.min(),l.max(),a.min(),a.max()]
    c = ax.contour(l,a,Z,colors='k', linestyles='solid')
    pb.clabel(c)
    ax.imshow(Z, extent=extent, origin='lower', interpolation='bilinear')
    ax.set_xlabel('$\log(l)$')
    ax.set_ylabel('$\log(a)$')
    ax.figure.tight_layout()    

def plot_max(Z, l, a, f, fig_name):
    d = Z.filter(like=f).mean(1).values.reshape(np.sqrt(Z.shape[0]),np.sqrt(Z.shape[0]))
    plot_contour(fig_name, l, a, d)

def grid(np):
    gridsize = 16 # x gridsipyze
    l, a = np.meshgrid(np.linspace(0, 5, gridsize), np.linspace(0, 6, gridsize))
    return gridsize, l, a

def save_plots(folder):
    for figname in ['EP', 'EP info', 'tVB', 'tVB info']:
        pb.figure(figname)
        pb.savefig(os.path.join(folder, "{}.pdf".format(figname)))

def plot_all(folder):
    pb.close('all')
    train, test = load_stats(folder)
    _, l, a = grid(np)
    plot_max(train, l, a, 'tVB', 'tVB')
    plot_max(train, l, a, 'EP', 'EP')
    plot_max(test, l, a, 'EP', 'EP info')
    plot_max(test, l, a, 'tVB', 'tVB info')
    save_plots(folder)

if __name__ == '__main__':
    pb.close('all')
    seed = np.random.randint(1e6)
    link_name = sys.argv[1]
    N = 200
    white = 2

    # read in the data:
    if 0:
        data = np.loadtxt("ionosphere.dat", str, delimiter=',')
        data_name = "ionosphere"
        X = np.array(data[:, 2:34], dtype=float)
        X = X - X.mean(0); X = X / X.std(0)
        labels = data[:, -1]
        Y = np.where(labels == 'b', 0, 1)[:, None].flatten()
    else:
        d = loadmat('benchmarks.mat')
        dataset_names = d.keys()
        data_name = "heart"
        #sort alphabetically
        dataset_names = np.sort(np.array(dataset_names, dtype=np.str))
        dn = 'heart'
        X = d[dn]['x'][0,0]
        Y = d[dn]['t'][0,0]
        Y = np.where(Y==1,1,0).flatten()
    
    name_format = 'white{}/{}/{}_N{}/seed{}.npz'
    get_name = lambda: name_format.format(str(white).replace('.', ''), data_name, link_name, N, seed)
    out_name = get_name()
    while os.path.exists(out_name):
        seed = np.random.randint(1e6)
        out_name = get_name()
    if not os.path.exists(os.path.dirname(out_name)):
        os.makedirs(os.path.dirname(out_name))
    np.savez(out_name)
    np.random.seed(seed)
    
    #cut some data out ( as kuss)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,train_size=N)
    
#     Xtrain = X[:N]
#     Ytrain = Y[:N]
# 
#     Xtest = X[N:]
#     Ytest = Y[N:]

    gridsize, l, a = grid(np)
    kern = GPy.kern.rbf(Xtrain.shape[1]) + GPy.kern.white(Xtrain.shape[1])

    #set up the tVB model
    m = class2(Xtrain, Ytrain, kern, link=link_name)
    m.tilted.do_entropy = False
    m.no_K_grads_please = True

    #set up the EP model
    if link_name == 'heaviside':
        link = GPy.likelihoods.noise_models.gp_transformations.Heaviside()
    else:
        link = GPy.likelihoods.noise_models.gp_transformations.Probit()
    lik = GPy.likelihoods.binomial(link)
    m_ep = GPy.models.GPClassification(Xtrain, Ytrain.reshape(-1,1), kernel=kern.copy())

    #loop!
    Z_tVB = np.zeros((gridsize, gridsize))
    Z_tVB_alt = np.zeros((gridsize, gridsize))
    Z_EP = np.zeros((gridsize, gridsize))
<<<<<<< HEAD
    pred_tVB = np.zeros((gridsize, gridsize))
    pred_EP = np.zeros((gridsize, gridsize))
        
=======
    def single_point(l,a, m, m_ep):
        #do the tVB model first
        m.constrain_fixed('rbf_len', np.exp(ll))
        m.constrain_fixed('rbf_var', np.exp(aa))
        m.constrain_fixed('white', 1e-6)
        m.randomize()
        m.optimize('bfgs', messages=0)#, bfgs_factor=1e20)
        #Z_tVB_alt[i,j] = m.alternative_log_likelihood()
            #Do EP
        m_ep._set_params(np.array([np.exp(aa), np.exp(ll), 1e-6]))
        m_ep.update_likelihood_approximation()

>>>>>>> ca0141651b63a029e773ecebbfec60d90e3102d4
    for i in range(gridsize):
        for j in range(gridsize):
            aa = a[i,j]
            ll = l[i,j]
            print "Doing point: {:.2} {:.2}".format(ll, aa)
<<<<<<< HEAD
            #do the tVB model first
            m.constrain_fixed('rbf_len', np.exp(ll))
            m.constrain_fixed('rbf_var', np.exp(aa))
            m.constrain_fixed('white', white)
            m.randomize()
            m.optimize('scg', messages=0, max_iters=2e4)#, bfgs_factor=1e20)
            #Z_tVB[i,j] =  m.alternative_log_likelihood()
            Z_tVB[i,j] =  m.log_likelihood()
            p = m.predict(Xtest)
            pred_tVB[i,j] = np.where(Ytest==1, np.log(p), np.log(1.-p)).mean()
            #Z_tVB_alt[i,j] = m.alternative_log_likelihood()

            #Do EP
            m_ep._set_params(np.array([np.exp(aa), np.exp(ll), white]))
            m_ep.update_likelihood_approximation()
            Z_EP[i,j] = m_ep.log_likelihood()
            p = m_ep.predict(Xtest)[0].flatten()
            pred_EP[i,j] = np.where(Ytest==1, np.log(p), np.log(1.-p)).mean()
            sys.stdout.flush()
    
    if 0:
        contours = np.r_[max(Z_EP.max(), Z_tVB.max()):max(Z_EP.max(), Z_tVB.max())-40:-2]
    
        plot_contour('tVB', l, a, Z_tVB)
        plot_contour('tVB info', l, a, pred_tVB)
        
        plot_contour('EP', l, a, Z_EP)
        plot_contour('EP info', l, a, pred_EP)
                
        #pb.figure('tVB_alt')
        #c = pb.contour(l,a,Z_tVB_alt, 10, color='k')
        #pb.imshow(Z_tVB_alt, extent=extent, origin='lower')
        #pb.clabel(c)
    
        
    np.savez(out_name, 
             Xtest=Xtest, Xtrain=Xtrain, Ytest=Ytest, Ytrain=Ytrain,
             Z_EP=Z_EP, pred_EP=pred_EP, Z_tVB=Z_tVB, pred_tVB=pred_tVB,
             seed=seed, l=l, a=a, gridsize=gridsize)
=======
            Z_EP[i,j] = m_ep.log_likelihood()
            Z_tVB[i,j] =  m.log_likelihood()


    pb.figure('tVB')
    c = pb.contour(l,a,Z_tVB, 10, colors='k', linestyles='solid')
    pb.clabel(c)
    pb.imshow(Z_tVB, extent=[0,5,0,6], origin='lower')

    #pb.figure('tVB_alt')
    #c = pb.contour(l,a,Z_tVB_alt, 10, color='k')
    #pb.imshow(Z_tVB_alt, extent=[0,5,0,6], origin='lower')
    #pb.clabel(c)

    pb.figure('EP')
    c = pb.contour(l,a,Z_EP, 10, colors='k', linestyles='solid')
    pb.imshow(Z_EP, extent=[0,5,0,6], origin='lower')
    pb.clabel(c)

>>>>>>> ca0141651b63a029e773ecebbfec60d90e3102d4
