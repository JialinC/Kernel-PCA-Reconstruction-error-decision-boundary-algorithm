import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from scipy import exp
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigs
from reconstruction_error import recerr

def kpcabound(data,sigma,numev,outlier,cstr):
    (n,d) = data.shape
    #n : number of data points
    #d : dimension of data points
    gamma = 0.5/(sigma*sigma)

    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    
    sq_dists = pdist(data, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)
    
    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)
    Krow = np.sum(K,0)/n
    Ksum = sum(Krow)/n
    
    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    print('extracting eigenvectors of K\n')
    eigvals, eigvecs = eigs(K,k=numev,which='LM') 
    eigvecs = eigvecs.real
    eigvals = np.diag(eigvals)
    eigvals = eigvals.real
    
    #residual variance:
    resvar = (np.trace(K)-np.trace(eigvals))
    ans = str(100*resvar/np.trace(K))
    print('residual variance relative to total variance in feature space: %s %%\n'%ans)

    #normalize alpha:
    eigvecs = eigvecs.dot(np.linalg.inv(np.sqrt(eigvals)))
    #compute some helper vectors:
    sumalpha = np.sum(eigvecs,0) #little different
    alphaKrow = Krow.dot(eigvecs)
    
    print('evaluating reconstruction error for all data points\n')
    err_or = np.zeros(n)

    for i in np.arange(n):
        x = data[i,:] #test point
        err_or[i] = recerr(x,data,gamma,eigvecs,alphaKrow,sumalpha,Ksum)

    serr = np.sort(err_or)
    maxerr = serr[n-1-outlier]
    
    print('computing recontruction error over data space\n')
    xymin = np.array([min(data[:,0]),min(data[:,1])])
    xymax = np.array([max(data[:,0]),max(data[:,1])])
    r = xymax-xymin

    offset = r*0.15 #provide some space around data points
    r = r*1.3
    xymin = xymin-offset
    xymax = xymax+offset

    m = 100 #choose even value
    steps = (1.0-1e-6)*r/(m-1)
    x = np.arange(xymin[0],xymax[0],steps[0])
    y = np.arange(xymin[1],xymax[1],2*steps[1])
    xv, yv = np.meshgrid(x, y)
    err = np.zeros((int(m/2),int(m)))

    for i in np.arange(int(m/2)):
        for ii in np.arange(int(m)):
            x = np.array([xv[0,ii],yv[i,0]] ).T
            err[i,ii] = recerr(x,data,gamma,eigvecs,alphaKrow,sumalpha,Ksum)
    
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.plot_surface(yv, xv, err, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #ax.view_init(70,270)
    
    #fig = plt.figure(figsize=(5,5))
    plt.plot(data[:,0],data[:,1],cstr)
    plt.contour(xv,yv,err,[maxerr])
    #plt.contour(data[:,1],data[:,0],err_or)


    
    
    
    
    
