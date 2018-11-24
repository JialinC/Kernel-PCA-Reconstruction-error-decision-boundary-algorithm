from scipy import exp
import numpy as np

def recerr(x,data,gamma,alpha,alphaKrow,sumalpha,Ksum):
  
   n = len(data)
   k = np.zeros(n)
   for j in np.arange(n):
       diff = x - data[j,:]
       k[j] = exp(-gamma*diff.dot(diff))
       
   #projections:
   f = k.dot(alpha) - sumalpha.dot(sum(k)/n - Ksum) - alphaKrow
   #reconstruction error:
   err = 1 - 2*sum(k)/n + Ksum - f.dot(f)
   return err
