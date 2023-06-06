"""
References: 
- D. Randall Wilson and Tony R. Martinez, Improved Heterogeneous Distance Functions, 
Journal of Artificial Intelligence Research, 1997 (https://arxiv.org/pdf/cs/9701101.pdf)
"""
from numba import jit
import numpy as np
import pandas as pd

class HVDM(object):
    def __init__(self):
        pass

    def normalized_vdm(self, data, target, valx, valy, a):
        """
        Compute the normalized difference between x and y for a nominal attribute
        
        Parameters:
        x,y: values for the variable a
        sigma: standard deviation of the variable a
        a: boolean saying if it is nominal or not
        """
        C = np.unique(target)

        mask_valx = data[:,a] == valx
        mask_valy = data[:,a] == valy
        nax = mask_valx.sum()
        nay = mask_valy.sum()
        total = 0
        
        for c in C:
            mask_valx_c = mask_valx & (target == c)
            mask_valy_c = mask_valy & (target == c)
            naxc = mask_valx_c.sum()
            nayc = mask_valy_c.sum()
            #if nax != 0 and nay != 0:
                #total += np.abs((naxc/nax) - (nayc/nay)) ** 2
            total += ((naxc/nax) - (nayc/nay)) ** 2
        return total#**0.5
    

    def normalized_diff(self, x, y, sigma):
        """
        Compute the normalized difference between x and y
        
        Parameters:
        x,y: values for the variable a
        sigma: standard deviation of the variable a
        """
        if 4*sigma == 0:
            return 1
        return abs(float(x)-float(y))/(4*sigma)


    def fit(self, data, target, nominal_attributes):
        n = data.shape[0]
        sigma = np.nanstd(data,axis=0)#, ddof=1) # sample -> ddof=1
        dist_matrix = np.zeros((n,n))
        for x in range(n-1):
            for y in range(x+1,n):
                    total = 0
                    for a in range(data.shape[1]):
                        x_a = data[x,a]
                        y_a = data[y,a]
                        if pd.isna(x_a) or pd.isna(y_a): # supports category type
                            #print('-----NAN')
                            d = 1
                        elif x_a == y_a:
                            #print('-----x=y')
                            d = 0
                        elif nominal_attributes[a]:
                            #print('-----NormVDM')
                            d = self.normalized_vdm(data,target,x_a,y_a,a)
                        else:
                            #print('-----NormDiff')
                            d = self.normalized_diff(x_a,y_a,sigma[a])
                        #print(d)
                        total += (d**2)
                    dist_matrix[x,y] = dist_matrix[y,x] = (total**0.5)
        
        return dist_matrix
