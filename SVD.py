import numpy as np
import scipy 
import pandas as pd


data = pd.read_csv("file.csv")

df = data.to_numpy()

def calculU(M): 
    B = np.dot(M, M.T) 
        
    eigenvalues, eigenvectors = np.linalg.eig(B) 
    ncols = np.argsort(eigenvalues)[::-1] 
    
    return eigenvectors[:,ncols] 
def calculVt(M): 
    B = np.dot(M.T, M)
        
    eigenvalues, eigenvectors = np.linalg.eig(B) 
    ncols = np.argsort(eigenvalues)[::-1] 
    
    return eigenvectors[:,ncols].T 

def calculSigma(M): 
    if (np.size(np.dot(M, M.T)) > np.size(np.dot(M.T, M))): 
        newM = np.dot(M.T, M) 
    else: 
        newM = np.dot(M, M.T) 
        
    eigenvalues, eigenvectors = np.linalg.eig(newM) 
    eigenvalues = np.sqrt(eigenvalues) 
    #Sorting in descending order as the svd function does 
    return eigenvalues[::-1] 

U = calculU(df) 
Sigma = calculSigma(df) 
Vt = calculVt(df)

print("-------------------U-------------------")
print(U)
print("\n--------------Sigma----------------")
print(Sigma)
print("\n-------------V transpose---------------")
print(Vt)

#Checking if we can remake the original matrix using U,Sigma,Vt

newSigma = np.zeros((2, 3))
newSigma[:2, :2] = np.diag(Sigma[:2])
print(A,"\n")
A_remake = (U @ newSigma @ Vt)
print(A_remake)
