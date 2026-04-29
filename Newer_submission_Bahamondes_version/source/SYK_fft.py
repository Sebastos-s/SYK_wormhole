from scipy.fft import fft as fft
from scipy.fft import ifft as ifft
import numpy as np

#rotuines taken from Aravindh's original code, and modified when necessary.

def time2freq(ftau,M,dt): 
    '''
    Real time/frequency version
    '''
    pref = 2 * M * dt
    OmegaminusM = np.arange(2*M) - M
    tau = np.arange(2*M)
    prefexp = np.exp(-1j * np.pi * OmegaminusM)
    return prefexp * pref * ifft(np.exp(-1j*np.pi*tau)*ftau) 

def freq2time(fomega,M,dt):
    '''
    real frequency/time version
    '''
    dw = np.pi/(M*dt)
    pref = dw
    tauminusM = np.arange(2*M) - M
    omega = np.arange(2*M)
    prefexp = np.exp(1j*np.pi*tauminusM)
    return (0.5/np.pi) * prefexp * pref * fft(np.exp(1j*np.pi*omega)*fomega)

def fermidirac(arg, default = True):
    '''
    returns 1/(1+ exp(x))
    '''
    if default:
        return (1.0/(1.0 + np.exp(arg)))
    else:
        answer = 1
        if arg < 0:
            answer = 1.0/(1.0 + np.exp(arg))
        else: 
            answer = np.exp(-arg)/(1.0 + np.exp(-arg))
        return answer

def fermidirac_stable(x):
    '''
    a more stable version of Fermi-Dirac statistics
    '''
    return np.where(x < 0,1.0/(1.0+np.exp(x)),np.exp(-x)/(1.0+np.exp(-x)))
    
def boseeinstein(arg, default = True):
    '''
    returns 1/(exp(x)-1)
    Watch out for x=0
    '''
    if default:
        return (1.0/(np.exp(arg)-1))
    else:
        answer = 1
        if arg < 0:
            answer = 1.0/(np.exp(arg)-1)
        elif arg>0: 
            answer = np.exp(-arg)/(1.0 - np.exp(-arg))
        return answer

def boseeinstein_stable(x):
    '''
    a more stable version of Bose-Einstein statistics
    '''
    return np.where(x < 0,1.0/(-1.0+np.exp(x)),np.exp(-x)/(1.0-np.exp(-x)))
    
def RealGridMaker(M,T):
    '''
    returns an omega and t grid used in all the real time (I)FFT
    parameters:
    M : int - Large positive integer, size of the grid is 2M - 1 
    T : float - Upper cutoff on time 
    returns: 
    omega : real frequency grid 
    t : real time grid 
    '''
    dt = (2*T)/((2*M))
    t = dt * (np.arange(2*M) - M)
    dw = np.pi/(M*dt)
    omega = dw * (np.arange(2*M) - M)
    return omega,t