from scipy.fft import fft as fft
from scipy.fft import ifft as ifft
import numpy as np

def freq2timeMidPoint(array,M,dt):
    ''' Fourier transform from (real) frequency domain
        to (real) time domain

        array: array of 2M points representing the
        the function to be Fourier-transformed. It is ASSUMED
        that the array is evaluated at the mid point of the underlying
        grid of frequency points (i.e: no evaluation at omega = 0).

        M: size of half of the frequency grid, as well as of the
        corresponding time-grid

        dt: spacing between succesive dimensionfull points in the resulting time grid.
        This means that, in the back of your head, dt has units of time. The corresponding 
        differential element of frequency is defined from this diferential time element so as to
        make this function the inverse of the time2freqMidPoint.   

        NOTE: All notation is inspired by Aravind's original code, so all credit for the inspiration of
        this function goes to him :).
    '''
    domega = np.pi/(M*dt)
    omegaMinusM = np.arange(2*M)-M
    t = np.arange(2*M)
    prefOut = (domega/(2*np.pi))*np.exp(2*np.pi*1j*(M-1/2)*t/(2*M))
    prefIn = np.exp(2*np.pi*1j*(omegaMinusM+1/2)*(M-1/2)/(2*M))

    return prefOut*fft(prefIn*array)

def freq2time(array,M,dt):
    ''' Fourier transform from (real) frequency domain
        to (real) time domain

        array: array of 2M points representing the
        the function to be Fourier-transformed, evaluated at the
        frequency array domain (symmetric around zero).

        M: size of half of the frequency grid, as well as of the
        corresponding time-grid

        dt: spacing between succesive dimensionfull points in the resulting time grid.
        This means that, in the back of your head, dt has units of time. The corresponding 
        differential element of frequency is defined from this diferential time element so as to
        make this function the inverse of the time2freq.   

        NOTE: All notation is inspired by Aravind's original code, so all credit for the inspiration of
        this function goes to him :).
    '''
    domega = np.pi/(M*dt)
    tMinusM = np.arange(2*M)-M
    omega = np.arange(2*M)
    prefOut = (domega/(2*np.pi))*np.exp(np.pi*1j*tMinusM)
    prefIn = np.exp(np.pi*1j*omega)

    return prefOut*fft(prefIn*array)


def time2freqMidPoint(array,M,dt):
    ''' Fourier transform from (real) frequency domain
        to (real) time domain

        array: array of 2M points representing the
        the function to be Fourier-transformed. It is ASSUMED
        that the array is evaluated at the mid point of the underlying
        grid of frequency points (i.e: no evaluation at omega = 0).

        M: size of half of the frequency grid, as well as of the
        corresponding time-grid

        dt: spacing between succesive dimensionfull points in the resulting time grid.
        This means that, in the back of your head, dt has units of time. 

        NOTE: All notation is inspired by Aravind's original code, so all credit for the inspiration of
        this function goes to him :).
    '''
    tMinusM = np.arange(2*M)-M
    omega = np.arange(2*M)
    prefOut = 2*M*dt*np.exp(-2*np.pi*1j*(M-1/2)*omega/(2*M))
    prefIn = np.exp(-2*np.pi*1j*(tMinusM+1/2)*(M-1/2)/(2*M))

    return prefOut*ifft(prefIn*array)

def time2freq(array,M,dt):
    ''' Fourier transform from (real) frequency domain
        to (real) time domain

        array: array of 2M points representing the
        the function to be Fourier-transformed.

        M: size of half of the frequency grid, as well as of the
        corresponding time-grid

        dt: spacing between succesive dimensionfull points in the resulting time grid.
        This means that, in the back of your head, dt has units of time. 

        NOTE: All notation is inspired by Aravind's original code, so all credit for the inspiration of
        this function goes to him :).
    '''
    omegaMinusM = np.arange(2*M)-M
    t = np.arange(2*M)
    prefOut = 2*M*dt*np.exp(-np.pi*1j*omegaMinusM)
    prefIn = np.exp(-np.pi*1j*t)

    return prefOut*ifft(prefIn*array)

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
