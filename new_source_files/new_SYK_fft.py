from scipy.fft import fft as fft
from scipy.fft import ifft as ifft
import scipy.integrate
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

def Freq2TimeBMidPoint(array, Nbig, beta, regulator = 0):
    ''' Fast fourier transform (from Bosonic matsubara frequency to Euclidean time).
    This routine is adapted to result in an array of values which approximate a function evaluated
    in the Euclidean time interval [0,beta]. Strictly speaking, the resulting indices of the array represent
    a discretization of the time interval in units of beta, so the resulting array is a list of points which 
    represent the function evaluated at the MID-POINT of N sub-intervals of the continous interval [0,1], so it
    is a discretization of the variable tau/beta.
    NOTE: Please introduce Nbig as even integer!
    '''
    TauOverBetaIndex = np.arange(Nbig,dtype=int)
    sumIndex= np.arange(Nbig,dtype=int)
    Mbig = Nbig//2
    prefOut = (1/beta)*1j*np.exp(np.pi*1j*TauOverBetaIndex)*np.exp(2*np.pi*1j*Mbig*regulator)
    prefIn = np.exp(-1j*np.pi*sumIndex/Nbig)*np.exp(-2*np.pi*1j*sumIndex*regulator)
    return prefOut * fft(prefIn*array)

def Time2FreqBMidPoint(array,Nbig,beta, regulator = 0, mess_around = False):
    '''
    Fast Fourier transform in Euclidean signature, which takes in
    a function with periodic boundary conditions in the Eucldiean time
    interval [0,beta] in the form of a discrete array, and returns an array
    evaluated at discrete array evaluated at integer points which represent
    bosonic Matsubara frequencies. This routine implements the mid-point rule,
    so the argument "array" should be assumed to be evaluated at the mid-point
    of each interval of the discretization of the interval [0,beta]. Also,
    the time has been re-scaled to be measured in units of beta.
    NOTE: The indexing of the resulting array starts from 0 and goes to Nbig-1,
    so in order to have a proper idea of the Matsubara frequencies associated to
    each value of the array, its list indexing should be shifted by N/2 to the left.
    '''
    BMatsubaraIndexShifted= np.arange(Nbig,step=1,dtype=int)-Nbig//2
    EuclideanTimeIndex = np.arange(Nbig)
    prefOut = beta*np.exp(np.pi*1j*BMatsubaraIndexShifted/Nbig)*np.exp(2*np.pi*1j*BMatsubaraIndexShifted*regulator)
    prefIn = np.exp(-1j*np.pi*EuclideanTimeIndex)
    result = prefOut * ifft(prefIn * array)
    if mess_around:
        result[Nbig//2] = (result[(Nbig//2)-1]+result[(Nbig//2)+1])/2
    return result

def Freq2TimeFMidPoint(array, Nbig, beta,regulator = 0):
    ''' Fast fourier transform (from Fermionic matsubara frequency to Euclidean time).
    This routine is adapted to result in an array of values which approximate a function evaluated
    in the Euclidean time interval [0,beta]. Strictly speaking, the resulting indices of the array represent
    a discretization of the time interval in units of beta, so the resulting array is a list of points which 
    represent the function evaluated at the MID-POINT of N sub-intervals of the continous interval [0,1], so it
    is a discretization of the variable tau/beta.
    NOTE: Please introduce Nbig as even integer!
    '''

    FMatsubaraIndex = np.arange(Nbig)
    EuclideanTimeIndex = np.arange(Nbig)
    prefOut = (1/beta) * 1j * np.exp(-1j*np.pi*EuclideanTimeIndex/Nbig) * np.exp(-1j * np.pi/(2*Nbig)) * np.exp(1j * np.pi *EuclideanTimeIndex) * np.exp(-1j*np.pi*regulator) * np.exp(1j*2*np.pi*regulator*(Nbig//2))
    prefIn = np.exp(-1j * np.pi * FMatsubaraIndex/Nbig) * np.exp(-1j*2*np.pi*FMatsubaraIndex*regulator)
    return prefOut * fft(prefIn * array)

def Time2FreqFMidPoint(array,Nbig,beta,regulator = 0, mess_around = False):
    '''
    Fast Fourier transform in Euclidean signature, which takes in
    a function with anti-periodic boundary conditions in the Eucldiean time
    interval [0,beta] in the form of a discrete array, and returns an array
    evaluated at discrete array evaluated at integer points which represent
    fermionic Matsubara frequencies. This routine implements the mid-point rule,
    so the argument "array" should be assumed to be evaluated at the mid-point
    of each interval of the discretization of the interval [0,beta]. Also,
    the time has been re-scaled to be measured in units of beta.
    NOTE: The indexing of the resulting array starts from 0 and goes to Nbig-1,
    so in order to have a proper idea of the Matsubara frequencies associated to
    each value of the array, its list indexing should be shifted by N/2 to the left.
    '''
    FMatsubaraIndexShifted = np.arange(Nbig,step=1,dtype=int)-Nbig//2
    EuclideanTimeIndex = np.arange(Nbig)
    prefOut = beta*np.exp(np.pi*1j*FMatsubaraIndexShifted/Nbig)*np.exp(np.pi*1j/(2*Nbig))*np.exp(1j*np.pi*(2*FMatsubaraIndexShifted+1)*regulator)
    prefIn = np.exp(np.pi*1j*EuclideanTimeIndex/Nbig)*np.exp(-np.pi*1j*EuclideanTimeIndex)
    result = prefOut * ifft(prefIn * array)
    if mess_around:
        result[Nbig//2] = (result[(Nbig//2)-1]+result[(Nbig//2)+1])/2
    return result

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

def simpson_integration_of_two_arrays(args):
    """
    routine that takes in a couple of arrays that are implicitly values of a function
    sampled on points of a real interval, and performs a Simpson rule integration. The
    argument args should be a tuple of three elements: integrand_1, integrand_2, integration_domain, in
    that order
    """
    integrand_1, integrand_2, integration_domain = args
    return [scipy.integrate.simpson(integrand_1,x=integration_domain),scipy.integrate.simpson(integrand_2,x=integration_domain)]