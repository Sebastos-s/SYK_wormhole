from scipy.fft import fft as fft
from scipy.fft import ifft as ifft
from scipy.integrate import simpson
from scipy.optimize import root_scalar
from scipy.interpolate import CubicSpline
import itertools
import mpmath
from multiprocessing import Pool
import numpy as np
import warnings
from new_source_files.new_SYK_fft import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython.display import display
import time
from concurrent.futures import ProcessPoolExecutor
import os
from functools import partial
from fastcubicspline import FCS, NPointPoly

def _ellipk_scalar(x):
    return complex(mpmath.ellipk(x))

def _ellipe_scalar(x):
    return complex(mpmath.ellipe(x))

def elegant_G(z,executor = None):
    z = np.asarray(z,dtype = complex)
    m = 1/z**2

    #step that tryes to catch overflow of argument before passing it to the elliptic K
    bad_condition = (~np.isfinite(m)) | (np.abs(z) < 1e-14)
    m_safe_args = np.where(bad_condition, 0.0 + 0.0j, m).ravel()
    if executor is not None:
        result = np.asarray(list(executor.map(_ellipk_scalar,m_safe_args)),dtype = complex)
        result = np.where(bad_condition,0.0 + 0.0j,result)
    else:
        v_ellipk_scalar = np.vectorize(_ellipk_scalar,otypes = [complex])
        result = v_ellipk_scalar(m_safe_args)
        result = np.where(bad_condition,0.0 + 0.0j,result)
    return np.where(bad_condition,0.0+0.0j,(2/(np.pi*z))*result).reshape(z.shape)

def derivative_of_elegant_G(z,executor = None):
    z = np.asarray(z,dtype = complex)
    m = 1/z**2

    #step that tries to catch overflow of argument before passing it to the elliptic functions
    bad_condition = (~np.isfinite(m)) | (np.abs(z) < 1e-14)
    m_safe_args = np.where(bad_condition, 0.0 + 0.0j, m).ravel()
    if executor is not None:
        result = np.asarray(list(executor.map(_ellipe_scalar,m_safe_args)),dtype=complex)
        result = np.where(bad_condition,0.0 + 0.0j,result)
    else:
        v_ellipe_scalar = np.vectorize(_ellipe_scalar,otypes = [complex])
        result = v_ellipe_scalar(m_safe_args)
        result = np.where(bad_condition,0.0 + 0.0j,result)
    return (2/(np.pi*(1-z**2)))*result.reshape(z.shape)

def second_derivative_of_elegant_G(z,executor = None):
    z = np.asarray(z,dtype = complex)
    m = 1/z**2
    #step that tries to catch overflow of argument before passing it to the elliptic functions
    bad_condition = (~np.isfinite(m)) | (np.abs(z) < 1e-14)
    m_safe_args = np.where(bad_condition, 0.0 + 0.0j, m).ravel()
    if executor is not None:
        result1 = np.asarray(list(executor.map(_ellipe_scalar,m_safe_args)),dtype = complex)
        result2 = np.asarray(list(executor.map(_ellipk_scalar,m_safe_args)),dtype = complex)
        result1 = np.where(bad_condition,0.0+0.0j,result1)
        result2 = np.where(bad_condition,0.0+0.0j,result2)
    else:
        v_ellipk_scalar = np.vectorize(_ellipk_scalar,otypes = [complex])
        v_ellipe_scalar = np.vectorize(_ellipe_scalar,otypes = [complex])
        result1 = v_ellipe_scalar(m_safe_args)
        result2 = v_ellipk_scalar(m_safe_args)
        result1 = np.where(bad_condition,0.0+0.0j,result1)
        result2 = np.where(bad_condition,0.0+0.0j,result2)
    result = (2/(np.pi*z*((-1+z**2)**2)))*((3*(z**2)-1)*result1-((z**2)-1)*result2)
    return result

def rho2sigma2DNormal(rhoG,rhoD,M,dt,t,omega,v,g,beta,delta=1e-6,midPoint=False):
    '''
    Function that returns the fermionic and bosonic self-energies in the normal state of
    the 2D system: alpha = 1 (i.e: no superconductivity and no anomalous Green's functions in the
    Schwinger-Dyson equations). The function takes in arrays evaluated in real time that correspond
    to the fermionic and bosonic Green's functions, as well as parameters relevant to the discretization
    of the time grid (M, dt), and the arrays of time and frequency of appropriate size (t, omega). It also
    takes in the values of the equations' parameters (beta, g), and has a default value for the infinitesimal
    regulator that shows up when properly defining the analytical continuation of thermal-to-retarded Green's functions
    (delta). Finally, we implement an additional argument that allows the user to choose wether to use the 
    mid-point rule in the calculation of continous Fourier transforms via the discrete FFT (implemented in new_SYK_fft.py).
    
    NOTE: If the user chooses to use the midPoint rule, they must make sure that the arrays rhoG and rhoD are evaluated at the
    mid-points of the frequency and time grids; i.e: omega and t.
    '''
    if not midPoint:
        eta = np.pi/(M*dt)*(0.001)
        rhoGrev = np.flip(rhoG)
        rhoFpp = freq2time(rhoG * fermidirac_stable(beta*omega),M,dt)
        rhoFmp = freq2time(rhoG * fermidirac_stable(-1.*beta*omega),M,dt)
        rhoFpm = freq2time(rhoGrev * fermidirac_stable(beta*(omega)),M,dt)
        rhoFmm = freq2time(rhoGrev * fermidirac_stable(-1.*beta*omega),M,dt)
        rhoBpp = freq2time(rhoD * boseeinstein_stable(beta*(omega+eta)),M,dt)
        rhoBmp = freq2time(rhoD * boseeinstein_stable(-1.*beta*(omega+eta)),M,dt)

        SigmaInTime = -1.0j * np.exp(-np.abs(delta*t)) * np.heaviside(t,0) * ((v**2)*(rhoFpp+rhoFmp)-(g**2)*(rhoFpp*rhoBpp-rhoFmp*rhoBmp))
        Sigma = time2freq(SigmaInTime,M,dt)

        PiInTime = -2.0j * np.exp(-np.abs(delta*t)) * np.heaviside(t,0) * (g**2) * (rhoFpp*rhoFpm - rhoFmp*rhoFmm)
        Pi = time2freq(PiInTime,M,dt)
        #changed both overall signs!
        
    else:
        #rhoG and rhoD are assumed to be evaluated at mid-points of omega and t grids.
        domega = np.pi/(M*dt)
        rhoGrevMidPoint = np.flip(rhoG)
        rhoFppMidPoint = freq2timeMidPoint(rhoG * fermidirac(beta*(omega+domega/2)),M,dt)
        rhoFmpMidPoint = freq2timeMidPoint(rhoG * fermidirac(-1.*beta*(omega+domega/2)),M,dt)
        rhoFpmMidPoint = freq2timeMidPoint(rhoGrevMidPoint * fermidirac(beta*(omega+domega/2)),M,dt)
        rhoFmmMidPoint = freq2timeMidPoint(rhoGrevMidPoint * fermidirac(-1.*beta*(omega+domega/2)),M,dt)
        rhoBppMidPoint = freq2timeMidPoint(rhoD * boseeinstein(beta*(omega+domega/2)),M,dt)
        rhoBmpMidPoint = freq2timeMidPoint(rhoD * boseeinstein(-1.*beta*(omega+domega/2)),M,dt)

        SigmaInTimeMidPoint = -1.0j * np.exp(-np.abs(delta*(t+dt/2))) * np.heaviside(t+dt/2,1) * ((v**2)*(rhoFppMidPoint +\
                            rhoFmpMidPoint)-(g**2)*(rhoFppMidPoint*rhoBppMidPoint-rhoFmpMidPoint*rhoBmpMidPoint))
        Sigma = time2freqMidPoint(SigmaInTimeMidPoint,M,dt)

        PiInTimeMidPoint = -2.0j * np.exp(-np.abs(delta*(t+dt/2))) * np.heaviside(t+dt/2,1) * (g**2) * (rhoFppMidPoint*rhoFpmMidPoint - \
                           rhoFmpMidPoint*rhoFmmMidPoint)
        Pi = time2freqMidPoint(PiInTimeMidPoint,M,dt)
        #changed both overall signs!
    return [Sigma,Pi]

def rho2sigma0DNormal(rhoG,rhoD,M,dt,t,omega,g,beta,delta=1e-6,midPoint=False):
    '''
    Function that returns the fermionic and bosonic self-energies in the normal state of
    the 0D (quantum dot) system: alpha = 1 (i.e: no superconductivity and no anomalous Green's functions in the
    Schwinger-Dyson equations). The function takes in arrays evaluated in real time that correspond
    to the fermionic and bosonic Green's functions, as well as parameters relevant to the discretization
    of the time grid (M, dt), and the arrays of time and frequency of appropriate size (t, omega). It also
    takes in the values of the equations' parameters (beta, g), and has a default value for the infinitesimal
    regulator that shows up when properly defining the analytical continuation of thermal-to-retarded Green's functions
    (delta).
    
    NOTE: If the user chooses to use the midPoint rule, they must make sure that the arrays rhoG and rhoD are evaluated at the
    mid-points of the frequency and time grids; i.e: omega and t.
    '''

    if not midPoint:
        eta = np.pi/(M*dt)*(0.001)
        rhoGrev = np.flip(rhoG)
        rhoFpp = freq2time(rhoG * fermidirac_stable(beta*omega),M,dt)
        rhoFmp = freq2time(rhoG * fermidirac_stable(-1.0*beta*omega),M,dt)
        rhoFpm = freq2time(rhoGrev * fermidirac_stable(beta*omega),M,dt)
        rhoFmm = freq2time(rhoGrev * fermidirac_stable(-1.0*beta*omega),M,dt)
        rhoBpp = freq2time(rhoD * boseeinstein_stable(beta*(omega+eta)),M,dt)
        rhoBmp = freq2time(rhoD * boseeinstein_stable(-1.0*beta*(omega+eta)),M,dt)
    
        SigmaInTime = (rhoFmp*rhoBmp - rhoFpp*rhoBpp) * np.exp(-np.abs(delta*t)) * np.heaviside(t,0)
        Sigma = -1j*(g**2) * time2freq(SigmaInTime,M,dt)
    
        PiInTime = (rhoFpp*rhoFpm - rhoFmp*rhoFmm) * np.exp(-np.abs(delta*t)) * np.heaviside(t,0)
        Pi = -2*1j*(g**2) * time2freq(PiInTime,M,dt)
        #changed both signs!
    else:
        #rhoG and rhoD are assumed to be evaluated at mid-points of omega and t grids.
        domega = np.pi/(M*dt)
        rhoGrevMidPoint = np.flip(rhoG)
        rhoFppMidPoint = freq2timeMidPoint(rhoG * fermidirac_stable(beta*(omega+domega/2)),M,dt)
        rhoFmpMidPoint = freq2timeMidPoint(rhoG * fermidirac_stable(-1.0*beta*(omega+domega/2)),M,dt)
        rhoFpmMidPoint = freq2timeMidPoint(rhoGrevMidPoint * fermidirac_stable(beta*(omega+domega/2)),M,dt)
        rhoFmmMidPoint = freq2timeMidPoint(rhoGrevMidPoint * fermidirac_stable(-1.0*beta*(omega+domega/2)),M,dt)
        rhoBppMidPoint = freq2timeMidPoint(rhoD * boseeinstein_stable(beta*(omega+domega/2)),M,dt)
        rhoBmpMidPoint = freq2timeMidPoint(rhoD * boseeinstein_stable(-1.0*beta*(omega+domega/2)),M,dt)

        SigmaInTimeMidPoint = (rhoFmpMidPoint*rhoBmpMidPoint - rhoFppMidPoint*rhoBppMidPoint)\
                            * np.exp(-np.abs(delta*(t+dt/2))) * np.heaviside(t+dt/2,0)
        Sigma = -1j*(g**2) * time2freqMidPoint(SigmaInTimeMidPoint,M,dt)
    
        PiInTimeMidPoint = (rhoFppMidPoint*rhoFpmMidPoint - rhoFmpMidPoint*rhoFmmMidPoint)\
                            * np.exp(-np.abs(delta*(t+dt/2))) * np.heaviside(t+dt/2,0)
        Pi = -2j*(g**2) * time2freqMidPoint(PiInTimeMidPoint,M,dt)
        #changed both signs!
    return [Sigma, Pi]

def rho2EuclideanGD2DNormal(rhoG,rhoD,N_half,omega,beta,executor = None):
    """
    Routine that calcualtes numerically the thermal bosonic and fermionic Green's functions
    from their relation to the spectral density of states. This routine can be run in parallel by passing the
    argument executor.
    """
    n = np.arange(2 * N_half)[:,np.newaxis] #[[0],[1],...] ==> (2N_half,1)
    omega_row = omega[np.newaxis,:] #[[omega[0],omega[1],omega[2],...] ==> (1,2M)

    #we create the integrand, watching out for 0/0 or NaN (inf) indeterminations
    denom_D = 1j*(2*np.pi*(n-N_half)/beta) - omega_row #(2Nhalf,2M)
    denom_G = 1j*(np.pi*(2*(n-N_half)+1)/beta) - omega_row #(2Nhalf,2M)

    safe_denom_G = np.where(np.abs(denom_G) < 1e-15,1.0+0.0j,denom_G)
    safe_denom_D = np.where(np.abs(denom_D) < 1e-15,1.0+0.0j,denom_D)
    integrandForDThermal = (1/(2*np.pi)) * rhoD/safe_denom_D #(2Nhalf,2M)
    integrandForGThermal = (1/(2*np.pi)) * rhoG/safe_denom_G #(2Nhalf,2M)

    tasks = [(integrandForGThermal[i],integrandForDThermal[i],omega) for i in range(2*N_half)]

    if executor is not None:
        results = np.asarray(list(executor.map(simpson_integration_of_two_arrays,tasks)))
    else:
        results = np.asarray(list(map(simpson_integration_of_two_arrays,tasks)))

    GThermalMatsubara, DThermalMatsubara = np.asarray(zip(*results))
    
    return [GThermalMatsubara,DThermalMatsubara]

def thermalDyson2DNormal(sigmaMatsubara,piMatsubara,M,v,g,mu,J,m02,beta,eta = 1e-6,executorFlag = None,spline = False):
    """
    Routine that calculates the bosonic and fermionic thermal Green's functions using the euclidean-signature Dyson equations.
    The function returns both propagators in Euclidean time, NOT in Matsubara frequency space. Notice that 2 * M must equal
    the size of the arrays sigmaMatsubara and piMatsubara.
    """
    if not spline:
        fermion_Matsubara_freqs = np.asarray([np.pi * (2*(n - M)+1)/beta for n in range(2 * M)])
        boson_Matsubara_freqs = np.asarray([2*np.pi * (n - M)/beta for n in range(2 * M)])

        GMatsubara = (0.25) * elegant_G(0.25*(1j*fermion_Matsubara_freqs + mu - sigmaMatsubara),executor = executorFlag)
        DMatsubara = (0.25/J) * elegant_G((0.25/J) * (boson_Matsubara_freqs**2 + m02 + 4*J - piMatsubara),executor = executorFlag)
    
        GTau = Freq2TimeFMidPoint(GMatsubara, 2*M, beta, regulator = eta)
        DTau = Freq2TimeBMidPoint(DMatsubara, 2*M, beta, regulator = eta)
        return [GTau, DTau]
    else:
        #M > 100
        inner_range_half = range(M-100, M+100, 2)
        outer_range = list(range(0,M-100)) + list(range(M+100,2*M)) 

        INTERIOR_fermion_Matsubara_freqs = np.asarray([np.pi * (2*(n - M)+1)/beta for n in inner_range_half])  
        INTERIOR_boson_Matsubara_freqs = np.asarray([2*np.pi * (n - M)/beta for n in inner_range_half])        
        
        INTERIOR_GMatsubara = (0.25) * elegant_G(0.25*(1j*INTERIOR_fermion_Matsubara_freqs + mu - sigmaMatsubara[M-100:M+100:2]),executor = executorFlag)
        INTERIOR_DMatsubara = (0.25/J) * elegant_G((0.25/J) * (INTERIOR_boson_Matsubara_freqs**2 + m02 + 4*J - piMatsubara[M-100:M+100:2]),executor = executorFlag)

        GMatsubaraInterpol = FCS(INTERIOR_fermion_Matsubara_freqs[0],INTERIOR_fermion_Matsubara_freqs[-1], INTERIOR_GMatsubara)
        DMatsubaraInterpol = FCS(INTERIOR_boson_Matsubara_freqs[0],INTERIOR_boson_Matsubara_freqs[-1], INTERIOR_DMatsubara)

        INTERIOR_fermion_Matsubara_freqs = np.asarray([np.pi * (2*(n - M)+1)/beta for n in range(M-100,M+100)])  
        INTERIOR_boson_Matsubara_freqs = np.asarray([2*np.pi * (n - M)/beta for n in range(M-100,M+100)])    
        
        INTERIOR_GMatsubara = GMatsubaraInterpol(INTERIOR_fermion_Matsubara_freqs)
        INTERIOR_DMatsubara = DMatsubaraInterpol(INTERIOR_boson_Matsubara_freqs)

        EXTERIOR_fermion_Matsubara_freqs = np.asarray([np.pi * (2*(n - M)+1)/beta for n in outer_range])
        EXTERIOR_boson_Matsubara_freqs = np.asarray([2*np.pi * (n - M)/beta for n in outer_range])

        EXTERIOR_GMatsubara = (0.25) * elegant_G(0.25*(1j*EXTERIOR_fermion_Matsubara_freqs + mu - sigmaMatsubara[outer_range]),executor = executorFlag)
        EXTERIOR_DMatsubara = (0.25/J) * elegant_G((0.25/J) * (EXTERIOR_boson_Matsubara_freqs**2 + m02 + 4*J - piMatsubara[outer_range]),executor = executorFlag)

        GMatsubara = np.concatenate([EXTERIOR_GMatsubara[0:M-100],INTERIOR_GMatsubara,EXTERIOR_GMatsubara[M-100:]])
        DMatsubara = np.concatenate([EXTERIOR_DMatsubara[0:M-100],INTERIOR_DMatsubara,EXTERIOR_DMatsubara[M-100:]])
        GTau = Freq2TimeFMidPoint(GMatsubara, 2*M, beta, regulator = eta)
        DTau = Freq2TimeBMidPoint(DMatsubara, 2*M, beta, regulator = eta)
        return [GTau, DTau]

def Dav_rho2sigma0DNormal(rhoG,rhoD,M,t,g,beta,BMf,delta=1e-6):
    '''
    Direct implementation of Davide's email
    '''
    dt = t[2]-t[1]
    fdplus,fdminus,beplus,beminus = BMf
    ADt = freq2time(rhoD,M,dt)
    aGt = freq2time(rhoG * fdplus, M,dt)
    AGt = freq2time(rhoG,M,dt)
    aDt = freq2time(rhoD * beplus, M,dt)
    #aGtm = freq2time(rhoG * fdminus, M,dt)
    aDtm = freq2time(rhoD * beminus, M,dt)

    argSigma = (ADt * aGt - AGt * np.conj(aDt)) * np.heaviside(t,0)*np.exp(-delta*np.abs(t))
    Sigma = -1j*(g**2)*time2freq(argSigma,M,dt)

    argPi = (AGt * np.conj(aGt) - np.conj(AGt) * aGt) * np.heaviside(t,0)*np.exp(-delta*np.abs(t))
    Pi = 2j*(g**2)*time2freq(argPi,M,dt)
    
    return [Sigma,Pi]

def realDyson_eq2DNormal(mu,SigmaRet,s,J,PiRet,omega,domega,midPoint=False,delta=1e-6,executorFlag = None):
    if midPoint==False:
        GRet = 0.25*elegant_G(0.25*(omega+1.0j*delta+mu-SigmaRet),executor = executorFlag)
        DRet = (0.25/J)*elegant_G((-(omega+1.0j*delta)**2+s+4*J-PiRet)/(4*J),executor = executorFlag)
    else:
        #arrays of retarded self-energies are assumed to be evaluated on midpoints
        GRet = 0.25*elegant_G(0.25*(omega+domega/2+1.0j*delta+mu-SigmaRet),executor = executorFlag)
        DRet = (0.25/J)*elegant_G((-(omega+domega/2+1.0j*delta)**2+s+4*J-PiRet)/(4*J),executor = executorFlag)
    return [GRet,DRet] 

def fixed_length_function_2DNormal_euclidean(M,J,beta,gamma,eta,executorFlag,m02,piMatsubara, spline = False):
    '''
    Function that returns D(tau = 0)-1/gamma, so that we can later implement the optimization routine
    that minimizes this object as a function of m02 (s).
    '''
    if not spline:
        boson_Matsubara_freqs = np.asarray([2*np.pi * (n - M)/beta for n in range(2 * M)])
        Dloc = (0.25/J) * elegant_G((0.25/J) * (boson_Matsubara_freqs**2 + m02 + 4*J - piMatsubara),executor = executorFlag)
        Dloc = Freq2TimeBMidPoint(Dloc,2 * M,beta,regulator = eta)
        function_to_minimize = Dloc[0]-1/gamma
        return function_to_minimize
    else:
        #M > 100
        inner_range_half = range(M-100, M+100, 2)
        outer_range = list(range(0,M-100)) + list(range(M+100,2*M)) 
        INTERIOR_boson_Matsubara_freqs = np.asarray([2*np.pi * (n - M)/beta for n in inner_range_half])

        INTERIOR_Dloc = (0.25/J) * elegant_G((0.25/J) * (INTERIOR_boson_Matsubara_freqs**2 + m02 + 4*J - piMatsubara[M-100:M+100:2]),executor = executorFlag)

        DlocInterpol = FCS(INTERIOR_boson_Matsubara_freqs[0], INTERIOR_boson_Matsubara_freqs[-1], INTERIOR_Dloc)

        INTERIOR_boson_Matsubara_freqs = np.asarray([2*np.pi * (n - M)/beta for n in range(M-100,M+100)])
        INTERIOR_Dloc = DlocInterpol(INTERIOR_boson_Matsubara_freqs)

        EXTERIOR_boson_Matsubara_freqs = np.asarray([2*np.pi * (n - M)/beta for n in outer_range])
        EXTERIOR_Dloc = (0.25/J) * elegant_G((0.25/J) * (EXTERIOR_boson_Matsubara_freqs**2 + m02 + 4*J - piMatsubara[outer_range]),executor = executorFlag)

        Dloc = np.concatenate([EXTERIOR_Dloc[0:M-100],INTERIOR_Dloc,EXTERIOR_Dloc[M-100:]])
        Dloc = Freq2TimeBMidPoint(Dloc,2 * M,beta,regulator = eta)
        function_to_minimize = Dloc[0]-1/gamma
        return function_to_minimize

def fixed_length_function_2DNormal_real(beta,J,M,domega,midPoint,delta,gamma,executorFlag,m02,omega,pi):
    '''
    Function that returns D(t = 0)-1/gamma, so that we can later implement the optimization routine
    that minimizes this object as a function of m02 (s).
    '''
    if midPoint==False:
        Dretloc = (0.25/J)*elegant_G((-(omega+1.0j*delta)**2+m02+4*J-pi)/(4*J),executor = executorFlag)
        rhoB = -2*np.imag(Dretloc)
        function_to_minimize = freq2time(-rhoB * boseeinstein_stable(beta * omega),M,np.pi/(domega*M))[M] - 1/gamma
        return np.real(function_to_minimize)
    else:
        #arrays of retarded self-energies are assumed to be evaluated on midpoints
        DRet = (0.25/J)*elegant_G((-(omega+domega/2+1.0j*delta)**2+m02+4*J-pi)/(4*J),executor = executorFlag)
        rhoB = -2*np.imag(DRet)
        function_to_minimize = freq2timeMidPoint(rhoB * boseeinstein(beta * (omega + domega/2)),M,np.pi/(domega*M))[M] + 1/gamma
        return np.real(function_to_minimize)

def derivative_of_fixed_length_function_2DNormal_euclidean(M,J,beta,gamma,eta,executorFlag,m02,piMatsubara,spline = False):
    '''
    Function that returns d D(t = 0)/ds.
    '''
    if not spline:
        boson_Matsubara_freqs = np.asarray([2*np.pi * (n - M)/beta for n in range(2 * M)])
        derivativeMatsubara = (1/(4*J)**2)*derivative_of_elegant_G((boson_Matsubara_freqs**2+m02+4*J-piMatsubara)/(4*J),executor = executorFlag)
        derivativeTau = Freq2TimeBMidPoint(derivativeMatsubara,2 * M,beta,regulator=eta)
        return derivativeTau[0]
    else:
        #M > 101
        inner_range_half = range(M-100, M+100, 2)
        outer_range = list(range(0,M-100)) + list(range(M+100, 2*M))
        INTERIOR_boson_Matsubara_freqs = np.asarray([2*np.pi * (n - M)/beta for n in inner_range_half]) 

        INTERIOR_derivativeMatsubara = (1/(4*J)**2)*derivative_of_elegant_G((INTERIOR_boson_Matsubara_freqs**2+m02+4*J-piMatsubara[M-100:M+100:2])/(4*J),executor = executorFlag)

        derivativeMatsubaraInterpol = FCS(INTERIOR_boson_Matsubara_freqs[0], INTERIOR_boson_Matsubara_freqs[-1], INTERIOR_derivativeMatsubara)

        INTERIOR_boson_Matsubara_freqs = np.asarray([2*np.pi * (n - M)/beta for n in range(M-100,M+100)])
        INTERIOR_derivativeMatsubara = derivativeMatsubaraInterpol(INTERIOR_boson_Matsubara_freqs)

        EXTERIOR_boson_Matsubara_freqs = np.asarray([2*np.pi * (n - M)/beta for n in outer_range])
        EXTERIOR_derivativeMatsubara = (1/(4*J)**2)*derivative_of_elegant_G((EXTERIOR_boson_Matsubara_freqs**2+m02+4*J-piMatsubara[outer_range])/(4*J),executor = executorFlag)

        derivativeMatsubara = np.concatenate([EXTERIOR_derivativeMatsubara[0:M-100],INTERIOR_derivativeMatsubara,EXTERIOR_derivativeMatsubara[M-100:]])
        derivativeTau = Freq2TimeBMidPoint(derivativeMatsubara,2 * M,beta,regulator=eta)
        return derivativeTau[0]
        
def derivative_of_fixed_length_function_2DNormal_real(J,M,domega,midPoint,delta,gamma,executorFlag,m02,omega,pi):
    '''
    Function that returns d D(t = 0)/ds.
    '''
    if midPoint:
        derivativeOmega = (1/(4*J)**2)*derivative_of_elegant_G((-(omega + domega/2 + 1j*delta)**2+m02+4*J-pi)/(4*J),executor = executorFlag)
        derivativeTime = freq2timeMidPoint(derivativeOmega,M,np.pi/(M*domega))
        return derivativeTime[M]
    else:
        derivativeOmega = (1/(4*J)**2)*derivative_of_elegant_G((-(omega + 1j*delta)**2+m02+4*J-pi)/(4*J),executor = executorFlag)
        derivativeTime = freq2time(derivativeOmega,M,np.pi/(M*domega))
        return derivativeTime[M]
#FIX OR DELETE!

def second_derivative_of_fixed_length_function_2DNormal_euclidean(M,J,beta,gamma,eta,executorFlag,m02,piMatsubara, spline = False):
    '''
    Function that returns d^2 D(t = 0)/ds^2.
    '''
    if not spline:
        boson_Matsubara_freqs = np.asarray([2*np.pi * (n - M)/beta for n in range(2 * M)])
        secondderivativeMatsubara = (1/(4*J)**3)*second_derivative_of_elegant_G((boson_Matsubara_freqs**2+m02+4*J-piMatsubara)/(4*J),executor = executorFlag)
        secondderivativeTau = Freq2TimeBMidPoint(secondderivativeMatsubara,2 * M,beta,regulator=eta)
        return secondderivativeTau[0]
    else:
        #M > 101
        inner_range_half = range(M-100, M+100, 2)
        outer_range = list(range(0,M-100)) + list(range(M+100, 2*M))
        INTERIOR_boson_Matsubara_freqs = np.asarray([2*np.pi * (n - M)/beta for n in inner_range_half]) 

        INTERIOR_secondderivativeMatsubara = (1/(4*J)**3)*second_derivative_of_elegant_G((INTERIOR_boson_Matsubara_freqs**2+m02+4*J-piMatsubara[M-100:M+100:2])/(4*J),executor = executorFlag)

        secondderivativeMatsubaraInterpol = FCS(INTERIOR_boson_Matsubara_freqs[0], INTERIOR_boson_Matsubara_freqs[-1], INTERIOR_secondderivativeMatsubara)

        INTERIOR_boson_Matsubara_freqs = np.asarray([2*np.pi * (n - M)/beta for n in range(M-100,M+100)])
        INTERIOR_secondderivativeMatsubara = secondderivativeMatsubaraInterpol(INTERIOR_boson_Matsubara_freqs)
        
        EXTERIOR_boson_Matsubara_freqs = np.asarray([2*np.pi * (n - M)/beta for n in outer_range])
        EXTERIOR_secondderivativeMatsubara = (1/(4*J)**3)*second_derivative_of_elegant_G((EXTERIOR_boson_Matsubara_freqs**2+m02+4*J-piMatsubara[outer_range])/(4*J),executor = executorFlag)
        
        secondderivativeMatsubara = np.concatenate([EXTERIOR_secondderivativeMatsubara[0:M-100],INTERIOR_secondderivativeMatsubara,EXTERIOR_secondderivativeMatsubara[M-100:]])
        secondderivativeTau = Freq2TimeBMidPoint(secondderivativeMatsubara,2 * M,beta,regulator=eta)
        return secondderivativeTau[0]
    
def second_derivative_of_fixed_length_function_2DNormal_real(J,M,domega,midPoint,delta,gamma,executorFlag,m02,omega,pi):
    '''
    Function that returns d^2 D(t = 0)/ds^2.
    '''
    if midPoint:
        secondderivativeOmega = (1/(4*J)**3)*second_derivative_of_elegant_G((-(omega + domega/2 + 1j*delta)**2+m02+4*J-pi)/(4*J),executor = executorFlag)
        secondderivativeTime = freq2timeMidPoint(secondderivativeOmega,M,np.pi/(M*domega))
        return secondderivativeTime[M]
    else:
        secondderivativeOmega = (1/(4*J)**3)*second_derivative_of_elegant_G((-(omega + 1j*delta)**2+m02+4*J-pi)/(4*J),executor = executorFlag)
        secondderivativeTime = freq2time(secondderivativeOmega,M,np.pi/(M*domega))
        return secondderivativeTime[M]
#FIX OR DELETE!

def YSYK_0DNormaliterator(GRomega,DRomega,grid,pars,beta,err=1e-5,ITERMAX=150,eta=1e-6,epsilon = 0.01,slowly = False,midPointFlag=False):
    '''
    This function returns a numerical profile of the Green's functions and self-energies for bosons and fermions
    in the quantum dot YSYK model in the normal phase (with time-reversal symmetry storngly broken, and therefore
    no possibility of superconductivity). It uses an Anderson iteration algorithm to approach the best self-consistent
    solution to the Schwinger-Dyson equations. The function takes as arguments the arrays GRomega and DRomega, which
    correspond to the seed Green's functions evaluated in the frequency grid. Such frequency grid is provided in the
    argument pars, which is of the form: grid = [M,omega,t], with M being the half-size of the frequency and time grids,
    which are themselves called omega and t. The argument pars = [g,mu,m0] is a list which contains the value of the
    coupling constant g, the chemical potential mu, and the bare boson mass m02. The argument beta, of course, is the
    system's inverse temperature. The err kwarg is the error within which we expect the correct solution to lie after
    a sufficient number of iterations. The kwarg eta is the value of the infinitesimal regulator that we use in the
    computation of the retarded self-energies of the system.
    '''
    M,omega,t = grid
    g,mu,m02 = pars
    dt = t[1]-t[0]
    dw = omega[1]-omega[0]
    itern = 0

    if slowly:
        fig, axs = plt.subplots(nrows = 3,ncols = 2,figsize=(14,7))

    diff = 1.
    diffG,diffD = (1.0,1.0)
    xG, xD = 0.5 - epsilon, 0.5 - epsilon
    xG2, xD2 = 0.5 - epsilon, 0.5 - epsilon

    flag = True
    diffs_list = []
    fdplus = np.array([fermidirac(beta*omegaval, default = False) for omegaval in omega])
    fdminus = np.array([fermidirac(-1.0*beta*omegaval, default = False) for omegaval in omega])
    beplus = np.array([boseeinstein(beta*omegaval, default = False) for omegaval in omega])
    beminus = np.array([boseeinstein(-1.0*beta*omegaval, default = False) for omegaval in omega])
    BMf = [fdplus, fdminus, beplus, beminus]

    while (diff>err and itern<ITERMAX and flag):   
        itern += 1 
        if itern == ITERMAX:
            warnings.warn('WARNING: ITERMAX reached for beta = ' + str(beta))   
        diffoldG,diffoldD = (diffG,diffD)
        if itern == 1:
            GRoldomega,DRoldomega = (1.0*GRomega, 1.0*DRomega)
        else:
            GRold2omega, DRold2omega = (1.0*GRoldomega, 1.0*DRoldomega)
            GRoldomega,DRoldomega = (1.0*GRomega, 1.0*DRomega)
    
        rhoG = -2*np.imag(GRomega)
        rhoD = -2*np.imag(DRomega)
        
        SigmaOmega,PiOmega = rho2sigma0DNormal(rhoG,rhoD,M,dt,t,omega,g,beta,delta=0,midPoint=midPointFlag)

        if np.imag(SigmaOmega[M] > 0) :
            print('Violation of causality : Pole of Gomega in UHP for beta = ' + str(beta))
            exit()
        if itern == 1:
            if midPointFlag == False:
                GRomega = (2*epsilon)/(omega + 1j*eta + mu - SigmaOmega) + (1-2*epsilon)*GRoldomega
                DRomega = (2*epsilon)/(-1.0*(omega+1j*eta)**2 + m02 - PiOmega) + (1-2*epsilon)*DRoldomega
            else:
                GRomega = (2*epsilon)/(omega + dw/2 + 1j*eta + mu - SigmaOmega) + (1-2*epsilon)*GRoldomega
                DRomega = (2*epsilon)/(-1.0*(omega + dw/2+1j*eta)**2 + m02 - PiOmega) + (1-2*epsilon)*DRoldomega
        else:
            if midPointFlag == False:
                GRomega = (2*epsilon)/(omega + 1j*eta + mu - SigmaOmega) + xG2*GRold2omega + xG*GRoldomega
                DRomega = (2*epsilon)/(-1.0*(omega+1j*eta)**2 + m02 - PiOmega) + xD2*DRold2omega + xD*DRoldomega
            else:
                GRomega = (2*epsilon)/(omega + dw/2 + 1j*eta + mu - SigmaOmega) + xG2*GRold2omega + xG*GRoldomega
                DRomega = (2*epsilon)/(-1.0*(omega + dw/2 +1j*eta)**2 + m02 - PiOmega) + xD2*DRold2omega + xD*DRoldomega                

        #causality constraint
        if midPointFlag:
            GRt = freq2timeMidPoint(GRomega,M,dt)
            DRt = freq2timeMidPoint(DRomega,M,dt)
            GRt[:M-1] = 0  
            DRt[:M-1] = 0
            GRomega = time2freqMidPoint(GRt,M,dt)
            DRomega = time2freqMidPoint(DRt,M,dt)
        else:
            GRt = freq2time(GRomega,M,dt)
            DRt = freq2time(DRomega,M,dt)
            GRt[:M] = 0  
            DRt[:M] = 0
            GRomega = time2freq(GRt,M,dt)
            DRomega = time2freq(DRt,M,dt)

        diffG = np.sqrt(np.sum((np.abs(GRomega-GRoldomega))**2))
        diffD = np.sqrt(np.sum((np.abs(DRomega-DRoldomega))**2))

        threshold = np.sqrt(err)
       
        diff = 0.5*(diffG+diffD)
        diffs_list.append(diff)
        diffG,diffD = diff,diff

        if slowly:
            for row in range(3):
                for col in range(2):
                    axs[row,col].clear()
                    axs[row,col].grid(True)
            axs[0,0].set_xlim(-2,2)
            axs[0,1].set_xlim(-2,2)
            axs[1,0].set_xlim(-2,2)
            axs[1,1].set_xlim(-2,2)
            axs[2,0].set_xlim(-1,1)
            axs[2,1].set_xlim(0,2)
            axs[0,0].set_xlabel(r'$\omega$')
            axs[0,1].set_xlabel(r'$\omega$')
            axs[0,0].set_ylabel(r'$\Re(\Sigma(\omega))$')
            axs[0,1].set_ylabel(r'$\Re(\Pi(\omega))$')
            axs[1,0].set_xlabel(r'$\omega$')
            axs[1,1].set_xlabel(r'$\omega$')
            axs[1,0].set_ylabel(r'$\Im(\Sigma(\omega))$')
            axs[1,1].set_ylabel(r'$\Im(\Pi(\omega))$')
            axs[2,0].set_xlabel(r'$\log(\omega)$')
            axs[2,1].set_xlabel(r'$\omega$')
            axs[2,0].set_ylabel(r'$\log(\rho_F(\omega))$')
            axs[2,1].set_ylabel(r'$-\rho_B(\omega)$')

            axs[0,0].plot(omega,np.real(SigmaOmega))
            axs[0,1].plot(omega,np.real(PiOmega))

            axs[1,0].plot(omega,np.imag(SigmaOmega))
            axs[1,1].plot(omega,np.imag(PiOmega))

            axs[2,0].plot(omega,-np.imag(GRomega))
            axs[2,1].plot(omega,np.imag(DRomega))

            axs[0,0].legend(loc = 'upper left')
            axs[0,1].legend(loc = 'upper left')
            axs[1,0].legend(loc = 'upper left')
            axs[1,1].legend(loc = 'upper left')

            fig.suptitle(rf'$\beta$ = {beta}. g = {g}. Iteration = {itern}. (diffG , diffD) = ({diffG} , {diffD})')

            display.clear_output(wait=True)
            display.display(fig)
            time.sleep(0.0001)

    INFO = (itern, diff)
    if slowly:
        plt.close()
    return (GRomega,DRomega, INFO)

def YSYK_2DNormaliteratorNoRest(GRomega,DRomega,grid,pars,beta,err = 1e-5,ITERMAX = 150,eta = 1e-6,epsilon = 0.01,slowly = False,midPointFlag = False,executorChoice = None):
    '''
    This function returns a numerical profile of the Green's functions and self-energies for bosons and fermions
    in the two-dimensional YSYK model in the normal phase (with time-reversal symmetry storngly broken, and therefore
    no possibility of superconductivity). It uses an Anderson iteration algorithm to approach the best self-consistent
    solution to the Schwinger-Dyson equations. The function takes as arguments the arrays GRomega and DRomega, which
    correspond to the seed Green's functions evaluated in the frequency grid. Such frequency grid is provided in the
    argument pars, which is of the form: grid = [M,omega,t], with M being the half-size of the frequency and time grids,
    which are themselves called omega and t. The argument pars = [g,v,mu,m02,J] is a list which contains the value of the
    coupling constant g, the chemical potential mu, the bare boson mass m02, the fermion potential v, and the interchange constat J for bosons, all in units of t. 
    The argument beta, of course, is the
    system's inverse temperature (also in units of t). The err kwarg is the error within which we expect the correct solution to lie after
    a sufficient number of iterations. The kwarg eta is the value of the infinitesimal regulator that we use in the
    computation of the retarded self-energies of the system.
    '''

    M,omega,t = grid
    g,v,mu,m02,J = pars
    dt = t[1]-t[0]
    dw = omega[1]-omega[0]
    itern = 0

    if slowly:
        fig, axs = plt.subplots(nrows = 3,ncols = 2,figsize=(14,7))
        plot_display = display(fig, display_id = True)
    diff = 1.
    diffG,diffD = (1.0,1.0)
    xG, xD = 0.5 - epsilon, 0.5 - epsilon
    xG2, xD2 = 0.5 - epsilon, 0.5 - epsilon

    flag = True
    diffs_list = []
    fdplus = np.array([fermidirac(beta*omegaval, default = False) for omegaval in omega])
    fdminus = np.array([fermidirac(-1.0*beta*omegaval, default = False) for omegaval in omega])
    beplus = np.array([boseeinstein(beta*omegaval, default = False) for omegaval in omega])
    beminus = np.array([boseeinstein(-1.0*beta*omegaval, default = False) for omegaval in omega])
    BMf = [fdplus, fdminus, beplus, beminus]

    while (diff>err and itern<ITERMAX and flag):   
        itern += 1 
        if itern == ITERMAX:
            warnings.warn('WARNING: ITERMAX reached for beta = ' + str(beta))   
        diffoldG,diffoldD = (diffG,diffD)
        if itern == 1:
            GRoldomega,DRoldomega = (1.0*GRomega, 1.0*DRomega)
        else:
            GRold2omega, DRold2omega = (1.0*GRoldomega, 1.0*DRoldomega)
            GRoldomega,DRoldomega = (1.0*GRomega, 1.0*DRomega)
        rhoG = -2*np.imag(GRomega)
        rhoD = -2*np.imag(DRomega)

        #spectral functions and Green's functions depend implicitly on crystal momentum,
        #but the SD eqs. imply that these are uniform in such variables, so the only real
        #dependence is on frequency
        SigmaOmega,PiOmega = rho2sigma2DNormal(rhoG,rhoD,M,dt,t,omega,v,g,beta,delta=eta,midPoint=midPointFlag)
        if np.imag(SigmaOmega[M] > 0) :
            print('Violation of causality : Pole of Gomega in UHP for beta = ' + str(beta))
            exit()
        ti = time.time()
        GRDyson,DRDyson = realDyson_eq2DNormal(mu,SigmaOmega,m02,J,PiOmega,omega,dw,midPoint=midPointFlag,delta=eta,executorFlag=executorChoice)
        print(f'Time required for calculating Gr and Dr: {time.time()-ti}s.')
        if itern == 1:
            GRomega = (2*epsilon)*GRDyson + (1-2*epsilon)*GRoldomega
            DRomega = (2*epsilon)*DRDyson + (1-2*epsilon)*DRoldomega
        else:
            GRomega = (2*epsilon)*GRDyson+ xG2*GRold2omega + xG*GRoldomega
            DRomega = (2*epsilon)*DRDyson + xD2*DRold2omega + xD*DRoldomega
        #causality constraint
        if midPointFlag:
            GRt = freq2timeMidPoint(GRomega,M,dt)
            DRt = freq2timeMidPoint(DRomega,M,dt)
            GRt[:M-1] = 0  
            DRt[:M-1] = 0
            GRomega = time2freqMidPoint(GRt,M,dt)
            DRomega = time2freqMidPoint(DRt,M,dt)
        else:
            GRt = freq2time(GRomega,M,dt)
            DRt = freq2time(DRomega,M,dt)
            GRt[:M] = 0  
            DRt[:M] = 0
            GRomega = time2freq(GRt,M,dt)
            DRomega = time2freq(DRt,M,dt)
        

        diffG = np.sqrt(np.sum((np.abs(GRomega-GRoldomega))**2))
        diffD = np.sqrt(np.sum((np.abs(DRomega-DRoldomega))**2))

        threshold = np.sqrt(err)
       
        diff = 0.5*(diffG+diffD)
        diffs_list.append(diff)
        diffG,diffD = diff,diff

        if slowly:
            for row in range(3):
                for col in range(2):
                    axs[row,col].clear()
                    axs[row,col].grid(True)
            axs[0,0].set_xlim(-10,10)
            axs[0,1].set_xlim(-10,10)
            axs[1,0].set_xlim(-10,10)
            axs[1,1].set_xlim(-10,10)
            axs[2,0].set_xlim(-10,10)
            axs[2,1].set_xlim(-10,10)
            axs[0,0].set_xlabel(r'$\omega$')
            axs[0,1].set_xlabel(r'$\omega$')
            axs[0,0].set_ylabel(r'$\Re(\Sigma(\omega))$')
            axs[0,1].set_ylabel(r'$\Re(\Pi(\omega))$')
            axs[1,0].set_xlabel(r'$\omega$')
            axs[1,1].set_xlabel(r'$\omega$')
            axs[1,0].set_ylabel(r'$-\Im(\Sigma(\omega))$')
            axs[1,1].set_ylabel(r'$-\Im(\Pi(\omega))$')
            axs[2,0].set_xlabel(r'$\omega$')
            axs[2,1].set_xlabel(r'$\omega$')
            axs[2,0].set_ylabel(r'$\rho_F(\omega)$')
            axs[2,1].set_ylabel(r'$-\rho_B(\omega)$')

            axs[0,0].plot(omega,np.real(SigmaOmega))
            axs[0,1].plot(omega,np.real(PiOmega))

            axs[1,0].plot(omega,-np.imag(SigmaOmega))
            axs[1,1].plot(omega,-np.imag(PiOmega))

            axs[2,0].plot(omega,-np.imag(GRomega))
            axs[2,1].plot(omega,np.imag(DRomega))

            #axs[0,0].legend(loc = 'upper left')
            #axs[0,1].legend(loc = 'upper left')
            #axs[1,0].legend(loc = 'upper left')
            #axs[1,1].legend(loc = 'upper left')

            fig.suptitle(rf'$\beta$ = {beta}. g = {g}. Iteration = {itern}. (diffG , diffD) = ({diffG} , {diffD})')

            #display.clear_output(wait=True)
            #display.display(fig)
            plot_display.update(fig)
            time.sleep(0.0001)

    INFO = (itern, diff)
    if slowly:
        plt.close(fig)
    return (GRomega,DRomega, INFO) 

def YSYK_2DNormaliterator(GRomega,DRomega,grid,pars,beta,err = 1e-5,ITERMAX = 150,eta = 1e-6,epsilon = 0.01,slowly = False,midPointFlag = False,executorChoice = None,optimization = 'scipy', workingPrec = 15):
    '''
    This function returns a numerical profile of the Green's functions and self-energies for bosons and fermions
    in the two-dimensional YSYK model in the normal phase (with time-reversal symmetry storngly broken, and therefore
    no possibility of superconductivity). It uses an Anderson iteration algorithm to approach the best self-consistent
    solution to the Schwinger-Dyson equations. The function takes as arguments the arrays GRomega and DRomega, which
    correspond to the seed Green's functions evaluated in the frequency grid. Such frequency grid is provided in the
    argument pars, which is of the form: grid = [M,omega,t], with M being the half-size of the frequency and time grids,
    which are themselves called omega and t. The argument pars = [g,v,mu,m02seed,J,gamma] is a list which contains the value of the
    coupling constant g, the chemical potential mu, the seed bare boson mass m02seed, the fermion potential v, and the interchange constat J for bosons, all in units of t.
    The parameter gamma is the fixed length constraint that needs to be imposed on the retarded propagator of bosons. 
    The argument beta, of course, is the
    system's inverse temperature (also in units of t). The err kwarg is the error within which we expect the correct solution to lie after
    a sufficient number of iterations. The kwarg eta is the value of the infinitesimal regulator that we use in the
    computation of the retarded self-energies of the system.
    '''

    M,omega,t = grid
    g,v,mu,m02,J,gamma = pars
    dt = t[1]-t[0]
    dw = omega[1]-omega[0]
    itern = 0

    if slowly:
        fig, axs = plt.subplots(nrows = 4,ncols = 2,figsize=(14,9),layout = 'constrained')
        axInset = inset_axes(axs[3,1],width = '40%',height = '35%',loc = 'upper center')
        plot_display = display(fig, display_id = True) 
    diff = 1.
    diffG,diffD = (1.0,1.0)
    diffm02 = 1
    xG, xD = 0.5 - epsilon, 0.5 - epsilon
    xG2, xD2 = 0.5 - epsilon, 0.5 - epsilon

    flag = True
    diffs_list = []
    restriction_values_list = []
    M_values_list = []
    #fdplus = np.array([fermidirac(beta*omegaval, default = False) for omegaval in omega])
    #fdminus = np.array([fermidirac(-1.0*beta*omegaval, default = False) for omegaval in omega])
    #beplus = np.array([boseeinstein(beta*omegaval, default = False) for omegaval in omega])
    #beminus = np.array([boseeinstein(-1.0*beta*omegaval, default = False) for omegaval in omega])
    #BMf = [fdplus, fdminus, beplus, beminus]

    while (diff>err and itern<ITERMAX and flag):   
        itern += 1 
        if itern == ITERMAX:
            warnings.warn('WARNING: ITERMAX reached for beta = ' + str(beta))   
        diffoldG,diffoldD = (diffG,diffD)
        if itern == 1:
            GRoldomega,DRoldomega = (1.0*GRomega, 1.0*DRomega)
        else:
            GRold2omega, DRold2omega = (1.0*GRoldomega, 1.0*DRoldomega)
            GRoldomega,DRoldomega = (1.0*GRomega, 1.0*DRomega)
        rhoG = -2*np.imag(GRomega)
        rhoD = -2*np.imag(DRomega)

        #spectral functions and Green's functions depend implicitly on crystal momentum,
        #but the SD eqs. imply that these are uniform in such variables, so the only real
        #dependence is on frequency
        SigmaOmega,PiOmega = rho2sigma2DNormal(rhoG,rhoD,M,dt,t,omega,v,g,beta,delta=eta,midPoint=midPointFlag)
        if np.imag(SigmaOmega[M] > 0):
            print('Violation of causality : Pole of Gomega in UHP for beta = ' + str(beta))
            exit()

        #first we impose the fixed length restriction
        print(f'Current value of m02 = ',m02,'\n')
        if itern == 1 or diffm02 > err:
            print('Starting new optimization round:\n')
            oldm02 = m02
            m02_optimization_func = lambda s: fixed_length_function_2DNormal_real(beta,J,M,dw,midPointFlag,eta,gamma,executorChoice,s,omega,PiOmega)
            if optimization == 'scipy':
                root_finding = root_scalar(m02_optimization_func, x0 = m02, method = 'newton')
                m02 = root_finding.root
                print('Optimization completed with m02 =',m02,'\n')
                print('Value of function evaluated at root:',m02_optimization_func(m02))
            else:
                m02 = mpmath.findroot(m02_optimization_func, (0,m02+m02/2), method = 'secant',verbose = True,dps = workingPrec)
                print('Optimization completed with m02 =',m02,'\n')
                print('Value of function evaluated at root:',m02_optimization_func(m02))
            diffm02 = np.abs(m02-oldm02)
        restriction_values_list.append(m02_optimization_func(float(m02)))
        M_thermal = m02 - PiOmega[M]
        M_values_list.append(M_thermal)

        GRDyson,DRDyson = realDyson_eq2DNormal(mu,SigmaOmega,float(m02),J,PiOmega,omega,dw,midPoint=midPointFlag,delta=eta,executorFlag = executorChoice)

        if itern == 1:
            GRomega = (2*epsilon)*GRDyson + (1-2*epsilon)*GRoldomega
            DRomega = (2*epsilon)*DRDyson + (1-2*epsilon)*DRoldomega
        else:
            GRomega = (2*epsilon)*GRDyson+ xG2*GRold2omega + xG*GRoldomega
            DRomega = (2*epsilon)*DRDyson + xD2*DRold2omega + xD*DRoldomega
        #causality constraint
        if midPointFlag:
            GRt = freq2timeMidPoint(GRomega,M,dt)
            DRt = freq2timeMidPoint(DRomega,M,dt)
            GRt[:M-1] = 0  
            DRt[:M-1] = 0
            GRomega = time2freqMidPoint(GRt,M,dt)
            DRomega = time2freqMidPoint(DRt,M,dt)
        else:
            GRt = freq2time(GRomega,M,dt)
            DRt = freq2time(DRomega,M,dt)
            GRt[:M-1] = 0  
            DRt[:M-1] = 0
            GRomega = time2freq(GRt,M,dt)
            DRomega = time2freq(DRt,M,dt)
        

        diffG = np.sqrt(np.sum((np.abs(GRomega-GRoldomega))**2))
        diffD = np.sqrt(np.sum((np.abs(DRomega-DRoldomega))**2))

        threshold = np.sqrt(err)
       
        diff = 0.5*(diffG+diffD)
        diffs_list.append(diff)
        diffG,diffD = diff,diff

        if slowly:
            for row in range(4): 
                for col in range(2):
                    axs[row,col].clear()
                    axs[row,col].grid(True)
            axInset.clear()
            axInset.grid(True)
            #axs[0,0].set_xlim(fermion_Matsubara_freqs[M-M//5],fermion_Matsubara_freqs[M+(M//5)])
            #axs[0,1].set_xlim(boson_Matsubara_freqs[M-M//5],boson_Matsubara_freqs[M+(M//5)])
            #axs[1,0].set_xlim(fermion_Matsubara_freqs[M-M//5],fermion_Matsubara_freqs[M+(M//5)])
            #axs[1,1].set_xlim(boson_Matsubara_freqs[M-M//5],boson_Matsubara_freqs[M+(M//5)])
            #axs[2,0].set_xlim(fermion_Matsubara_freqs[M-M//5],fermion_Matsubara_freqs[M+(M//5)])
            #axs[2,1].set_xlim(boson_Matsubara_freqs[M-M//5],boson_Matsubara_freqs[M+(M//5)])
            axs[0,0].set_xlabel(r'$\omega$')
            axs[0,1].set_xlabel(r'$\omega$')
            axs[0,0].set_ylabel(r'$\Re(\Sigma_R(\omega))$')
            axs[0,1].set_ylabel(r'$\Re(\Pi_R(\omega))$')
            axs[1,0].set_xlabel(r'$\omega$')
            axs[1,1].set_xlabel(r'$\omega$')
            axs[1,0].set_ylabel(r'$\Im(\Sigma_R(\omega))$')
            axs[1,1].set_ylabel(r'$\Im(\Pi_R(\omega))$')
            axs[2,0].set_xlabel(r'$\omega$')
            axs[2,1].set_xlabel(r'$\omega$')
            axs[2,0].set_ylabel(r'$\Im(G_R(\omega))$')
            axs[2,1].set_ylabel(r'$\Im(D_R(\omega))$')
            axs[3,0].set_xlabel('iteration') 
            axs[3,1].set_xlabel('iteration') 
            axs[3,0].set_ylabel(r'$\left|\mathcal{D}(\tau=0)-1/\gamma\right|$') 
            axs[3,0].set_yscale('log')
            axs[3,1].set_ylabel(r'$M^2(T) = m_0^2-\Pi(\omega = 0)$') 
        
            axs[0,0].plot(omega,np.real(SigmaOmega),color='black',markersize = 2)
            axs[0,1].plot(omega,np.real(PiOmega),color='black',markersize = 2)
        
            axs[1,0].plot(omega,np.imag(SigmaOmega),color='black',markersize = 2)
            axs[1,1].plot(omega,np.imag(PiOmega),color='black',markersize = 2)
        
            axs[2,0].plot(omega,np.imag(GRomega),color='black',markersize = 2)
            axs[2,1].plot(omega,np.imag(DRomega),color='black',markersize = 2)
        
            axs[3,0].plot(np.arange(1,itern+1,step=1,dtype=int),np.abs(np.asarray(restriction_values_list)),'-o',color = 'blue',markersize=3)
            axs[3,1].plot(np.arange(1,itern+1,step=1,dtype=int),np.real(np.array(M_values_list,dtype=complex)),'-o',color = 'red',markersize=3,label = r'$\Re(.)$')
        
            axInset.plot(np.arange(1,itern+1,step=1,dtype=int),np.abs(np.imag(np.array(M_values_list,dtype=complex))),'-o',color = 'orange',markersize=3,label = r'$\Im(.)$')
            axInset.set_yscale('log')
        
            fig.suptitle(rf'$\beta$ = {beta}. g = {g}. Iteration = {itern}. ($diff_G$ , $diff_D$) = ({diffG} , {diffD})')
        
            #display.clear_output(wait=True)
            #display.display(fig)
            plot_display.update(fig)
            time.sleep(0.0001)

    INFO = (itern, diff)
    if slowly:
        plt.close()
    return [SigmaOmega,PiOmega,GRomega,DRomega,m02,M_thermal,INFO]  

def YSYK_2DNormalSuperiteratorEuclideanNoRest(sigmaMatsubara,piMatsubara,N_half,pars,beta,err = 1e-5,ITERMAX = 150,eta = 1e-6,epsilon = 0.01,slowly = False,executorChoice = None):
    '''
    This function returns a numerical profile of the thermal self-energies for bosons and fermions
    in the two-dimensional YSYK model in the normal phase (with time-reversal symmetry storngly broken, and therefore
    no possibility of superconductivity). The function also keeps track of the thermal Green's functions. It uses an Anderson iteration algorithm to approach the best self-consistent
    solution to the Schwinger-Dyson equations. The function takes as arguments the arrays GRomega and DRomega, which
    correspond to the seed Green's functions evaluated in the Matsubara frequency grid. Such Matsubara frequencies are constructed
    from the parameter N_half, which is half the size of the set of Matsubara frequencies we will consider. The argument pars = [g,v,mu,m02,J] is a list which contains the value of the
    coupling constant g, the chemical potential mu, the bare boson mass m02, the fermion potential v, and the interchange constat J for bosons, all in units of t. 
    The argument beta, of course, is the
    system's inverse temperature (also in units of t). The err kwarg is the error within which we expect the correct solution to lie after
    a sufficient number of iterations. The kwarg eta is the value of the infinitesimal regulator that we use in the
    computation of the retarded self-energies of the system.
    '''

    M = N_half
    g,v,mu,m02,J = pars
    itern = 0

    fermion_Matsubara_freqs = np.asarray([np.pi * (2*(n - M)+1)/beta for n in range(2 * M)])
    boson_Matsubara_freqs = np.asarray([2*np.pi * (n - M)/beta for n in range(2 * M)])

    if slowly:
        fig, axs = plt.subplots(nrows = 3,ncols = 2,figsize=(14,7))
        plot_display = display(fig, display_id = True) 
    diff = 1.
    diffSigma,diffPi = (1.0,1.0)
    xSigma, xPi = 0.5 - epsilon, 0.5 - epsilon
    xSigma2, xPi2 = 0.5 - epsilon, 0.5 - epsilon

    flag = True
    diffs_list = []

    while (diff>err and itern<ITERMAX and flag):   
        itern += 1 
        if itern == ITERMAX:
            warnings.warn('WARNING: ITERMAX reached for beta = ' + str(beta))   
        diffoldSigma,diffoldPi = (diffSigma,diffPi)
        if itern == 1:
            sigmaoldMatsubara,pioldMatsubara = (1.0*sigmaMatsubara, 1.0*piMatsubara)
        else:
            sigmaold2Matsubara, piold2Matsubara = (1.0*sigmaoldMatsubara, 1.0*pioldMatsubara)
            sigmaoldMatsubara,pioldMatsubara = (1.0*sigmaMatsubara, 1.0*piMatsubara)

        
        GTau, DTau = thermalDyson2DNormal(sigmaMatsubara,piMatsubara,M,v,g,mu,J,m02,beta,eta = eta,executorFlag = executorChoice)
        if itern == 1:
            sigmaMatsubara = (2*epsilon)*Time2FreqFMidPoint((g**2)*GTau*DTau + (v**2)*GTau,2*M,beta) + (1-2*epsilon)*sigmaoldMatsubara
            piMatsubara = (2*epsilon)*Time2FreqBMidPoint(-2*(g**2)*GTau*np.flip(GTau),2*M,beta) + (1-2*epsilon)*pioldMatsubara
        else:
            sigmaMatsubara = (2*epsilon)*Time2FreqFMidPoint((g**2)*GTau*DTau + (v**2)*GTau,2*M,beta) + xSigma2*sigmaold2Matsubara + xSigma*sigmaoldMatsubara
            piMatsubara = (2*epsilon)*Time2FreqBMidPoint(-2*(g**2)*GTau*np.flip(GTau),2*M,beta) + xPi2*piold2Matsubara + xPi*pioldMatsubara
        
        GThermalMatsubara = Time2FreqFMidPoint(GTau,2*M,beta)
        DThermalMatsubara = Time2FreqBMidPoint(DTau,2*M,beta)
        diffSigma = np.sqrt(np.sum((np.abs(sigmaMatsubara-sigmaoldMatsubara))**2))
        diffPi = np.sqrt(np.sum((np.abs(piMatsubara-pioldMatsubara))**2))

        threshold = np.sqrt(err)
       
        diff = 0.5*(diffSigma+diffPi)
        diffs_list.append(diff)
        diffSigma,diffPi = diff,diff

        if slowly:
            for row in range(3):
                for col in range(2):
                    axs[row,col].clear()
                    axs[row,col].grid(True)
            #axs[0,0].set_xlim(fermion_Matsubara_freqs[M-M//5],fermion_Matsubara_freqs[M+(M//5)])
            #axs[0,1].set_xlim(boson_Matsubara_freqs[M-M//5],boson_Matsubara_freqs[M+(M//5)])
            #axs[1,0].set_xlim(fermion_Matsubara_freqs[M-M//5],fermion_Matsubara_freqs[M+(M//5)])
            #axs[1,1].set_xlim(boson_Matsubara_freqs[M-M//5],boson_Matsubara_freqs[M+(M//5)])
            #axs[2,0].set_xlim(fermion_Matsubara_freqs[M-M//5],fermion_Matsubara_freqs[M+(M//5)])
            #axs[2,1].set_xlim(boson_Matsubara_freqs[M-M//5],boson_Matsubara_freqs[M+(M//5)])
            axs[0,0].set_xlabel(r'$\omega_n$')
            axs[0,1].set_xlabel(r'$\nu_n$')
            axs[0,0].set_ylabel(r'$\Re(\Sigma(\omega_n))$')
            axs[0,1].set_ylabel(r'$\Re(\Pi(\nu_n))$')
            axs[1,0].set_xlabel(r'$\omega_n$')
            axs[1,1].set_xlabel(r'$\nu_n$')
            axs[1,0].set_ylabel(r'$\Im(\Sigma(\omega_n))$')
            axs[1,1].set_ylabel(r'$\Im(\Pi(i\nu_n))$')
            axs[2,0].set_xlabel(r'$\omega_n$')
            axs[2,1].set_xlabel(r'$\nu_n$')
            axs[2,0].set_ylabel(r'$\Im(\mathcal{G}(\omega_n))$')
            axs[2,1].set_ylabel(r'$\Im(\mathcal{D}(\nu_n))$')

            axs[0,0].plot(fermion_Matsubara_freqs[::10],np.real(sigmaMatsubara[::10]),'-o',color='black',markersize = 2)
            axs[0,1].plot(boson_Matsubara_freqs[::10],np.real(piMatsubara[::10]),'-o',color='black',markersize = 2)

            axs[1,0].plot(fermion_Matsubara_freqs[::10],np.imag(sigmaMatsubara[::10]),'-o',color='black',markersize = 2)
            axs[1,1].plot(boson_Matsubara_freqs[::10],np.imag(piMatsubara[::10]),'-o',color='black',markersize = 2)

            axs[2,0].plot(fermion_Matsubara_freqs[::10],np.imag(GThermalMatsubara[::10]),'-o',color='black',markersize = 2)
            axs[2,1].plot(boson_Matsubara_freqs[::10],np.imag(DThermalMatsubara[::10]),'-o',color='black',markersize = 2)

            #axs[0,0].legend(loc = 'upper left')
            #axs[0,1].legend(loc = 'upper left')
            #axs[1,0].legend(loc = 'upper left')
            #axs[1,1].legend(loc = 'upper left')

            fig.suptitle(rf'$\beta$ = {beta}. g = {g}. Iteration = {itern}. ($diff_\Sigma$ , $diff_\Pi$) = ({diffSigma} , {diffPi})')

            plot_display.update(fig)
            time.sleep(0.0001)
            time.sleep(0.0001)

    INFO = (itern, diff)
    if slowly:
        plt.close()
    return [sigmaMatsubara,piMatsubara,GThermalMatsubara,DThermalMatsubara,INFO] 

def YSYK_2DNormalSuperiteratorEuclidean(sigmaMatsubara,piMatsubara,N_half,pars,beta,err = 1e-5,ITERMAX = 150,eta = 1e-6,epsilon = 0.01,slowly = False,executorChoice = None,optimization = 'scipy',splineFlag = False, workingPrec = 15):
    '''
    This function returns a numerical profile of the thermal self-energies for bosons and fermions
    in the two-dimensional YSYK model in the normal phase (with time-reversal symmetry storngly broken, and therefore
    no possibility of superconductivity). The function also keeps track of the thermal Green's functions. It uses an Anderson iteration algorithm to approach the best self-consistent
    solution to the Schwinger-Dyson equations. The function takes as arguments the arrays GRomega and DRomega, which
    correspond to the seed Green's functions evaluated in the Matsubara frequency grid. Such Matsubara frequencies are constructed
    from the parameter N_half, which is half the size of the set of Matsubara frequencies we will consider. The argument pars = [g,v,mu,m02seed,J,gamma] is a list which contains the value of the
    coupling constant g, the chemical potential mu, the seed bare boson mass m02seed, the fermion potential v, and the interchange constat J for bosons, 
    and the fixed length coefficient gamma, all in units of t. 
    The argument beta, of course, is the
    system's inverse temperature (also in units of t). The err kwarg is the error within which we expect the correct solution to lie after
    a sufficient number of iterations. The kwarg eta is the value of the infinitesimal regulator that we use in the
    computation of the retarded self-energies of the system.
    '''

    M = N_half
    g,v,mu,m02,J,gamma = pars 
    itern = 0

    fermion_Matsubara_freqs = np.asarray([np.pi * (2*(n - M)+1)/beta for n in range(2 * M)])
    boson_Matsubara_freqs = np.asarray([2*np.pi * (n - M)/beta for n in range(2 * M)])

    if slowly:
        fig, axs = plt.subplots(nrows = 4,ncols = 2,figsize=(14,9),layout = 'constrained')
        axInset = inset_axes(axs[3,1],width = '40%',height = '35%',loc = 'upper center')
        plot_display = display(fig, display_id = True) 
    diff = 1.
    diffm02 = 1.
    diffSigma,diffPi = (1.0,1.0)
    xSigma, xPi = 0.5 - epsilon, 0.5 - epsilon
    xSigma2, xPi2 = 0.5 - epsilon, 0.5 - epsilon

    flag = True
    diffs_list = []
    restriction_values_list = []
    M_values_list = [m02-piMatsubara[M]] 
    while (diff>err and itern<ITERMAX and flag):   
        itern += 1 
        if itern == ITERMAX:
            warnings.warn('WARNING: ITERMAX reached for beta = ' + str(beta))   
        diffoldSigma,diffoldPi = (diffSigma,diffPi)
        if itern == 1:
            sigmaoldMatsubara,pioldMatsubara = (1.0*sigmaMatsubara, 1.0*piMatsubara)
        else:
            sigmaold2Matsubara, piold2Matsubara = (1.0*sigmaoldMatsubara, 1.0*pioldMatsubara)
            sigmaoldMatsubara,pioldMatsubara = (1.0*sigmaMatsubara, 1.0*piMatsubara)

        #first we impose the fixed length restriction
        #we only optimize if the value of m02 has not already clearly converged to a stable point
        print(f'Starting m02 optimization. Current value of m02 = ',m02,'\n')
        if itern == 1 or diffm02 > err:
            oldm02 = m02
            m02_optimization_func = lambda s: np.abs(fixed_length_function_2DNormal_euclidean(M,J,beta,gamma,eta,executorChoice,s,piMatsubara,spline=splineFlag))**2
            m02_optimization_func_derivative = lambda s: 2*np.real(np.conjugate(fixed_length_function_2DNormal_euclidean(M,J,beta,gamma,eta,executorChoice,s,piMatsubara,spline=splineFlag))
            *derivative_of_fixed_length_function_2DNormal_euclidean(M,J,beta,gamma,eta,executorChoice,s,piMatsubara,spline=splineFlag))
            m02_optimization_func_second_derivative = lambda s: 2*np.real(np.conjugate(fixed_length_function_2DNormal_euclidean(M,J,beta,gamma,eta,executorChoice,s,piMatsubara,spline=splineFlag))*second_derivative_of_fixed_length_function_2DNormal_euclidean(M,J,beta,gamma,eta,executorChoice,s,piMatsubara,spline=splineFlag)) + 2*np.abs(derivative_of_fixed_length_function_2DNormal_euclidean(M,J,beta,gamma,eta,executorChoice,s,piMatsubara,spline=splineFlag))**2
            if optimization == 'scipy':
                root_finding = root_scalar(m02_optimization_func, x0 = m02, method = 'newton',fprime = m02_optimization_func_derivative)
                m02 = root_finding.root
                print('Optimization completed with m02 =',m02,'\n')
                print('Value of function evaluated at root:',m02_optimization_func(m02))
            elif optimization == 'mpmath':
                with mpmath.workdps(workingPrec):
                    m02 = mpmath.findroot(m02_optimization_func, mpmath.mpf(str(m02)), method = 'halley',df = m02_optimization_func_derivative,df2 = m02_optimization_func_second_derivative,verbose = True,dps = workingPrec)
                    print('Optimization completed with m02 =',m02,'\n')
                    print('Value of function evaluated at root:',m02_optimization_func(m02))
            else:
                m02_optimization_func = lambda s: np.real(fixed_length_function_2DNormal_euclidean(M,J,beta,gamma,eta,executorChoice,s,piMatsubara,spline=splineFlag))
                m02_optimization_func_derivative = lambda s: np.real(derivative_of_fixed_length_function_2DNormal_euclidean(M,J,beta,gamma,eta,executorChoice,s,piMatsubara,spline=splineFlag))
                m02 = mpmath.findroot(m02_optimization_func, (0,m02), method = 'bisection',df = m02_optimization_func_derivative,df2 = m02_optimization_func_second_derivative,verbose = True)
                print('Optimization completed with m02 =',m02,'\n')
                print('Value of function evaluated at root:',m02_optimization_func(m02))
            diffm02 = np.abs(m02-oldm02)
        restriction_values_list.append(m02_optimization_func(m02))

        #now we update the thermal Green's functions using Dyson's equations 
        GTau, DTau = thermalDyson2DNormal(sigmaMatsubara,piMatsubara,M,v,g,mu,J,m02,beta,eta = eta,spline=splineFlag)
        if itern == 1:
            sigmaMatsubara = (2*epsilon)*Time2FreqFMidPoint((g**2)*GTau*DTau + (v**2)*GTau,2*M,beta) + (1-2*epsilon)*sigmaoldMatsubara
            piMatsubara = (2*epsilon)*Time2FreqBMidPoint(2*(g**2)*GTau*np.flip(GTau),2*M,beta) + (1-2*epsilon)*pioldMatsubara
            #removed minus sign that was misplaced!
        else:
            sigmaMatsubara = (2*epsilon)*Time2FreqFMidPoint((g**2)*GTau*DTau + (v**2)*GTau,2*M,beta) + xSigma2*sigmaold2Matsubara + xSigma*sigmaoldMatsubara
            piMatsubara = (2*epsilon)*Time2FreqBMidPoint(2*(g**2)*GTau*np.flip(GTau),2*M,beta) + xPi2*piold2Matsubara + xPi*pioldMatsubara
            #removed minus sign that was misplaced!
        M_thermal = m02 - piMatsubara[M]
        M_values_list.append(M_thermal)
        GThermalMatsubara = Time2FreqFMidPoint(GTau,2*M,beta)
        DThermalMatsubara = Time2FreqBMidPoint(DTau,2*M,beta)
        diffSigma = np.sqrt(np.sum((np.abs(sigmaMatsubara-sigmaoldMatsubara))**2))
        diffPi = np.sqrt(np.sum((np.abs(piMatsubara-pioldMatsubara))**2))

        threshold = np.sqrt(err)
       
        diff = 0.5*(diffSigma+diffPi)
        diffs_list.append(diff)
        diffSigma,diffPi = diff,diff

        if slowly:
            for row in range(4): 
                for col in range(2):
                    axs[row,col].clear()
                    axs[row,col].grid(True)
            axInset.clear()
            axInset.grid(True)
            axs[0,0].set_xlim(fermion_Matsubara_freqs[M-M//5],fermion_Matsubara_freqs[M+(M//5)])
            axs[0,1].set_xlim(boson_Matsubara_freqs[M-M//5],boson_Matsubara_freqs[M+(M//5)])
            axs[1,0].set_xlim(fermion_Matsubara_freqs[M-M//5],fermion_Matsubara_freqs[M+(M//5)])
            axs[1,1].set_xlim(boson_Matsubara_freqs[M-M//5],boson_Matsubara_freqs[M+(M//5)])
            axs[2,0].set_xlim(fermion_Matsubara_freqs[M-M//5],fermion_Matsubara_freqs[M+(M//5)])
            axs[2,1].set_xlim(boson_Matsubara_freqs[M-M//5],boson_Matsubara_freqs[M+(M//5)])
            axs[0,0].set_xlabel(r'$\omega_n$')
            axs[0,1].set_xlabel(r'$\nu_n$')
            axs[0,0].set_ylabel(r'$\Re(\Sigma(\omega_n))$')
            axs[0,1].set_ylabel(r'$\Re(\Pi(\nu_n))$')
            axs[1,0].set_xlabel(r'$\omega_n$')
            axs[1,1].set_xlabel(r'$\nu_n$')
            axs[1,0].set_ylabel(r'$\Im(\Sigma(\omega_n))$')
            axs[1,1].set_ylabel(r'$\Im(\Pi(i\nu_n))$')
            axs[2,0].set_xlabel(r'$\omega_n$')
            axs[2,1].set_xlabel(r'$\nu_n$')
            axs[2,0].set_ylabel(r'$\Im(\mathcal{G}(\omega_n))$')
            axs[2,1].set_ylabel(r'$\Im(\mathcal{D}(\nu_n))$')
            axs[3,0].set_xlabel('iteration') 
            axs[3,1].set_xlabel('iteration') 
            axs[3,0].set_ylabel(r'$\left|D(\tau=0)-1/\gamma\right|$') 
            axs[3,0].set_yscale('log')
            axs[3,1].set_ylabel(r'$M^2(T) = m_0^2-\Pi(\omega = 0)$') 

            axs[0,0].plot(fermion_Matsubara_freqs[::10],np.real(sigmaMatsubara[::10]),'-o',color='black',markersize = 2)
            axs[0,1].plot(boson_Matsubara_freqs[::10],np.real(piMatsubara[::10]),'-o',color='black',markersize = 2)

            axs[1,0].plot(fermion_Matsubara_freqs[::10],np.imag(sigmaMatsubara[::10]),'-o',color='black',markersize = 2)
            axs[1,1].plot(boson_Matsubara_freqs[::10],np.imag(piMatsubara[::10]),'-o',color='black',markersize = 2)

            axs[2,0].plot(fermion_Matsubara_freqs[::10],np.imag(GThermalMatsubara[::10]),'-o',color='black',markersize = 2)
            axs[2,1].plot(boson_Matsubara_freqs[::10],np.imag(DThermalMatsubara[::10]),'-o',color='black',markersize = 2)

            axs[3,0].plot(np.arange(1,itern+1,step=1,dtype=int),np.abs(np.asarray(restriction_values_list)),'-o',color = 'blue',markersize=3)
            axs[3,1].plot(np.arange(0,itern+1,step=1,dtype=int),np.real(np.array(M_values_list,dtype=complex)),'-o',color = 'red',markersize=3,label = r'$\Re(.)$')

            axInset.plot(np.arange(0,itern+1,step=1,dtype=int),np.abs(np.imag(np.array(M_values_list,dtype=complex))),'-o',color = 'orange',markersize=3,label = r'$\Im(.)$')
            axInset.set_yscale('log')

            fig.suptitle(rf'$\beta$ = {beta}. g = {g}. Iteration = {itern}. ($diff_\Sigma$ , $diff_\Pi$) = ({diffSigma} , {diffPi})')

            #display.clear_output(wait=True)
            #display.display(fig)
            plot_display.update(fig)
            time.sleep(0.0001)

    INFO = (itern, diff)
    if slowly:
        plt.close()
    return [sigmaMatsubara,piMatsubara,GThermalMatsubara,DThermalMatsubara,m02,M_thermal,INFO] 

def CrazyGconfReal(omega,g,beta,eta=0):
    ''' 
    Arguments omega,g,beta
    So far we will implement only the kappa=1 result eq.16 of esterlis-schmalian
    omega is the grid of fermionic matsubara frequencies
    '''
    c1 = 1.154700
    delta = 0.420374134464041
    ompluit = omega + 1j*eta
    denom = ompluit + c1*(g**(4*delta)) * (1j**(2*delta)) * ompluit**(1-2*delta)
    print('bar')
    return 1./denom

def CrazyDconfReal(omega,g,beta,eta=0):
    ''' 
    Arguments: nu,g,beta
    So far we will implement only the kappa=1 result eq.16 of esterlis-schmalian
    nu is the grid of fermionic matsubara frequencies
    '''
    T = 1.0/beta
    c2 = 0.561228
    c3 = 0.709618
    delta = 0.420374134464041
    omegar2 = c2 * (T/(g**2))**(4*delta - 1)
    print('boo')
    return 1.0/(1*(omega+1j*eta)**2 + omegar2 + c3*(np.abs((omega+1j*eta)/(1j* (g**2))))**(4*delta - 1))

def CrazyGconfThermal(M,g,beta):
    fermion_Matsubara_freqs = np.asarray([np.pi * (2*(n - M)+1)/beta for n in range(2 * M)])

    c1 = 1.1547000
    bigDelta = 0.42037413
    resultRecip = 1j*fermion_Matsubara_freqs*(1 + c1*np.abs((g**2)/fermion_Matsubara_freqs)**(1*bigDelta))
    return 1/resultRecip

def CrazyDconfThermal(M,g,beta):
    boson_Matsubara_freqs = np.asarray([2*np.pi * (n - M)/beta for n in range(2 * M)])

    c2 = 0.561228
    c3 = 0.709618
    bigDelta = 0.42037413
    resultRecip = boson_Matsubara_freqs**2 + c2*(1/(beta*g**2))**(4*bigDelta - 1) + c3*np.abs(boson_Matsubara_freqs/(g**2))**(4*bigDelta - 1) 
    return 1/resultRecip

if __name__ == "__main__":
    trial_z = np.linspace(-5,5,100)
    try:
        derivative_of_elegant_G(trial_z,executor = None)
        print(f'Success evaluating the derivative.')
        for z in trial_z:
            print(f"z = {z} , G'(z) = {derivative_of_elegant_G(z)}")
    except:
        print('Oops!')