from scipy.fft import fft as fft
from scipy.fft import ifft as ifft
import numpy as np
import warnings
from new_source_files.new_SYK_fft import *
import matplotlib.pyplot as plt

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
        rhoGrev = np.concatenate(([rhoG[-1]], rhoG[1:][::-1]))
        rhoFpp = freq2time(rhoG * fermidirac_stable(beta*omega),M,dt)
        rhoFmp = freq2time(rhoG * fermidirac_stable(-1.*beta*omega),M,dt)
        rhoFpm = freq2time(rhoGrev * fermidirac_stable(beta*(omega)),M,dt)
        rhoFmm = freq2time(rhoGrev * fermidirac_stable(-1.*beta*omega),M,dt)
        rhoBpp = freq2time(rhoD * boseeinstein_stable(beta*(omega+eta)),M,dt)
        rhoBmp = freq2time(rhoD * boseeinstein_stable(-1.*beta*(omega+eta)),M,dt)

        SigmaInTime = 1.0j * np.exp(-np.abs(delta*t)) * np.heaviside(t,0) * ((v**2)*(rhoFpp+rhoFmp)-(g**2)*(rhoFpp*rhoBpp-rhoFmp*rhoBmp))
        Sigma = time2freq(SigmaInTime,M,dt)

        PiInTime = 1.0j * np.exp(-np.abs(delta*t)) * np.heaviside(t,0) * (g**2) * (rhoFpp*rhoFpm - rhoFmp*rhoFmm)
        Pi = time2freq(PiInTime,M,dt)
    else:
        #rhoG and rhoD are assumed to be evaluated at mid-points of omega and t grids.
        domega = np.pi/(M*dt)
        rhoGrevMidPoint = np.concatenate(([rhoG[-1]], rhoG[1:][::-1]))
        rhoFppMidPoint = freq2timeMidPoint(rhoG * fermidirac_stable(beta*(omega+domega/2)),M,dt)
        rhoFmpMidPoint = freq2timeMidPoint(rhoG * fermidirac_stable(-1.*beta*(omega+domega/2)),M,dt)
        rhoFpmMidPoint = freq2timeMidPoint(rhoGrevMidPoint * fermidirac_stable(beta*(omega+domega/2)),M,dt)
        rhoFmmMidPoint = freq2timeMidPoint(rhoGrevMidPoint * fermidirac_stable(-1.*beta*(omega+domega/2)),M,dt)
        rhoBppMidPoint = freq2timeMidPoint(rhoD * boseeinstein_stable(beta*(omega+domega/2)),M,dt)
        rhoBmpMidPoint = freq2timeMidPoint(rhoD * boseeinstein_stable(-1.*beta*(omega+domega/2)),M,dt)

        SigmaInTimeMidPoint = 1.0j * np.exp(-np.abs(delta*(t+dt/2))) * np.heaviside(t+dt/2,1) * ((v**2)*(rhoFppMidPoint +\
                            rhoFmpMidPoint)-(g**2)*(rhoFppMidPoint*rhoBppMidPoint-rhoFmpMidPoint*rhoBmpMidPoint))
        Sigma = time2freqMidPoint(SigmaInTimeMidPoint,M,dt)

        PiInTimeMidPoint = 1.0j * np.exp(-np.abs(delta*(t+dt/2))) * np.heaviside(t+dt/2,1) * (g**2) * (rhoFppMidPoint*rhoFpmMidPoint - \
                           rhoFmpMidPoint*rhoFmmMidPoint)
        Pi = time2freqMidPoint(PiInTimeMidPoint,M,dt)
        
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
        rhoGrev = np.concatenate(([rhoG[-1]], rhoG[1:][::-1]))
        rhoFpp = freq2time(rhoG * fermidirac_stable(beta*omega),M,dt)
        rhoFmp = freq2time(rhoG * fermidirac_stable(-1.0*beta*omega),M,dt)
        rhoFpm = freq2time(rhoGrev * fermidirac_stable(beta*omega),M,dt)
        rhoFmm = freq2time(rhoGrev * fermidirac_stable(-1.0*beta*omega),M,dt)
        rhoBpp = freq2time(rhoD * boseeinstein_stable(beta*(omega+eta)),M,dt)
        rhoBmp = freq2time(rhoD * boseeinstein_stable(-1.0*beta*(omega+eta)),M,dt)
    
        SigmaInTime = (rhoFmp*rhoBmp - rhoFpp*rhoBpp) * np.exp(-np.abs(delta*t)) * np.heaviside(t,0)
        Sigma = 1j*(g**2) * time2freq(SigmaInTime,M,dt)
    
        PiInTime = (rhoFpp*rhoFpm - rhoFmp*rhoFmm) * np.exp(-np.abs(delta*t)) * np.heaviside(t,0)
        Pi = 2j*(g**2) * time2freq(PiInTime,M,dt)
    else:
        #rhoG and rhoD are assumed to be evaluated at mid-points of omega and t grids.
        domega = np.pi/(M*dt)
        rhoGrevMidPoint = np.concatenate(([rhoG[-1]], rhoG[1:][::-1]))
        rhoFppMidPoint = freq2timeMidPoint(rhoG * fermidirac_stable(beta*(omega+domega/2)),M,dt)
        rhoFmpMidPoint = freq2timeMidPoint(rhoG * fermidirac_stable(-1.0*beta*(omega+domega/2)),M,dt)
        rhoFpmMidPoint = freq2timeMidPoint(rhoGrevMidPoint * fermidirac_stable(beta*(omega+domega/2)),M,dt)
        rhoFmmMidPoint = freq2timeMidPoint(rhoGrevMidPoint * fermidirac_stable(-1.0*beta*(omega+domega/2)),M,dt)
        rhoBppMidPoint = freq2timeMidPoint(rhoD * boseeinstein_stable(beta*(omega+domega/2)),M,dt)
        rhoBmpMidPoint = freq2timeMidPoint(rhoD * boseeinstein_stable(-1.0*beta*(omega+domega/2)),M,dt)

        SigmaInTimeMidPoint = (rhoFmpMidPoint*rhoBmpMidPoint - rhoFppMidPoint*rhoBppMidPoint)\
                            * np.exp(-np.abs(delta*(t+dt/2))) * np.heaviside(t+dt/2,0)
        Sigma = 1j*(g**2) * time2freqMidPoint(SigmaInTimeMidPoint,M,dt)
    
        PiInTimeMidPoint = (rhoFppMidPoint*rhoFpmMidPoint - rhoFmpMidPoint*rhoFmmMidPoint)\
                            * np.exp(-np.abs(delta*(t+dt/2))) * np.heaviside(t+dt/2,0)
        Pi = 2j*(g**2) * time2freqMidPoint(PiInTimeMidPoint,M,dt)

    return [Sigma, Pi]

def Dav_rho2sigma0DNormal(rhoG,rhoD,M,t,g,beta,BMf,kappa=1,delta=1e-6):
    '''
    Direct implementation of Davide's email
    '''
    dt = t[2]-t[1]
    fdplus,fdminus,beplus,beminus = BMf
    ADt = (1/np.pi) * freq2time(rhoD,M,dt)
    aGt = (1/np.pi) * freq2time(rhoG * fdplus, M,dt)
    AGt = (1/np.pi) * freq2time(rhoG,M,dt)
    aDt = (1/np.pi) * freq2time(rhoD * beplus, M,dt)

    argSigma = (ADt * aGt - AGt * np.conj(aDt)) * np.heaviside(t,0)
    Sigma = -1j*(g**2)*kappa* time2freq(argSigma,M,dt)

    argPi = (AGt * np.conj(aGt) - np.conj(AGt) * (aGt)) * np.heaviside(t,0)
    Pi = 2j*(g**2)*kappa* time2freq(argPi,M,dt)

    return [Sigma,Pi]

def YSYK_0DNormaliterator(GRomega,DRomega,grid,pars,beta,err=1e-5,ITERMAX=150,eta=1e-6,verbose=True,midPointFlag=False):
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
    itern = 0

    diff = 1.
    diffG,diffD = (1.0,1.0)
    epsilon = 0.5

    xG, xD = 0.5 - epsilon, 0.5 - epsilon
    xG2, xD2 = 0.5 - epsilon, 0.5 - epsilon

    flag = True
    diffs_list = []

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
    
        rhoG = -0.5*np.imag(GRomega)
        rhoD = -0.5*np.imag(DRomega)
        
        SigmaOmega,PiOmega = rho2sigma0DNormal(rhoG,rhoD,M,dt,t,omega,g,beta,delta=0,midPoint=midPointFlag)

        if np.imag(SigmaOmega[M] > 0) :
            warnings.warn('Violation of causality : Pole of Gomega in UHP for beta = ' + str(beta))
        if itern == 1:
            GRomega = (2*epsilon)/(omega + 1j*eta + mu - SigmaOmega) + (1-2*epsilon)*GRoldomega
            DRomega = (2*epsilon)/(-1.0*(omega+1j*eta)**2 + m02 - PiOmega) + (1-2*epsilon)*DRoldomega
        else:
            GRomega = (2*epsilon)/(omega + 1j*eta + mu - SigmaOmega) + xG2*GRold2omega + xG*GRoldomega
            DRomega = (2*epsilon)/(-1.0*(omega+1j*eta)**2 + m02 - PiOmega) + xD2*DRold2omega + xD*DRoldomega

        #causality constraint
        if midPointFlag:
            GRt = freq2timeMidPoint(GRomega,M,dt)
            DRt = freq2timeMidPoint(DRomega,M,dt)
            GRt[:M] = 0  
            DRt[:M] = 0
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

        if verbose:
            print("itern = ",itern, " , diff = ", diffG, diffD," , x = ", xG, xD)
        if verbose and itern % 100 == 0:
            fig, ax = plt.subplots(4)
            fig.suptitle('Iteration = ' + str(itern) + ', beta = ' + str(beta))
            ax[0].plot(omega, -0.5*np.imag(GRomega), label = 'Im(GR)')
            ax[1].plot(omega, -0.5*np.imag(DRomega), label = 'Im(DR)')
            ax[0].set_xlabel(r'$\omega$')
            ax[0].set_ylabel(r'$-Im G^R(\omega)$')
            ax[0].set_xlim(0,4)
            ax[1].set_xlabel(r'$\omega$')
            ax[1].set_ylabel(r'$-Im D^R(\omega)$')
            ax[1].set_xlim(0,2)
            ax[2].plot(omega, -0.5*np.imag(SigmaOmega), label = 'Sigma')
            ax[3].plot(omega, -0.5*np.imag(PiOmega), label = 'Pi')

            #bigger size and tight layout
            fig.set_size_inches(6,6)
            fig.tight_layout()

            plt.show()
    INFO = (itern, diff)
    return (GRomega,DRomega, INFO)

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


