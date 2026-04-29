from scipy.fft import fft as fft
from scipy.fft import ifft as ifft
import numpy as np
import warnings
from new_source_files.new_SYK_fft import *
import matplotlib.pyplot as plt
from IPython import display
import time

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
        rhoGrev = np.flip(rhoG)
        rhoFpp = freq2time(rhoG * fermidirac_stable(beta*omega),M,dt)
        rhoFmp = freq2time(rhoG * fermidirac_stable(-1.0*beta*omega),M,dt)
        rhoFpm = freq2time(rhoGrev * fermidirac_stable(beta*omega),M,dt)
        rhoFmm = freq2time(rhoGrev * fermidirac_stable(-1.0*beta*omega),M,dt)
        rhoBpp = freq2time(rhoD * boseeinstein_stable(beta*(omega+eta)),M,dt)
        rhoBmp = freq2time(rhoD * boseeinstein_stable(-1.0*beta*(omega+eta)),M,dt)
    
        SigmaInTime = (rhoFmp*rhoBmp - rhoFpp*rhoBpp) * np.exp(-np.abs(delta*t)) * np.heaviside(t,0)
        Sigma = -1j*(g**2) * time2freq(SigmaInTime,M,dt)
        #changed sign!
        PiInTime = (rhoFpp*rhoFpm - rhoFmp*rhoFmm) * np.exp(-np.abs(delta*t)) * np.heaviside(t,0)
        Pi = 2*1j*(g**2) * time2freq(PiInTime,M,dt)
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
        Sigma = 1j*(g**2) * time2freqMidPoint(SigmaInTimeMidPoint,M,dt)
    
        PiInTimeMidPoint = (rhoFppMidPoint*rhoFpmMidPoint - rhoFmpMidPoint*rhoFmmMidPoint)\
                            * np.exp(-np.abs(delta*(t+dt/2))) * np.heaviside(t+dt/2,0)
        Pi = 2j*(g**2) * time2freqMidPoint(PiInTimeMidPoint,M,dt)

    return [Sigma, Pi]

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
    
        rhoG = -0.5*np.imag(GRomega)
        rhoD = -0.5*np.imag(DRomega)
        
        SigmaOmega,PiOmega = Dav_rho2sigma0DNormal(rhoG,rhoD,M,t,g,beta,BMf,delta=0)

        if np.imag(SigmaOmega[M] > 0) :
            print('Violation of causality : Pole of Gomega in UHP for beta = ' + str(beta))
            break
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
            axs[2,0].set_xlim(0,1)
            axs[2,1].set_xlim(0,1)
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

            axs[2,0].plot(np.log(omega[M+1:]),np.log(np.abs(0.5*np.imag(GRomega[M+1:]))))
            axs[2,1].plot(omega,0.5*np.imag(DRomega))

            axs[0,0].legend(loc = 'upper left')
            axs[0,1].legend(loc = 'upper left')
            axs[1,0].legend(loc = 'upper left')
            axs[1,1].legend(loc = 'upper left')

            fig.suptitle(rf'$\beta$ = {beta}. Iteration = {itern}. (diffG , diffD) = ({diffG} , {diffD})')

            display.clear_output(wait=True)
            display.display(fig)
            time.sleep(0.0001)

    INFO = (itern, diff)
    if slowly:
        plt.close()
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


