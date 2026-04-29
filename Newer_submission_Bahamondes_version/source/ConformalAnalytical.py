import numpy as np
from source.SYK_fft import *
import warnings
import matplotlib.pyplot as plt
import scipy

def rhotosigma(rhoG,rhoD,M,dt,t,omega,g,beta,delta=1e-6):
    '''
    returns [Sigma,Pi] given rhos
    '''
    eta = np.pi/(M*dt)*(0.001)
    rhoGrev = np.concatenate(([rhoG[-1]], rhoG[1:][::-1]))
    rhoFpp = freq2time(rhoG * fermidirac(beta*omega),M,dt)
    rhoFpm = freq2time(rhoG * fermidirac(-1.*beta*omega),M,dt)
    rhoFmp = freq2time(rhoGrev * fermidirac(beta*(omega)),M,dt)
    rhoFmm = freq2time(rhoGrev * fermidirac(-1.*beta*omega),M,dt)
    rhoBpp = freq2time(rhoD * boseeinstein(beta*(omega+eta)),M,dt)
    rhoBpm = freq2time(rhoD * boseeinstein(-1.*beta*(omega+eta)),M,dt)
    
    argSigma = (rhoFpm*rhoBpm - rhoFpp*rhoBpp) * np.exp(-np.abs(delta*t)) * np.heaviside(t,1)
    #argSigma = (rhoFpm*rhoBpm - rhoFpp*rhoBpp) * np.heaviside(t,1)
    Sigma = 1j*(g**2) * time2freq(argSigma,M,dt)
    
    argPi = (rhoFpp*rhoFmp - rhoFpm*rhoFmm) * np.exp(-np.abs(delta*t)) * np.heaviside(t,1)
    #argPi = (rhoFpp*rhoFmp - rhoFpm*rhoFmm) * np.heaviside(t,1)
    Pi = 2*1j*(g**2) * time2freq(argPi,M,dt)
    
    return [Sigma, Pi]

def Dav_rhotosigma(rhoG,rhoD,M,t,g,beta,BMf,delta=1e-6):
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
    Sigma = -1j*(g**2)* time2freq(argSigma,M,dt)

    argPi = (AGt * np.conj(aGt) - np.conj(AGt) * (aGt)) * np.heaviside(t,0)
    Pi = 2j*(g**2)* time2freq(argPi,M,dt)

    return [Sigma,Pi]

def RE_YSYK_iterator(GRomega,DRomega,grid,pars,beta,err=1e-5,ITERMAX=150,eta=1e-6, verbose=True, diffcheck = False):
    '''
    signature:
    GRomega,DRomega,grid,pars,beta,err=1e-5,ITERMAX=150,eta=1e-6, verbose=True, diffcheck = False
    grid is a list [M,omega,t]
    pars is a list [g,mu,r]

    NB: This function has a different annealing scheme for x, to ensure convergence in the strong coupling regime

    '''
    M,omega,t = grid
    g,mu,r = pars
    itern = 0

    diff = 1.
    diffG,diffD = (1.0,1.0)
    epsilon = 0.01

    xG, xD = 0.5 - epsilon, 0.5 - epsilon
    xG2, xD2 = 0.5 - epsilon, 0.5 - epsilon


    diffseries = []
    flag = True
    fdplus = np.array([fermidirac(beta*omegaval, default = False) for omegaval in omega])
    fdminus = np.array([fermidirac(-1.0*beta*omegaval, default = False) for omegaval in omega])
    beplus = np.array([boseeinstein(beta*omegaval, default = False) for omegaval in omega])
    beminus = np.array([boseeinstein(-1.0*beta*omegaval, default = False) for omegaval in omega])
    BMf = [fdplus, fdminus, beplus, beminus]
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

        SigmaOmega,PiOmega = Dav_rhotosigma(rhoG,rhoD,M,t,g,beta,BMf,delta=eta)
        

        if np.imag(SigmaOmega[M] > 0) :
            warnings.warn('Violation of causality : Pole of Gomega in UHP for beta = ' + str(beta))

        if itern == 1:
            GRomega = (2*epsilon)/(omega + 1j*eta + mu - SigmaOmega) + (1-2*epsilon)*GRoldomega
            DRomega = (2*epsilon)/(-1.0*(omega+1j*eta)**2 + r - PiOmega) + (1-2*epsilon)*DRoldomega
        
        else:
            GRomega = (2*epsilon)/(omega + 1j*eta + mu - SigmaOmega) + xG2*GRold2omega + xG*GRoldomega
            DRomega = (2*epsilon)/(-1.0*(omega+1j*eta)**2 + r - PiOmega) + xD2*DRold2omega + xD*DRoldomega

        #causality constraint
        dt = t[2]-t[1]
        GRt = freq2time(GRomega,M,dt)
        DRt = freq2time(DRomega,M,dt)
        GRt[:M] = 0  
        DRt[:M] = 0
        GRomega = time2freq(GRt,M,dt)
        DRomega = time2freq(DRt,M,dt)

        diffG = np. sqrt(np.sum((np.abs(GRomega-GRoldomega))**2)) #changed
        diffD = np. sqrt(np.sum((np.abs(DRomega-DRoldomega))**2))

        treshold = np.sqrt(err)
       
        diff = 0.5*(diffG+diffD)
        diffs_list.append(diff)
        diffG,diffD = diff,diff
        
        if verbose:
            print("itern = ",itern, " , diff = ", diffG, diffD," , x = ", xG, xD)
  
        if verbose and itern % 100 == 0:
            fig, ax = plt.subplots(4)
            fig.suptitle('Iteration = ' + str(itern) + ', beta = ' + str(beta))
            ax[0].plot(omega, -1*np.imag(GRomega), label = 'Im(GR)')
            ax[1].plot(omega, -1*np.imag(DRomega), label = 'Im(DR)')
            ax[0].set_xlabel(r'$\omega$')
            ax[0].set_ylabel(r'$-Im G^R(\omega)$')
            ax[0].set_xlim(0,4)
            ax[1].set_xlabel(r'$\omega$')
            ax[1].set_ylabel(r'$-Im D^R(\omega)$')
            ax[1].set_xlim(0,2)
            ax[2].plot(omega, -1*np.imag(SigmaOmega), label = 'Sigma')
            ax[3].plot(omega, -1*np.imag(PiOmega), label = 'Pi')
            # ax[3].set_xlim(-2,2)
            # ax[2].set_xlim(-2,2)

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