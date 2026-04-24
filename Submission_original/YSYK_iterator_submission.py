import numpy as np 
from SYK_fft import *
from ConformalAnalytical import *
import warnings
import matplotlib.pyplot as plt
import scipy

def Dav_rhotosigma(rhoG,rhoD,M,t,g,beta,BMf,kappa=1,delta=1e-6):
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
    omegar2 = ret_omegar2(g,beta)

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
    
        rhoG = -1.0*np.imag(GRomega)
        rhoD = -1.0*np.imag(DRomega)

        SigmaOmega,PiOmega = Dav_rhotosigma(rhoG,rhoD,M,t,g,beta,BMf,kappa=1,delta=eta)
        

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
        GRt = (0.5/np.pi) * freq2time(GRomega,M,dt)
        DRt = (0.5/np.pi) * freq2time(DRomega,M,dt)
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
        if diffcheck:
            diffseries += [diff]
            flag = testingscripts.diff_checker(diffseries, tol = 1e-3, periods = 5)
        
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

def RE_WHYSYK_iterator(GFs,grid,pars,beta,lamb,J,x = 0.01,err=1e-5,ITERMAX=150,eta=1e-6, verbose=True, diffcheck = False):
    '''
    signature:
    GFs = GDRomega, GODRomega, DDRomega, DODRomega
    GFs,grid,pars,beta,err=1e-5,ITERMAX=150,eta=1e-6, verbose=True, diffcheck = False
    grid is a list [M,omega,t]
    pars is a list [g,mu,r]
    '''
    GDRomega, GODRomega, DDRomega, DODRomega = GFs
    M,omega,t = grid
    g,mu,r = pars
    itern = 0

    diff = 1.
    diffold = 1.
    #x = 0.01

    diffseries = []
    flag = True
    fdplus = np.array([fermidirac(beta*omegaval, default = False) for omegaval in omega])
    fdminus = np.array([fermidirac(-1.0*beta*omegaval, default = False) for omegaval in omega])
    beplus = np.array([boseeinstein(beta*omegaval, default = False) for omegaval in omega])
    beminus = np.array([boseeinstein(-1.0*beta*omegaval, default = False) for omegaval in omega])
    BMf = [fdplus, fdminus, beplus, beminus]

    for xval in (x,):
        diff = 1
        while (diff>err and itern<ITERMAX and flag): 
            itern += 1 
            diffold = diff
            if itern == ITERMAX:
                warnings.warn('WARNING: ITERMAX reached for beta = ' + str(beta))
            #diffoldG,diffoldD = (diffG,diffD)
            GDRoldomega,DDRoldomega = (1.0*GDRomega, 1.0*DDRomega)
            GODRoldomega,DODRoldomega = (1.0*GODRomega, 1.0*DODRomega)

            rhoGD = -1.0*np.imag(GDRomega)
            rhoDD = -1.0*np.imag(DDRomega)
            rhoGOD = -1.0*np.imag(GODRomega)
            rhoDOD = -1.0*np.imag(DODRomega)

            SigmaDomega,PiDomega = Dav_rhotosigma(rhoGD,rhoDD,M,t,g,beta,BMf,kappa=1,delta=eta)
            SigmaODomega,PiODomega = Dav_rhotosigma(rhoGOD,rhoDOD,M,t,g,beta,BMf,kappa=1,delta=eta)
         
            detGmat = (omega+1j*eta + mu - SigmaDomega)**2 - (lamb + SigmaODomega)**2
            detDmat = (r-(omega+1j*eta)**2 - PiDomega)**2 - (J-PiODomega)**2

            GDRomega = xval*((omega+1j*eta + mu - SigmaDomega)/detGmat) + (1-xval)*GDRoldomega
            GODRomega = xval*((lamb + SigmaODomega)/detGmat) + (1-xval)*GODRoldomega
            DDRomega = xval*((r - (omega+1j*eta)**2 - PiDomega)/detDmat) + (1-xval)*DDRoldomega
            DODRomega = xval*(-1.0*(J - PiODomega)/detDmat) + (1-xval)*DODRoldomega


            diffGD = np.sqrt(np.sum((np.abs(GDRomega-GDRoldomega))**2))#changed
            diffGOD = np.sqrt(np.sum((np.abs(GODRomega-GODRoldomega))**2)) #changed
            diffDD = np.sqrt(np.sum((np.abs(DDRomega-DDRoldomega))**2)) #changed
            diffDOD = np.sqrt(np.sum((np.abs(DODRomega-DODRoldomega))**2))#changed
            diff = 0.25*(diffGD+diffDOD+diffDD+diffGOD)

            if diffcheck == True:
                diffseries += [diff]
                if itern >10:
                    flag = testingscripts.diff_checker(diffseries, tol = 1e-6, periods = 7)
            
            if verbose and itern % 100 == 0:
                print("itern = ",itern, " , diff = ", diff, " , xval = ", xval,flush=True)
                plt.plot(omega,-np.imag(GDRomega))
                plt.xlim(-1,1)
                plt.show()

    GFs = [GDRomega, GODRomega, DDRomega, DODRomega]
    INFO = (itern, diff, xval)
    return (GFs, INFO)




























