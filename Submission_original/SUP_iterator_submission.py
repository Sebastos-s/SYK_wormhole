import numpy as np 
from SYK_fft import *
import matplotlib.pyplot as plt
from ConformalAnalytical import *
import warnings



def SupDav_rhotosigma(rhoG,rhoD,rhoF,M,t,g,beta,BMf,eta, kappa=1,alpha=0.):
    '''
    Direct implementation of Davide's email
    '''
    dt = t[2]-t[1]
    fdplus,fdminus,beplus,beminus = BMf
    ADt = (1/np.pi) * freq2time(rhoD,M,dt)
    aGt = (1/np.pi) * freq2time(rhoG * fdplus, M,dt)
    AGt = (1/np.pi) * freq2time(rhoG,M,dt)
    aDt = (1/np.pi) * freq2time(rhoD * beplus, M,dt)
    aFt = (1/np.pi) * freq2time(rhoF * fdplus, M,dt)
    AFt = (1/np.pi) * freq2time(rhoF,M,dt)

    argSigma = (ADt * aGt - AGt * np.conj(aDt)) * np.heaviside(t,0)
    Sigma = -1j*(g**2)*kappa* time2freq(argSigma,M,dt)

    argPhi = (ADt * aFt - AFt * np.conj(aDt)) * np.heaviside(t,0)
    Phi = 1j*(1.-alpha)*(g**2)*kappa* time2freq(argPhi,M,dt)
    argPi = ((AGt * np.conj(aGt) - np.conj(AGt) * (aGt)) - (1.-alpha)*(AFt * np.conj(aFt) - np.conj(AFt) * (aFt))) * np.heaviside(t,0) #* np.exp(-eta*np.abs(t)/3)

    Pi = 2j*(g**2)*kappa* time2freq(argPi,M,dt)

    return [Sigma,Pi,Phi]

def rhotoPiF(rhoF, M, t, beta, BMf, eta):
    dt = t[2] - [1]
    fdplus,fdminus,beplus,beminus = BMf
    aFt = (1/np.pi) * freq2time(rhoF * fdplus, M,dt) 
    AFt = (1/np.pi) * freq2time(rhoF,M,dt)
    argPiF = (np.conj(aFt) * AFt + aFt * AFt) * np.heaviside(t,0) #* np.exp(-eta*np.abs(t)/3)
    PiF = -1j*time2freq(argPiF, M ,dt)
    return PiF

def acjosephson(omega, lamb, V, PiF, t):
    #find index around omega = V:
    idx = (np.abs(omega - V)).argmin()
    print(omega[idx], PiF[idx])
    # I = 2*(lamb**2)*np.imag(np.exp(-2*1j*V*t)*PiF[idx])
    I = 2*(lamb**2)*np.cos(2*V*t)*np.imag(PiF[idx])
    return I

def acjosephson_amplitude(omega, lamb, V, PiF, t):
    #find index around omega = V:
    idx = (np.abs(omega - V)).argmin()
    print(omega[idx], PiF[idx])
    I = 2*(lamb**2)*np.imag(np.exp(-2*1j*V*t)*PiF[idx])
    # I = 2*(lamb**2)*np.cos(2*V*t)*np.imag(PiF[idx])
    return np.max(I)-np.min(I) #peak to peak amplitude    

def GF_RE_SUP_iterator(GRomega,DRomega,FRomega,grid,pars,beta, alpha, err=1e-5,ITERMAX=150,eta=1e-6, verbose=True, diffcheck = False):
    '''
    signature:
    GRomega,DRomega,grid,pars,beta,err=1e-5,ITERMAX=150,eta=1e-6, verbose=True, diffcheck = False
    grid is a list [M,omega,t]
    pars is a list [g,mu,r]
    '''
    M,omega,t = grid
    g,mu,r = pars
    itern = 0
    print(alpha)

    diff = 1.
    diffG,diffD = (1.0,1.0)
    x = 0.01

    xG, xD = x,x
    diffseries = []
    flag = True
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
        GRoldomega,DRoldomega,FRoldomega = (1.0*GRomega, 1.0*DRomega,1.0*FRomega)

        rhoG = -1.0*np.imag(GRomega)
        rhoD = -1.0*np.imag(DRomega)
        rhoF = -1.0*np.imag(FRomega)

        SigmaOmega,PiOmega,Phiomega = SupDav_rhotosigma(rhoG,rhoD,rhoF,M,t,g,beta,BMf,eta, kappa=1,alpha=alpha)
        
        if np.imag(SigmaOmega[M] > 0) :
            warnings.warn('Violation of causality : Pole of Gomega in UHP for beta = ' + str(beta))

        DRomega = 1.0*xD/(-1.0*(omega+1j*eta)**2 + r - PiOmega) + (1-xD)*DRoldomega
        detGmat = (omega+1j*eta - mu - SigmaOmega)**2 - (Phiomega)**2
        GRomega = 1.0*xG*(omega + 1j*eta - mu - SigmaOmega)/(detGmat) + (1-xG)*GRoldomega
        FRomega = 1.0*xG*(Phiomega)/(detGmat) + (1-xG)*FRoldomega
        

        diffG = np. sqrt(np.sum((np.abs(GRomega-GRoldomega))**2)) #changed
        diffD = np. sqrt(np.sum((np.abs(DRomega-DRoldomega))**2))
        diffF = np. sqrt(np.sum((np.abs(FRomega-FRoldomega))**2))
        diff = 0.33*(diffG+diffD+diffF)
        diffG,diffD,diffF = diff,diff,diff
        if diffcheck:
            diffseries += [diff]
            flag = testingscripts.diff_checker(diffseries, tol = 1e-3, periods = 5)
        
        if verbose:
            print("itern = ",itern, " , diff = ", diff, " , x = ", x,  "sum F abs = ", np.sum(np.abs(FRomega[M+1:M+10]),axis=0))
        
        if verbose and (itern == 1 or itern % 100 == 0):
            fig, ax = plt.subplots(5)
            #title:
            fig.suptitle('Iteration = ' + str(itern) + ', beta = ' + str(beta))
            ax[0].plot(omega, -1*np.imag(GRomega), label = '-Im(GR)')
            ax[1].plot(omega, -1*np.imag(DRomega), label = '-Im(DR)')
            ax[0].set_xlabel(r'$\omega$')
            ax[0].set_ylabel(r'$-Im G^R(\omega)$')
            ax[0].set_xlim(-1,1)
            ax[1].set_xlabel(r'$\omega$')
            ax[1].set_ylabel(r'$-Im D^R(\omega)$')
            ax[1].set_xlim(-2,2)
            ax[2].plot(omega, -1*np.imag(FRomega), label = 'Sigma')
            ax[2].plot(omega, np.real(FRomega))
            ax[2].set_xlim(-2, 2)
            ax[2].set_xlabel(r'$\omega$')
            ax[2].set_ylabel(r'$-Im F^R(\omega)$')
            ax[3].plot(omega, -np.imag(Phiomega))
            ax[3].set_xlim(-2, 2)
            ax[3].set_xlabel(r'$\omega$')
            ax[3].set_ylabel(r'$-Im \Phi(\omega)$')
            ax[3].plot(omega, -np.real(Phiomega))
            ax[4].plot(omega, np.real(PiOmega))
            ax[4].set_xlim(-1, 1)
            ax[4].set_ylim(-1, 1)

        
            #bigger size and tight layout
            fig.set_size_inches(9,9)
            fig.tight_layout()

            plt.show()


    INFO = (itern, diff)
    # return (GRomega,DRomega,FRomega, SigmaOmega, PiOmega, Phiomega, INFO)
    return (GRomega,DRomega,FRomega, INFO)

def RE_SUPWH_iterator_conj_GF(GFs,grid,pars,beta,lamb,J, alpha,x = 0.01,err=1e-5,ITERMAX=150,eta=1e-6, verbose=True, diffcheck = False):
    '''
    signature:
    GFs = GDRomega, GODRomega, DDRomega, DODRomega, FDRomega, FODRomega
    GFs,grid,pars,beta,err=1e-5,ITERMAX=150,eta=1e-6, verbose=True, diffcheck = False
    grid is a list [M,omega,t]
    pars is a list [g,mu,r]
    '''
    GDRomega, GODRomega, DDRomega, DODRomega, FDRomega, FODRomega = GFs
    M,omega,t = grid
    g,mu,r = pars
    itern = 0
    xG, xD = x, x
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
         
            GDRoldomega,DDRoldomega,FDRoldomega = (1.0*GDRomega, 1.0*DDRomega, 1.0*FDRomega)
            GODRoldomega,DODRoldomega, FODRoldomega = (1.0*GODRomega, 1.0*DODRomega, 1.0*FODRomega)

            rhoGD = -1.0*np.imag(GDRomega)
            rhoDD = -1.0*np.imag(DDRomega)
            rhoGOD = -1.0*np.imag(GODRomega)
            rhoDOD = -1.0*np.imag(DODRomega)
            rhoFD = -1.0*np.imag(FDRomega)
            rhoFOD = -1.0*np.imag(FODRomega)
            SigmaDomega,PiDomega, PhiDomega = SupDav_rhotosigma(rhoGD,rhoDD, rhoFD, M,t,g,beta,BMf,eta, kappa=1,alpha=alpha)
            SigmaODomega,PiODomega, PhiODomega = SupDav_rhotosigma(rhoGOD,rhoDOD,rhoFOD, M,t,g,beta,BMf,eta, kappa=1,alpha=alpha)

            #dyson equations in real time
            detD = (r-(omega+1j*eta)**2 - PiDomega)**2 - (J-PiODomega)**2

            GDRomega = 0.5*x*((lamb - (omega+1j*eta) - -1*SigmaDomega + SigmaODomega)/((lamb - SigmaDomega + SigmaODomega + (omega+1j*eta))*(lamb - (omega+1j*eta) - -1*SigmaDomega + SigmaODomega) + (PhiDomega - PhiODomega)**2) - (lamb + (omega+1j*eta) + -1*SigmaDomega + SigmaODomega)/((lamb + SigmaDomega + SigmaODomega - (omega+1j*eta))*(lamb + (omega+1j*eta) + -1*SigmaDomega + SigmaODomega) + (PhiDomega + PhiODomega)**2)) + (1-x)*GDRoldomega

            FDRomega = 0.5*x*((-PhiDomega + PhiODomega)/((lamb - SigmaDomega + SigmaODomega + (omega+1j*eta))*(lamb - (omega+1j*eta) - -1*SigmaDomega + SigmaODomega) + (PhiDomega - PhiODomega)**2) - (PhiDomega + PhiODomega)/((lamb + SigmaDomega + SigmaODomega - (omega+1j*eta))*(lamb + (omega+1j*eta) + -1*SigmaDomega + SigmaODomega) + (PhiDomega + PhiODomega)**2)) + (1-x)*FDRoldomega

            GODRomega = 0.5*x*(-((lamb - (omega+1j*eta) - -1*SigmaDomega + SigmaODomega)/((lamb - SigmaDomega + SigmaODomega + (omega+1j*eta))*(lamb - (omega+1j*eta) - -1*SigmaDomega + SigmaODomega) + (PhiDomega - PhiODomega)**2)) - (lamb + (omega+1j*eta) + -1*SigmaDomega + SigmaODomega)/((lamb + SigmaDomega + SigmaODomega - (omega+1j*eta))*(lamb + (omega+1j*eta) + -1*SigmaDomega + SigmaODomega) + (PhiDomega + PhiODomega)**2)) + (1-x)*GODRoldomega

            FODRomega = 0.5*x*((PhiDomega - PhiODomega)/((lamb - SigmaDomega + SigmaODomega + (omega+1j*eta))*(lamb - (omega+1j*eta) - -1*SigmaDomega + SigmaODomega) + (PhiDomega - PhiODomega)**2) - (PhiDomega + PhiODomega)/((lamb + SigmaDomega + SigmaODomega - (omega+1j*eta))*(lamb + (omega+1j*eta) + -1*SigmaDomega + SigmaODomega) + (PhiDomega + PhiODomega)**2)) + (1-x)*FODRoldomega

            DDRomega = xval*((r - (omega+1j*eta)**2 - PiDomega)/detD) + (1-xval)*DDRoldomega

            DODRomega = xval*(-1.0*(J - PiODomega)/detD) + (1-xval)*DODRoldomega

            diffGD = np.sqrt(np.sum((np.abs(GDRomega-GDRoldomega))**2))#changed
            diffGOD = np.sqrt(np.sum((np.abs(GODRomega-GODRoldomega))**2)) #changed
            diffDD = np.sqrt(np.sum((np.abs(DDRomega-DDRoldomega))**2))#changed
            diffDOD = np.sqrt(np.sum((np.abs(DODRomega-DODRoldomega))**2)) #changed
            diffFD = np.sqrt(np.sum((np.abs(FDRomega-FDRoldomega))**2)) #changed
            diffFOD = np.sqrt(np.sum((np.abs(FODRomega-FODRoldomega))**2))

            #diffD = np.sum((np.abs(DRomega-DRoldomega))**2)
            # diff = (1/6)*(diffGD+diffDOD+diffDD+diffGOD+diffFD+diffFOD)
            diff = (1/3)*(diffGD+diffDD+diffFD)

 

            #diffG,diffD = diff,diff
            if diffcheck == True:
                diffseries += [diff]
                if itern >10:
                    flag = testingscripts.diff_checker(diffseries, tol = 1e-6, periods = 7)
            
            if verbose and (itern == 1 or itern % 100 == 0):
                fig, ax = plt.subplots(5)
                #title:
                fig.suptitle('Iteration = ' + str(itern) + ', beta = ' + str(beta))
                ax[0].plot(omega, -np.imag(GDRomega))
                ax[1].plot(omega, -np.imag(DDRomega))
                ax[0].set_xlabel(r'$\omega$')
                ax[0].set_ylabel(r'$-Im G^R(\omega)$')
                ax[0].set_xlim(0.,.4)
                ax[1].set_xlabel(r'$\omega$')
                ax[1].set_ylabel(r'$-Im D^R(\omega)$')
                ax[1].set_xlim(-2,2)
                ax[2].plot(omega, -1*np.imag(FDRomega), label = 'Sigma')
                ax[2].plot(omega, np.real(FDRomega))
                ax[2].set_xlim(-2, 2)
                ax[2].set_xlabel(r'$\omega$')
                ax[2].set_ylabel(r'$-Im F^R(\omega)$')
                ax[3].plot(omega, -np.imag(PhiDomega))
                ax[3].set_xlim(-2, 2)
                ax[3].set_xlabel(r'$\omega$')
                ax[3].set_ylabel(r'$-Im \Phi(\omega)$')
                ax[3].plot(omega, -np.real(PhiDomega))
                ax[4].plot(omega, np.real(PiDomega))
                ax[4].set_xlim(-1, 1)
                ax[4].set_ylim(-1, 1)

            
                #bigger size and tight layout
                fig.set_size_inches(9,9)
                fig.tight_layout()

                plt.show()

            
            if verbose:
                print("itern = ",itern, " , diff = ", diff, " , x = ", x,  "sum F abs = ", np.sum(np.abs(FDRomega[M+1:M+10]),axis=0))

    GFs = [GDRomega, GODRomega, DDRomega, DODRomega, FDRomega, FODRomega]
    INFO = (itern, diff, xval)
    return (GFs, INFO)

























