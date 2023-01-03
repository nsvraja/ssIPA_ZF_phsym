import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import cmath
import math
from scipy import signal
from scipy import integrate

#Global constants

one=complex(1.0,0.0)
ii=complex(0.0,1.0)
zero=complex(0.0,0.0)
pi=np.pi

eta=1.0e-06
t=1.0  #Hopping energy scale

#Frequency Grid - Uniform from -10 to 10 with step_size=0.001
wm=10.0
step_size=1.0e-03
dw=step_size
w=np.arange(start=-wm,stop=wm+step_size,step=step_size,dtype=float)
nw=len(w)


#------------------------------------------------------
def gauss(z):
  G=-ii*np.sqrt(pi)*special.wofz(z/t)/t
  return G
#------------------------------------------------------

#------------------------------------------------------
def sem(z):
  G=np.zeros(len(z),dtype='complex')
  i=0
  for z1 in z:
    z2=cmath.sqrt(z1**2-4.0)
    G[i]=2.0/(z1-z2)
    if G[i].imag > 0.0:
      G[i]=2.0/(z1+z2)
    i+=1
  return G
#------------------------------------------------------

#------------------------------------------------------
#Read input parameters from file par.dat
f=open('par.dat','r')
content=f.readlines()
nums=content[1].split()  # U, soc, Temp
U=float(nums[0]); soc=float(nums[1]); Temp=float(nums[2])
nums=content[3].split()  # Bias value
bias=float(nums[0]); idos=int(nums[1])
#------------------------------------------------------



z=w*one+ii*eta
d0=0.1
if idos==1:
  r0=-gauss(z).imag/pi
  Vh=np.sqrt(d0/(pi*r0[nw//2+1]))
elif idos==2:
  r0=-sem(z).imag/pi
  Vh=np.sqrt(d0/(pi*r0[nw//2+1]))

Vh=Vh/np.sqrt(2.0)

  
print("Delta_0=",d0)

# Global t, w, dw, nw, step_size


#------------------------------------------------------
#CONVOLUTION FUNCTION
#------------------------------------------------------
#          infty
#y3(t)=int^      y1(tau)*y2(t-tau)
#         -infty

def convolve(y1,y2):
  n=len(y1)
  y1p=np.pad(y1,(n,n),'constant',constant_values=(0,0))
  y2p=np.pad(y2,(n,n),'constant',constant_values=(0,0))
  y3p=signal.convolve(y1p,y2p,mode='same')
  y3=y3p[n:2*n]*dw
  return y3
#------------------------------------------------------
  

#------------------------------------------------------
#Kramers-Kronig transform
#------------------------------------------------------
def kkt(rhot):
  wl=w+np.ones(len(w))*dw/2.0
  owl=1.0/wl
  return(convolve(rhot,owl))
#------------------------------------------------------


#------------------------------------------------------
def fermi_integral(E, T):
    if T>0:
      ee=np.exp(-abs(E)/T)
      return np.heaviside(-E,0.5)/(ee+1.0) + np.heaviside(E,0.5)*ee/(ee+1.0)
    elif T==0.0:
      return np.heaviside(-E,0.5)
      
#------------------------------------------------------




#------------------------------------------------------
#Prepare hybridization
def hybridization(idost,soct,biast):
  global hyb_L, hyb_R
  z=w*one+ii*eta
  mu_L=biast/2.0;mu_R=-biast/2.0
  #mu_L=0.0; mu_R=0.0
  if idost == 1:  # Gaussian DoS
    hyb_L=Vh**2*gauss(z-mu_L-soct)
    hyb_L+=Vh**2*gauss(z-mu_L+soct)

    hyb_R=Vh**2*gauss(z-mu_R-soct)
    hyb_R+=Vh**2*gauss(z-mu_R+soct)
  elif idost==2:  # Semi-elliptical DoS
    hyb_L=Vh**2*sem(z-mu_L-soct)
    hyb_L+=Vh**2*sem(z-mu_L+soct)

    hyb_R=Vh**2*sem(z-mu_R-soct)
    hyb_R+=Vh**2*sem(z-mu_R+soct)
  else: # Infinitely wide flat-band
    hyb_L=-ii*d0

    hyb_R=hyb_L
#------------------------------------------------------
    


#------------------------------------------------------
#Prepare the initial Green's functions
def NI_Green_fns(idost,soct,biast,Tempt):
  global ferm_L,ferm_R,delta_L,delta_R,ftilde,delta_LR,Gdr0,Gdl0
  z=w*one+ii*eta
  mu_L=biast/2.0;mu_R=-biast/2.0

  hybridization(idost,soct,biast)

  delta_L=-hyb_L.imag
  delta_R=-hyb_R.imag

  ferm_L=fermi_integral(w-mu_L,Tempt)
  ferm_R=fermi_integral(w-mu_R,Tempt)
  ftilde= (delta_L*ferm_L + delta_R*ferm_R)/(delta_L + delta_R) #Weighted Fermi fn

  delta_LR=delta_L*ferm_L
  delta_LR+=delta_R*ferm_R

  Gdr0=1.0/(z-hyb_L-hyb_R) # Retarded Green's function
  Gdl0=2.0*ii*(abs(Gdr0))**2*delta_LR  # Lesser Green's function

  #plt.plot(w,-Gdr0.imag/pi)
  #plt.plot(w, Gdl0.imag)
  #plt.show()


  nd0=dw*sum(Gdl0.imag)/(2.0*pi)
  norm=-dw*sum(Gdr0.imag)/pi

  #print('nd0,norm=',nd0,norm)
#------------------------------------------------------

#------------------------------------------------------
#SELF-ENERGY
def selfenergy(Ut):
  global SelfE_Ret, SelfE_L
  rhot=-Gdr0.imag/pi
  F1=np.flip(rhot)*ftilde
  F2=rhot*ftilde
  chi1=convolve(F1,F2)
  chi2=convolve(np.flip(F2),np.flip(F1))
  rho_SE=Ut**2*(convolve(np.flip(F1),np.flip(chi1))+convolve(F2,np.flip(chi2)))
  #Impose p-h symmetry
  rho_SE=0.5*(rho_SE+np.flip(rho_SE))
  Re_SE=kkt(rho_SE)
  SelfE_Ret=Re_SE-ii*pi*rho_SE
  
  chi1=convolve(np.flip(F2),np.flip(F1))
  SelfE_L=-2.0*ii*pi*Ut**2*convolve(F2,np.flip(chi1))
#------------------------------------------------------


#------------------------------------------------------
#Compute full Green's functions
def Full_G():
  global Gdr,Gdl
  Gdr=1.0/(1.0/Gdr0 - SelfE_Ret)
  Gdl=(abs(Gdr))**2*(Gdl0/(abs(Gdr0))**2 - SelfE_L)


  nd=dw*sum(Gdl.imag)/(2.0*pi)
  norm=-dw*sum(Gdr.imag)/pi
  #print('nd0,norm=',nd,norm)


  #plt.plot(w,-Gdr0.imag/pi)
  #plt.plot(w,-Gdr.imag/pi)
  #plt.plot(w, Gdl.imag)
  #plt.show()
#------------------------------------------------------

#------------------------------------------------------
# Obtain current
def current(idost,Ut,soct,biast,Tempt):
  curr=[]
  for bias in biast:
    NI_Green_fns(idost,soct,bias,Tempt)
    selfenergy(Ut)
    Full_G()
    jw=2.0*ii*(delta_L-delta_R)*Gdl
    jw+=-4.0*(delta_L*ferm_L-delta_R*ferm_R)*Gdr.imag
    jw=jw/2.0
    jval=dw*sum(jw.real)
    print('V,I=',bias,jval)
    curr.append(jval)

  return np.asarray(curr)
#------------------------------------------------------


#------------------------------------------------------#
#------------------------------------------------------#
#	Computing starts here
#------------------------------------------------------#
#------------------------------------------------------#

from scipy.interpolate import UnivariateSpline

print('wm= ',wm,' step_size= ',step_size,' nw= ',nw)
print('U= ',U,' soc= ',soc,' bias= ',bias,' Temp= ',Temp)

b_l=0.0
b_h=40.0
bias_vec=d0*np.arange(b_l,b_h,0.1)


U_vec=d0*np.array([0.0,4.0,8.0,12.0,20.0])
soc=0.0
Temp=0.0

for U in U_vec:
  print("U/Delta_0=",U/d0)

  curr_vec=current(idos, U, soc, bias_vec, Temp)
  fj='J_U'+str(U)+'.dat'
  dat=np.array([bias_vec/d0,curr_vec/d0])
  dat=dat.T
  np.savetxt(fj,dat,delimiter=' ')

