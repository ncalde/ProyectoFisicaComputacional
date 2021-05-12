import numpy as np
import scipy.optimize as s_opt
import scipy.integrate as s_int
import matplotlib.pyplot as plt

h = 6.582*10**(-16)
m = 4.33227224*10**(-31)
V0 = 20
L = 1

def Griffiths(e_eval):
    z = np.sqrt(2*m*e_eval)*L/h
    z0 = np.sqrt(2*m*V0)*L/h
    f = lambda z: np.tan(z)-np.sqrt((z0/z)**2-1)
    total_ceros = []
    total_energia = []
    signo = np.sign(f(z))
    for i in range (len(signo)-1):
        if signo[i] + signo[i+1] == 0:
            cero = s_opt.brentq(f,z[i],z[i+1])
            total_ceros.append(cero)
    for i in range (len(total_ceros)):
        energia = (total_ceros[i]*h/L)**2*(1/(2*m))
        total_energia.append(energia)
    return total_energia

e_eval = np.linspace(0.01,V0,100)
lista = Griffiths(e_eval)
print(lista)
