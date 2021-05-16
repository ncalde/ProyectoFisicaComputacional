import numpy as np
import scipy.optimize as s_opt
import scipy.integrate as s_int
import matplotlib.pyplot as plt
import random

h = 6.582*10**(-16)
m = 4.33227224*10**(-31)
V0 = 10
L = 1.66667

def Griffiths(e_eval):
    """
    Esta función está basada en el Griffiths de Introducción a la Mecánica Cuántica.
    Los ceros de la función matemática del Griffiths se puede relacionar con las energías en las que el sistema
    no diverge en los extremos de la función de onda.
    La función de Python busca en que punto de la función matemática hay un cambio de signo y busca un cero Z, este cero Z
    será despejado para encontrar su energía correspondiente.
    """
    z = np.sqrt(2*m*e_eval)*L/h
    z0 = np.sqrt(2*m*V0)*L/h
    f = lambda z: np.tan(z)-np.sqrt((z0/z)**2-1)  #Se define la función para casos donde la ecuacion de onda es simetrica
    total_ceros = [] #En esta lista se almacenan los ceros de la función, es decir los Z que dependen de E
    total_energia = [] #En esta lista se almacenan las E despejadas de los Z
    signof = np.sign(f(z)) #Se almacenan los signos de f(z)
    for i in range (len(signof)-1): #Se encuentran los ceros
        if signof[i] + signof[i+1] == 0:
            cero = s_opt.brentq(f,z[i],z[i+1])
            total_ceros.append(cero)
    for i in range (0,len(total_ceros),2): #Se despejan los E y se eliminan aquellos que corresponden a soluciones de la forma (2n-1)pi puesto que ahí diverge la tangente
        energia = (total_ceros[i]*h/L)**2*(1/(2*m))
        total_energia.append(energia)
    total_ceros = []
    g = lambda z: -1/np.tan(z) - np.sqrt((z0 / z) ** 2 - 1) #Se repite el procedimiento con g para los casos de función de onda antisimétrica.
    signog = np.sign(g(z))
    for j in range (len(signog)-1):
        if signog[j]+signog[j+1]==0:
            cero = s_opt.brentq(g,z[j],z[j+1])
            total_ceros.append(cero)
    for i in range (0,len(total_ceros),2): #Se despejan los E y se eliminan aquellos que corresponden a soluciones de la forma (2n)pi puesto que ahí diverge la cotangente
        energia = (total_ceros[i]*h/L)**2*(1/(2*m))
        total_energia.append(energia)
    return total_energia



def V(x):
    """
    Se establece la barrera de potencial
    """
    if np.abs(x)>L:
        V = V0
    else:
        V = 0
    return V

def ODE (x,y,E):
    """
    Se define el sistema de ecuaciones diferenciales.
    E = lista de energías estables
    """
    dy = (y[1], (2*m/h**2)*(V(x)-E)*y[0])
    return dy

e_eval = np.linspace(0.01,V0,100) #Se crea la lista con las posibles energías estables
lista = Griffiths(e_eval) #Se encuentran las energías estables
lista.sort() #La lista se acomoda de mayor a menor

xIzq = np.linspace(-2,0,1000) #Se crea un conjunto de punto para el lado izquierdo
xDer = np.linspace(2,0,1000) #Se crea un conjunto de punto para el lado derecho

colores = ['b-','g-','r-','c-','m-','k-','orange','lawngreen','pink','peru'] #Lista colores para asignar a cada energía una vez que se grafiquen


#Se calcula la solución del sistema de ecuaciones y se grafica cada resultado con un color aleatorio.
for k in range (len(lista)-1):
    color = colores[random.randint(0,len(colores)-1)]
    solucionIzquierda = s_int.solve_ivp(lambda x,y: ODE(x,y,lista[k]),[-2,0],[0,1], t_eval=xIzq)
    solucionDerecha = s_int.solve_ivp(lambda x,y: ODE(x,y,lista[k]),[2,0],[0,-1], t_eval=xDer)
    plt.plot(xDer, solucionDerecha.y[0], color, label = ('Energía: ', lista[k], 'eV'))
    plt.plot(xIzq, solucionIzquierda.y[0], color)
plt.xlabel("x (fm)")
plt.grid(True)
plt.title("Función de onda no normalizada para distintas energías")
plt.legend(loc='upper right')

plt.show()





