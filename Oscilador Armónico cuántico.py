import numpy as np
import scipy.integrate as spint
import matplotlib.pyplot as plt

h_daga = 1
m = 1
k = 1
w = (k/m)**(1/2)

def NivelEnergia (v):
    E = h_daga*w*(v+1/2)
    return E

def Potencial (x):
    V = (1/2)*k*x**2
    return V

def ODE (x,y,v):
    """
    Se define el sistema de ecuaciones diferenciales.
    E = lista de energías estables
    """
    dy = (y[1], (2*m/h_daga**2)*(Potencial(x)-NivelEnergia(v))*y[0])
    return dy
colores = ['b-','g-','r-','c-','m-','k-','orange','lawngreen']
metodo = 'RK45'
v = [0,1,2,3,4,5,6,7]
x_lista_izq = np.linspace(-10,0,1000)
x_lista_der = np.linspace(10, 0, 1000)
for i in range(len(v)-1):
    color = colores[i]
    solucion = spint.solve_ivp(lambda x,y: ODE(x,y,v[i]), [-10, 0], [0,1], t_eval=x_lista_izq, method=metodo)
    solucion2 = spint.solve_ivp(lambda x,y: ODE(x,y,v[i]), [10, 0], [0,1], t_eval=x_lista_der, method=metodo)
    norma = np.linalg.norm(solucion.y[0]**2)
    solucionnormalizadaIzq = ((solucion.y[0]**2)/norma)+i
    solucionnormalizadaDer = ((solucion2.y[0]**2)/norma)+i
    plt.plot(x_lista_izq, solucionnormalizadaIzq, color)
    plt.plot(x_lista_der, solucionnormalizadaDer, color, label = ('Energía: ', NivelEnergia(v[i])))

xtotal = np.linspace(-4,4,2000)
puntosPotencial = []
for j in range (len(xtotal)):
    Potencial_X = Potencial(xtotal[j])
    puntosPotencial.append(Potencial_X)

plt.plot(xtotal,puntosPotencial, label = ('Potencial'))
plt.xlabel("x")
plt.xlabel("y")
plt.grid(True)
plt.title("Distribución de probabilidad normalizada")
plt.legend(loc='upper right')
plt.show()

