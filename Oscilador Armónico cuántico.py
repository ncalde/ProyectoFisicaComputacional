import numpy as np
import scipy.integrate as spint
import matplotlib.pyplot as plt


h_daga = 1  # constante de Planck reducida
m = 1  # masa de la partícula
k = 1  # constante elástica
w = (k / m) ** (1 / 2)  # frecuencia angular


def NivelEnergia(n):
    """
    Se calcula la energía del estado n.
    Entrada     n: int, nivel de energía
    Salida      E: float, valor numérico de la energía para ese estado
    """
    E = h_daga * w * (n + 1 / 2)
    return E


def Potencial(x):
    """
    Se define el potencial para el oscilador armónico cuántico.
    Entrada     x: posición en el eje
    Salida      V: valor del potencial evaluado en esa posición
    """
    V = (1 / 2) * k * x ** 2
    return V


def ODE(x, y, n):
    """
    Se definen las funciones que correspoden al lado derecho de cada ecuación
    del sistema de EDO de primer orden y[0]'=F0(x, y) y y[1]'=F1(x, y)
    Entradas    x: float, variable de posición sobre la que se desarrolla el sistema
                y: lista de floats, variables utilizadas para generar el sistema
                    y[0]: función de onda psi(x)
                    y[1]: derivada de la función de onda psi'(x)
                n: int, nivel de energía
    Salidas     dy: lista con los valores de las funciones y[0]' y y[1]' evaluadas en x
    """
    dy = (y[1], (2 * m / h_daga ** 2) * (Potencial(x) - NivelEnergia(n)) * y[0])
    return dy


colores = ['b-', 'g-', 'r-', 'c-', 'm-', 'k-', 'orange','lawngreen']  # Lista de colores, su único propósito es para graficación.
metodo = 'RK45'  # Indica el método por el cual se solucionaran las EDO.
n = [0, 1, 2, 3, 4, 5, 6, 7]  # Lista de niveles de energía a evaluar
x_lista_izq = np.linspace(-10, 0, 1000)  # Valores negativos de x a evaluar, se escoge un xMin lejano al potencial (tiende a -inf)
x_lista_der = np.linspace(10, 0, 1000)  # Valores positivos de x a evaluar, se escoge un xMax lejano al potencial (tiende a inf)


# El siguiente ciclo resuelve la ecuación de Schrodinger para todos los niveles de energía deseados. Primero encuentra la solución para
# x negativos y positivos por separado. Luego ambos resultados son elevados al cuadrado y normalizados para obtener la distribución de
# probabilidad. Ambos resultados son luego juntados en un solo gráfico.
for i in range(len(n)):
    color = colores[i]
    # Para resolver el sistema de ecuaciones diferenciales se requieren los valores iniciales de y[0] y y[1]:
        # Para y[0]: se considera que xMin y xMax tienden a -inf y inf, por lo que el valor de la función de onda es 0
        # Para y[1]: se considera que la derivada de la función de onda tiende a cero, por lo que se toma un valor de 0.1
    solucion = spint.solve_ivp(lambda x, y: ODE(x, y, n[i]), [-10, 0], [0, 0.1], t_eval=x_lista_izq, method=metodo)
    solucion2 = spint.solve_ivp(lambda x, y: ODE(x, y, n[i]), [10, 0], [0, 0.1], t_eval=x_lista_der, method=metodo)
    norma = np.linalg.norm(solucion.y[0] ** 2)
    solucionnormalizadaIzq = ((solucion.y[0] ** 2) / norma) + i + 0.5  # Se suma para lograr el n+1/2
    solucionnormalizadaDer = ((solucion2.y[0] ** 2) / norma) + i + 0.5  # Se suma para lograr el n+1/2
    plt.plot(x_lista_izq, solucionnormalizadaIzq, color)
    plt.plot(x_lista_der, solucionnormalizadaDer, color, label=(n[i]))


# Este bloque de código está hecho para obtener las herramientas necesarias para graficar la barrera de potencial.
xtotal = np.linspace(-5, 5, 2000)
puntosPotencial = []
for j in range(len(xtotal)):
    Potencial_X = Potencial(xtotal[j])
    puntosPotencial.append(Potencial_X)

# Se grafica el potencial sobre las soluciones para los distintos niveles de energía.
plt.plot(xtotal, puntosPotencial, label=('Potencial'))
plt.xlabel("x")
plt.ylabel("Energía ($\hbar\omega$)")
plt.grid(True, 'both')
plt.title("Distribución de probabilidad normalizada")
plt.legend(title="Nivel de energía", loc='lower right', fontsize=8)
plt.xlim(-10, 10)
plt.ylim(0, 8)
plt.show()
