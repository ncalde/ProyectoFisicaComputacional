import numpy as np
import scipy.integrate as spint
import matplotlib.pyplot as plt

# Definición de parámetros / variables globales
a = 2  # un medio del largo del pozo de potencial
V0 = 83  # altura del pozo de potencial
c = 0.0483  # 2m/(hbarra**2)

def V(x):
    """
    Se define la función del potencial V(x)

    Parámetros de la función
    ------------------------
    x : variable x, posición

    Salida de la función
    --------------------
    valorV : valor numérico del potencial evaluado en x
    """
    if np.abs(x) > a:
        valorV = 0
    else:
        valorV = -V0
    return valorV

def F0F1(x, y, k):
    """
    Se definen las fuciones que correspoden al lado derecho de cada ecuación
    del sistema de EDO de primer orden y[0]'=F0(t, y) y y[1]'=F1(t, y)

    Parámetros de la función
    ------------------------
    x : variable x sobre la que se desarrolla el sistema de EDO
    y : arreglo con las dos variables utilizadas para generar el sistema de EDO
    k : número de onda, del cual se busca su eigenvalue

    Salida de la función
    --------------------
    valorF0F1 : arreglo con los valores de las funciones F0(t, y) y F1(t, y)
    evaluadas en x
    """
    valorV = V(x)
    valorF0F1 = [y[1], (k ** 2 + c * valorV) * y[0]]
    return valorF0F1


def main():
    # Se define el universo de valores de xIzq y xDer de interés, que se extienden
    # desde -xMax hasta xMatch y desde xMax hasta xMatch respectivamente, el número de valores es n en cada caso

    xMax = 10*a  # Se necesita xMax >> a
    xMatch = 0.0  # Su valor es irrelevante / La solución es independiente de este valor
    n = 1000
    xIzq = np.linspace(-xMax, xMatch, n)
    xDer = np.linspace(xMax, xMatch, n)

    # Se definen los valores iniciales de cada variable del sistema de EDO
    y0 = 0.0  # La función de onda evaluada en -xMax y en xMax tiende a cero
    y1 = 0.1

    # Se define una estimación inicial para E y k
    E = -V0 + V0/2  # Una buena estimación es E > -V0
    k = np.sqrt(c*np.abs(E))

    # Se calcula la solución del sistema de EDO utilizando la biblioteca SciPy
    metodo = 'RK45'
    solucionSistemaIzq = spint.solve_ivp(lambda x,y: F0F1(x,y,k), [-xMax, xMatch], [y0, y1], t_eval=xIzq, method=metodo)
    solucionSistemaDer = spint.solve_ivp(lambda x,y: F0F1(x,y,k), [xMax, xMatch], [y0, y1], t_eval=xDer, method=metodo)

    # Calcular delta
    # Estimar un nuevo E
    # Estimar un nuevo k
    # Repetir hasta que delta esta dentro de cierta tolerancia
    # Obtener k

    # Graficación
    plt.plot(solucionSistemaIzq.t, solucionSistemaIzq.y[0], 'r-', label='Psi izq')
    plt.plot(solucionSistemaDer.t, solucionSistemaDer.y[0], 'b-', label='Psi der')
    plt.xlabel("x (fm)")
    plt.grid(True)
    plt.title("Función de onda no normalizada")
    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()



