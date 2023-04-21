#------------------------------------------------#
#
# Blatt 2 - Musterlösung
# Aufgabe 1
# 20.04.2023
# Paul Keller
#
#------------------------------------------------#

from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


# ------ 1. Fourier Transformation ------
# (1) sin(x) schwingt mit einer Periode von 2pi,
# Ihre Amplituden sind 1 und -1, welche bei 1/2pi und 3/2pi liegen
# Und ihre Nullpunkte sind bei 0 und pi
# Ihre Frequenz entspricht 1/2pi (1 / Periode)
#
# Um eine vollständige Abtastung zu erhalten, müssen wir die Abtastrate
# mindestens 2x so hoch wählen, wie die höchste Frequenz ist, die wir abbilden wollen.
# In diesem Fall wäre dies 2 * 1/2pi = 1/pi
# Um unsere Abtastrate in einen Sampling-Abstand umzuwandeln,
# muss die Inverse der Abtastrate genommen werden, hier also 1/(1/pi) = pi
# Es ist zu raten, hier aber mehr als das Doppelte zu nehmen, im Beispiel hier 64.

# (2) sin(x) + (3*sin(2*x+1)-1) besteht aus zwei Sinus-Funktionen
# sin(x) hat die Frequzenz 1/2pi
# sin(2x+1) ist eine Verdopplung [x->2x] der Frequenz von sin(x)
# und somit die Periode pi, weshalb die Frequenz 1/pi ist
#
# wir wählen die höchste Frequenz beider Frequenzkomponenten aus, hier also 1/pi
# und multiplizieren sie mit 2, um eine vollständige Abtastung zu erhalten
# 2 * 1/pi= 2/pi
# Und unsere Sampling Abstand wäre also 1/(2/pi) = pi/2

def get_discret_values(frequence: float, function: Callable[[float], float], sampling_scale: float = 2, lower_bound: float = 0, upper_bound: float = 4 * np.pi):
    """Generates a list of discret values for a given function

    Args:
        frequence (float): the frequency of the function
        function (Callable[[float], float]): the function to be evaluated
        sampling_scale (float, optional): The sampling scale. Defaults to 2.
        lower_bound (float, optional): The lower bound. Defaults to 0.
        upper_bound (float, optional): The upper bound. Defaults to 4*np.pi.

    Returns:
        list[float]: the discret values
    """
    # calculate sampling distance by inversing the sampling rate
    sampling_distance = 1/(sampling_scale * frequence)
    # array of discrete values
    values = []
    # for every value in the range 0 to 4pi, with a step size of sampling_distance
    for x in np.arange(lower_bound, upper_bound, sampling_distance):
        # calculate the function at that point (equivalent to dirac delta)
        values.append(function(x))
    return values

def get_x_axis_values(frequence: float, sampling_scale: float = 2, lower_bound: float = 0, upper_bound: float = 4 * np.pi):
    """Generates a list of x axis values in a given range an frequency of the underlying function

    Args:
        frequence (float): the frequency of the underlying function
        sampling_scale (float, optional): the sampling scale. Defaults to 2.
        lower_bound (float, optional): the lower bound. Defaults to 0.
        upper_bound (float, optional): the upper bound. Defaults to 4*np.pi.

    Returns:
        list[float]: the x axis values
    """
    sampling_rate = 1/(frequence * sampling_scale)
    values = []
    for x in np.arange(lower_bound, upper_bound, sampling_rate):
        values.append(x)
    return values

def discrete_fourier(values: list[float]):
    """Calculate the DFT of a given list of values

    Args:
        values (list[float]): The list of values (with equal distance between them)

    Returns:
        npt.NDArray[np.complex64]: The DFT of the given values
    """
    # get the length of the list
    N = len(values)
    # create an empty list for the fourier values
    fourier = []
    # for every value in the list
    for k in range(N):
        # create a temporary sum
        temp_sum = 0
        # for every value in the list
        for n in range(N):
            # calculate DFT value and add to sum
            temp_sum += np.exp(-2j * np.pi * n * k / N)*values[n]
        # append the sum to the fourier list
        fourier.append(1/N * temp_sum)
    return np.array(fourier)

def inverse_discrete_fourier(fourier):
    """Calculates the inverse of the DFT of a given list of fourier values

    Args:
        fourier (npt.NDArray[np.complex64]): the given fourier values

    Returns:
        npt.NDArray[np.complex64]: The given fourier values
    """
    N = len(fourier)
    values = []
    for k in range(N):
        temp_sum = 0
        for n in range(N):
            # same formula as in discrete_fourier, but with the inverse of the exponent
            temp_sum += np.exp(2j * np.pi * n * k / N)*fourier[n]
        values.append(1/N * temp_sum)
    return np.array(values)

def plot_fourier(original: list[float],
                 fourier,
                 inverse_fourier,
                 numpy_fourier,
                 x_axis_values: list[float]):
    """Plots 4 graphs in one row

    Args:
        original (list[float]): The original discrete values
        fourier (npt.NDArray[np.complex64]): The fourier transformed values
        inverse_fourier (npt.NDArray[np.complex64]): The inverse fourier transformed values
        numpy_fourier (npt.NDArray[np.complex64]): the fourier transformed values calculated by numpy
        x_axis_values (list[float]): the x-axis values
    """
    # create sublot with 1 row, 4 columns, at position 1,1
    ax = plt.subplot(1,4,1)
    plt.title("Original")
    # plot the values to the corresponding x axis values
    ax.plot(x_axis_values, original)
    # this is really not necessary, but it transforms
    # the x axis to fractions of pi, rather than numbers
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda val,_: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'
    ))
    ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))

    # position 1,2
    # we only need the first half of the fourier values, 
    # because the second half is just the reflection of the first half
    # we also take only the real part and absolute value
    fourier_oneside = np.abs(np.real(fourier))[:len(fourier)//2]
    # since our function abruptly starts at 0, we ignore any wrong values above
    # a cutoff (defined as 3 here)
    fourier_oneside = np.array([0 if x > 3 else x for x in fourier_oneside])
    ax = plt.subplot(1,4,2)
    plt.title("Fourier")
    ax.plot(fourier_oneside)
    
    # position 1,3
    ax = plt.subplot(1,4,3)
    plt.title("Inverse Fourier")
    ax.plot(x_axis_values, inverse_fourier)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda val,_: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'
    ))
    ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))
    
    # position 1,4
    numpy_fourier_oneside = np.abs(np.real(numpy_fourier))[:len(numpy_fourier)//2]
    # the Cutoff for numpy fft is much higher
    numpy_fourier_oneside = np.array([0 if x > 1000 else x for x in numpy_fourier_oneside])
    ax = plt.subplot(1,4,4)
    plt.title("Numpy Fourier")
    ax.plot(numpy_fourier_oneside)
    
    plt.show()


def function1(x: float) -> float:
    """sin(x)
    shifted by 1 to the top, to get better output
    
    Args:
        x (float): x value

    Returns:
        float: result
    """
    return np.sin(x)+1

def function2(x: float) -> float:
    """sin(x) + (3 * sin(2 * x + 1) - 1)
    shifted by 5 to the top, to get better output
    Args:
        x (float): x value

    Returns:
        float: result
    """
    return np.sin(x) + (3 * np.sin(2 * x + 1) - 1) + 5

def main():
    # ------ 1. Fourier Transformation ------
    # define frequency and sampling scale for both functions
    frequency1 = 1 / (2 * np.pi)
    sampling_scale1: float = 64
    frequency2 = 1 / np.pi
    sampling_scale2: float = 64

    # get the discrete values for both functions
    discret_values1 = get_discret_values(frequency1, function1, sampling_scale1)
    discret_values2 = get_discret_values(frequency2, function2, sampling_scale2)

    # calculate fourier transform, numpy fourier transform and inverse fourier transform
    fourier_values1 = discrete_fourier(discret_values1)
    numpy_fourier1 = np.fft.fft(np.array(discret_values1))
    inverse_fourier1 = inverse_discrete_fourier(fourier_values1)
    fourier_values2 = discrete_fourier(discret_values2)
    numpy_fourier2 = np.fft.fft(np.array(discret_values2))
    inverse_fourier2 = inverse_discrete_fourier(fourier_values2)

    # get the x axis values for both functions
    x_axis_values1 = get_x_axis_values(frequency1, sampling_scale1)
    x_axis_values2 = get_x_axis_values(frequency2, sampling_scale2)

    # plot the graphs
    plot_fourier(discret_values1, fourier_values1, inverse_fourier1, numpy_fourier1, x_axis_values1)
    plot_fourier(discret_values2, fourier_values2, inverse_fourier2, numpy_fourier2, x_axis_values2)



if __name__ == "__main__":
    main()
