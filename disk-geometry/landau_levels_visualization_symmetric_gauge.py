#!/usr/bin/env python
# coding: utf-8
import scipy.special as ss
import numpy as np
# import matplotlib.pyplot as plt
import scipy.optimize as so
import scipy.integrate as si
# plt.rcParams["figure.figsize"] = (16,9)


def associated_laguerre(n, m):
    """ Associated Laguerre Polynomials of type L_{n}^{m}.""""
    return lambda x: ss.poch(m+1, n)*ss.hyp1f1(-n, m+1, x)/ss.factorial(n)
def LL_pdf(r, n, m):
    """ Probability Distribution Function as a functions of the radius for a Landau Level n at angular momentum m."""
    return (ss.factorial(n)/ss.factorial(n+m)/2**m)*np.exp(-r**2/2)*r**(2*m+1)*(associated_laguerre(n, m)(r**2/2))**2


# rgrid = np.linspace(1e-3, 20.0, 10**4)
# for n in range(0, 10):
#     for m in range(-n, -n+5):
#         plt.plot(rgrid, np.vectorize(lambda r: LL_pdf(r, n, m))(rgrid), label=str(m))
#     plt.xlabel(r"$r$")
#     plt.ylabel(r"$r|\psi(\mathbf{r})|^{2}$")
#     plt.title("Landau Levels in Disk Geometry for the n=%d level"%(n))
#     plt.legend()
#     plt.savefig("./landau_levels_disk_geometry_n_%d.svg"%(n), bbox_inches="tight")
#     plt.close()




def radius_estimator(n, m, α): ### Estimates radius for which α amount of pdf is contained for n,m ll.
    curve = lambda R: si.quad(lambda r: LL_pdf(r, n, m), 1e-6, R)[0] - α
    return so.root(curve, np.sqrt(2*(m+n) + 1))

# data = {}
# for n in range(0, 3):
#     data[n] = []
#     for m in range(-n, -n+5):
#         sol = radius_estimator(n, m, 0.9)
#         data[n].append((m, sol.x))


# data

