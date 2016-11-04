# odetools.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def solve_de(f, t0, x0, tmin, tmax, tdelta=0.01, args=()):
    def fsystem(x, t, *args):
        return np.array([f(t, x[0], *args)])

    tvecp = np.arange(t0, tmax + tdelta, tdelta)
    tvecn = np.arange(t0, tmin - tdelta, -tdelta)
    tvec = np.hstack([tvecn[-1:0:-1], tvecp])
 
    init = np.array([x0])
    xsolp = odeint(fsystem, init, tvecp, args=args)[:,0]
    xsoln = odeint(fsystem, init, tvecn, args=args)[:,0]
    xsol = np.hstack([xsoln[-1:0:-1], xsolp])

    return tvec, xsol


def solve_de_system(f, t0, x0, tmin, tmax, tdelta=0.01, args=()):
    def fsystem(x, t, *args):
        return f(t, x, *args)

    tvecp = np.arange(t0, tmax + tdelta, tdelta)
    tvecn = np.arange(t0, tmin - tdelta, -tdelta)
    tvec = np.hstack([tvecn[-1:0:-1], tvecp])

    xsolp = odeint(fsystem, x0, tvecp, args=args)
    xsoln = odeint(fsystem, x0, tvecn, args=args)
    xsol = np.vstack([xsoln[-1:0:-1], xsolp]).transpose()

    return tvec, xsol


def direction_field(fsystem, xbounds, ybounds, tvalue=0.0, args=(), **kw):
    xvalues = np.linspace(*xbounds)
    yvalues = np.linspace(*ybounds)
    xymesh = np.meshgrid(xvalues, yvalues)
    xvalues, yvalues = xymesh
    uvalues, vvalues = fsystem(tvalue, xymesh, *args)
    scale = np.sqrt(uvalues**2+vvalues**2)
    scale[scale == 0] = 1 
    uvalues /= scale
    vvalues /= scale
    return plt.quiver(xvalues, yvalues, uvalues, vvalues, **kw)


def slope_field(f, xbounds, ybounds, args=(), **kw):
    def fsystem(t, xyvec, *args):
        x, y = xyvec
        return [1, f(x, y)]
    return direction_field(fsystem, xbounds, ybounds, args=args, 
                           headwidth=0, headlength=0.001, 
                           headaxislength=0, pivot='middle', **kw)
