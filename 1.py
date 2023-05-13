""" %matplotlib notebook
%matplotlib tk """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp


def Rot2D(X,Y,Alpha):
    RotX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RotY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RotX, RotY


t = sp.Symbol('t')
r = t
phi = 2*t
x = r*sp.cos(phi)
y = r*sp.sin(phi)
Vx = sp.diff(x,t)
Vy = sp.diff(y,t)

F_x = sp.lambdify(t, x)
F_y = sp.lambdify(t, y)
F_Vx = sp.lambdify(t, Vx)
F_Vy = sp.lambdify(t, Vy)

t = np.linspace(0,10,1001)

x = F_x(t)
y = F_y(t)
Vx = F_Vx(t)
Vy = F_Vy(t)

Alpha_V = np.arctan2(Vy,Vx)

fig = plt.figure(figsize=[10,10])
ax = fig.add_subplot(1,1,1)
ax.axis('equal')
ax.set(xlim=[-12,12],ylim=[-12,12])

k_V = 0.5

ax.plot(x,y)
P = ax.plot(x[0],y[0],marker='o')[0]
V_line = ax.plot([x[0], x[0]+k_V*Vx[0]],[y[0], 
y[0]+k_V*Vy[0]],color=[1,0,0])[0]
a=0.1
b=0.03
x_arr = np.array([-a, 0, -a])
y_arr = np.array([b, 0, -b])
RotX, RotY = Rot2D(x_arr,y_arr,Alpha_V[0])
V_Arrow = ax.plot(x[0]+k_V*Vx[0] + RotX, y[0]+k_V*Vy[0] + 
RotY,color=[1,0,0])[0]

def TheMagicOfThtMovement(i):
    P.set_data(x[i],y[i])
    V_line.set_data([x[i], x[i]+k_V*Vx[i]],[y[i], y[i]+k_V*Vy[i]])
    RotX, RotY = Rot2D(x_arr, y_arr, Alpha_V[i])
    V_Arrow.set_data(x[i]+k_V*Vx[i] + RotX, y[i]+k_V*Vy[i] + RotY)
    return [P, V_line, V_Arrow]

kino = FuncAnimation(fig,TheMagicOfThtMovement, frames=len(t), interval=20)



plt.show()
print('hello\n')