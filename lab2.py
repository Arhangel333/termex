import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

def Rot2D(X, Y, Alpha):#rotates point (X,Y) on angle alpha with respect to Origin
    RX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RX, RY

def Rot2DS(X, Y, Alpha):#rotates point (X,Y) on angle alpha with respect to Origin
    RX = X*sp.cos(Alpha) - Y*sp.sin(Alpha)
    RY = X*sp.sin(Alpha) + Y*sp.cos(Alpha)
    return RX, RY

def Prizma(x0, y0, a, b):#return lists for a prism
    PX = [x0-4/5*a, x0+(1/5)*a, x0+(1/5)*a, x0-4/5*a]
    PY = [y0+(1/2)*b, y0+(1/2)*b, y0-(1/2)*b, y0+(1/2)*b]
    return PX, PY
#circle radius
radius = 5
nm = 19
def Circle(x0, y0, phi, rad):#return lists for a Circle
    SX = [x0+nm*rad*sp.cos(t)/(6*math.pi) for t in np.linspace(0, 6*math.pi+(1/2)*math.pi+phi,100)]
    SY = [y0+nm*rad*sp.sin(t)/(6*math.pi) for t in np.linspace(0, 6*math.pi+(1/2)*math.pi+phi,100)]
    return SX, SY

#defining parameters
#the angle of the plane (and the prism)
alpha = 0
# size of the prism
a = 10
b = a*sp.tan(alpha)
#size of the beam
l = 4.5
l1 = 5

#defining t as a symbol (it will be the independent variable)
t = sp.Symbol('t')
#here x, y, Vx, Vy, Wx, Wy, xC are functions of 't'
s = 4*sp.cos(3*t)
phi = 4*sp.sin(t-10)
psi = s*sp.sin(t)

#Motion of the prism with a Circle (translational motion)
Xspr = s*sp.cos(alpha)+4/5*a
Yspr = -s*sp.sin(alpha)-1/2*b

VmodSignPrism = sp.diff(s, t)
VxSpr = VmodSignPrism*sp.cos(alpha)
VySpr = -VmodSignPrism*sp.sin(alpha)

WmodSignPrism = sp.diff(VmodSignPrism, t)
WxSpr = WmodSignPrism*sp.cos(alpha)
WySpr = -WmodSignPrism*sp.sin(alpha)

#Motion of the beam with respect to a Circle (A - the farthest point on the beam from the Circle)
xA = Xspr+l*sp.sin(s/4)
yA = Yspr-l*sp.cos(s/4)

xB = s + 8 + l1*sp.sin(s/3)
yB = l1*sp.cos(s/3)


omega = sp.diff(phi,t)
omega1 = sp.diff(psi, t)


VxA = VxSpr - omega*l*sp.cos(phi)
VyA = VySpr - omega*l*sp.sin(phi)

VxB = VxSpr - omega1*l*sp.cos(psi)
VyB = VySpr - omega1*l*sp.sin(psi)



VxArel = - omega*l*sp.cos(phi)
VyArel = - omega*l*sp.sin(phi)

VxBrel = - omega1*l*sp.cos(psi)
VyBrel = - omega1*l*sp.sin(psi)

#constructing corresponding arrays
T = np.linspace(0, 20, 1000)
XSpr = np.zeros_like(T)
YSpr = np.zeros_like(T)
VXSpr = np.zeros_like(T)
VYSpr = np.zeros_like(T)
Phi = np.zeros_like(T)
XA = np.zeros_like(T)
YA = np.zeros_like(T)
VXA = np.zeros_like(T)
VYA = np.zeros_like(T)
VXArel = np.zeros_like(T)
VYArel = np.zeros_like(T)
Psi = np.zeros_like(T)
XB = np.zeros_like(T)
YB = np.zeros_like(T)
VXB = np.zeros_like(T)
VYB = np.zeros_like(T)
VXBrel = np.zeros_like(T)
VYBrel = np.zeros_like(T)


L_Xspr = sp.lambdify(t, Xspr)
L_Yspr = sp.lambdify(t, Yspr)
L_VxSpr = sp.lambdify(t, VxSpr)
L_VySpr = sp.lambdify(t, VySpr)
L_phi = sp.lambdify(t, phi)
L_xA = sp.lambdify(t, xA)
L_yA = sp.lambdify(t, yA)
L_VxA = sp.lambdify(t, VxA)
L_VyA = sp.lambdify(t, VyA)
L_VxArel = sp.lambdify(t, VxArel)
L_VyArel = sp.lambdify(t, VyArel)

L_psi = sp.lambdify(t, psi)
L_xB = sp.lambdify(t, xB)
L_yB = sp.lambdify(t, yB)
L_VxB = sp.lambdify(t, VxB)
L_VyB = sp.lambdify(t, VyB)
L_VxBrel = sp.lambdify(t, VxBrel)
L_VyBrel = sp.lambdify(t, VyBrel)

#filling arrays with corresponding values
for i in np.arange(len(T)):
    XSpr[i] = L_Xspr(T[i])
    YSpr[i] = L_Yspr(T[i])
    VXSpr[i] = L_VxSpr(T[i])
    VYSpr[i] = L_VySpr(T[i])
    Phi[i] = L_phi(T[i])
    XA[i] = L_xA(T[i])
    YA[i] = L_yA(T[i])
    VXA[i] = L_VxA(T[i])
    VYA[i] = L_VyA(T[i])
    VXArel[i] = L_VxArel(T[i])
    VYArel[i] = L_VyArel(T[i])



    Psi[i] = L_psi(T[i])
    XB[i] = L_xB(T[i])
    YB[i] = L_yB(T[i])
    VXB[i] = L_VxB(T[i])
    VYB[i] = L_VyB(T[i])
    VXBrel[i] = L_VxBrel(T[i])
    VYBrel[i] = L_VyBrel(T[i])

#here we start to plot
fig = plt.figure(figsize=(17, 8))

ax1 = fig.add_subplot(1, 2, 1)
ax1.axis('equal')
ax1.set(xlim=[XSpr.min()-2*a, XSpr.max()+2*a], ylim=[YSpr.min()-2*a, YSpr.max()+2*a])
#ax1.set(xlim=[-5, 5], ylim=[-5, 5])

#plotting a plane
ax1.plot([XSpr.min()-a, XSpr.max()+a], [-(XSpr.min()-a)*sp.tan(alpha) - radius, -(XSpr.max()+a)*sp.tan(alpha) - radius],
'black')

#plotting initial positions

#plotting a prism
#PrX, PrY = Prizma(XSpr[0],YSpr[0],a,b)
#Prism = ax1.plot(PrX, PrY, 'yellow')[0]

#plotting a Circle
SpX, SpY = Circle(XSpr[0],YSpr[0], Phi[0], radius)
Spr, = ax1.plot(SpX, SpY, 'black')

#plotting a beam
#plotting beam first point
XBm = XB
YBm = YB

Beam, = ax1.plot([XBm[0], XB[0]], [YBm[0], YB[0]], 'blue')
Chain, = ax1.plot([XBm[0], XB[0]], [YBm[0], YB[0]], 'green')

#plotting chain on beam
def mid(x, y):
    return (x + y)/2
def subdiv(arr):# подразделяет массив( разбивает пружинку на цепи)
    i = 0
    x = len(arr)
    while i < x:
        """print(i, x, '\n')
        print(arr, '\n')"""
        arr.insert(i+1, mid(arr[i], arr[i + 1]))
        i = i + 2
    return arr

def multisub(arr, n = 1):
    for i in range(0, n):
        subdiv(arr)
    return arr

def bum(arr, r = 1):#разбрасывает цепи пружинки вверх и вниз
    if(len(arr) > 1):
        #print(arr[0], arr[1], arr[1] - arr[0])
        x = r
        #print(x)
        for it in range(0, len(arr)):
            if(it%2 == 0):
                arr[it] -= x
            else:
                arr[it] += x
        return arr
    else:
        return 0

def spring(arrx, arry, n = 1, rbum = 1):
    al = np.arctan2(arry[len(arry) - 1] - arry[0], arrx[len(arrx) - 1] - arrx[0])
    arrx = bum(multisub(arrx, n), rbum*np.cos(al))
    arry = bum(multisub(arry, n), rbum* np.cos(al))
    return arrx, arry

def centring(x, y):
    sx = len(x)
    sy = len(y)
    mx = (max(x) - min(x))/2
    my = (max(y) - min(y))/2
    if(sx > 1):
        c = sx//2
        for i in range(0, sx):
            dif = x[c] + mx
            if(x[c] > 0):
                x[i] -= dif
            elif(x[c] < 0):
                x[i] += dif
    if (sy > 1):
        c = sy // 2
        for i in range(0, sy):
            dif = y[c] + my
            if (y[c] > 0):
                y[i] -= dif
            elif (y[c] < 0):
                y[i] += dif
    return x, y



drow = 15
long = 5
chx = [XBm[0]-drow + long, XB[0] -drow]
chy = [YBm[0], YB[0]]
centring(chx, chy)
ch = [chx, chy]
#ch = [chx, chy]
#print(chx)
#print(bum(chy))


#for i in range(0, 5):
    #print(ch[i])

"""
n = 3
chx, chy = spring(chx, chy, 1)
chx = np.array(chx)
chy = np.array(chy)
chx, chy = Rot2D(chx, chy, Alpha_V[0])"""

#Chain1 = ax1.plot(multisub(chx, n), bum(multisub(chy, n), 1), 'green')
#Chain1 = ax1.plot(chx, chy, 'green')

varphi=np.linspace(0, 2*math.pi, 20)
r=l/10
Point = ax1.plot(XA[0]+r*np.cos(varphi), YA[0]+r*np.sin(varphi),color=[1, 0, 1])[0]

ax2 = fig.add_subplot(4, 2, 2)
ax2.plot(T, VXSpr)

ax2.set_xlabel('T')
ax2.set_ylabel('VXPrizm')

ax3 = fig.add_subplot(4, 2, 4)
ax3.plot(T, VYSpr)

ax3.set_xlabel('T')
ax3.set_ylabel('VYPrizm')

ax4 = fig.add_subplot(4, 2, 6)
ax4.plot(T, VXA)

ax4.set_xlabel('T')
ax4.set_ylabel('VXA')

ax5 = fig.add_subplot(4, 2, 8)
ax5.plot(T, VYA)

ax5.set_xlabel('T')
ax5.set_ylabel('VYA')

plt.subplots_adjust(wspace=0.3, hspace = 0.7)

Alpha_V = np.arctan2(XB-XA, YB-YA)

#function for recounting the positions
def anima(i):
    #PrX, PrY = Prizma(XSpr[i],YSpr[i],a,b)
    #Prism.set_data(PrX, PrY)
    SpX, SpY = Circle(XSpr[i],YSpr[i], Phi[i], radius)
    Spr.set_data(SpX, SpY)
    x_arr, y_arr = [XBm[i], XA[i]], [YBm[i], YA[i]]
    x_arr, y_arr = centring(x_arr, y_arr)
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    if(i>0):
        RotYA = YA[i] - YA[i-1]
        RotYB = YBm[i] - YBm[i-1]
        RotXA = XA[i] - XA[i - 1]
        RotXB = XBm[i] - XBm[i - 1]
    else:
        RotYA = RotYB = RotXA = RotXB = 0


    #Beam.set_data(spring([XBm[0] + RotXB, XA[0] + RotXA], [YBm[0] + RotYB, YA[0] + RotYA], 2, 1))



    x, y = spring([XBm[i], XA[i]], [YBm[i], YA[i]], 3, 2)
    #x, y = spring([XBm[i] , (XBm[i] + XA[i])/2, XA[i]], [YBm[i] , (YBm[i] + YA[i])/2, YA[i] ], 1, 10)

    x[0] = XBm[i]
    y[0] = YBm[i]
    #x.insert(len(x) - 1, XBm[i])
    #y.insert(len(y) - 1, YBm[i])
    inn = len(x)
    x.insert(inn, XA[i])
    inn = len(y)
    y.insert(inn, YA[i])
    Chain.set_data(x, y)


    Point.set_data(XA[i]+r*np.cos(varphi), YA[i]+r*np.sin(varphi))
    return Spr, Point, Chain#, Beam

# animation function
anim = FuncAnimation(fig, anima, frames=1000, interval=10, blit=True)

plt.show()
