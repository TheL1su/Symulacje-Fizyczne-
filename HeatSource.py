import numpy as np

class SingleHeatSource:
    def __init__(self):
        pass

    def heat_source(self,x,y,t,Ql=15,r=3):
        def alpha(t):
            return 10-100*t
        def beta(t):
            return 0
        inside = -((x-alpha(t))**2 + (y-beta(t))**2)/r**2
        return Ql*np.exp(inside)
    
class MultiHeatSource:
    def __init__(self):
        pass

    def calculate(self,x,y,t,Ql=15,r=2):
        def alpha1(t):
            return 10-100*t
        
        def alpha2(t):
            return -10+100*t
        
        def beta(t):
            return 0
        inside1 = -((x-alpha1(t))**2 + (y-beta(t))**2)/r**2
        inside2 = -((x-alpha2(t))**2 + (y-beta(t))**2)/r**2
        return Ql*np.exp(inside1) + Ql*np.exp(inside2)
    
    def heat_source(self,x,y,t,Ql=15,r=2):
        return self.calculate(x,y,t)

class CircleHeatSource:
    def __init__(self):
        pass

    def heat_source(self,x,y,t,Ql=15,r=2):
        def alpha(t):
            return 6*np.cos(np.pi * t*8)
        def beta(t):
            return 6*np.sin(np.pi * t*8)
        inside = -((x-alpha(t))**2 + (y-beta(t))**2)/r**2
        return Ql*np.exp(inside)