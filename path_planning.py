from os import path
from abco import *
from create_grid_map import *
import numpy as np
import math
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# def path_penalty(obs, Px, Py):
#     err = 0
#     for i in range(len(obs)):
#         data = obs[i]
#         xc, yc = data[:2]
#         d = np.sqrt((Px - xc) ** 2 + (Py - yc) ** 2)
#         r, Kv = data[2:]
#         if(r >= d):
#             if(d>0):
#                 penalty = Kv/d
#             elif(d == 0):
#                 penalty = 100
#         else:
#             penalty = 0.0
#         err += np.nanmean(penalty)
#     return err
def path_penalty(obs, Px = [0,0], Py = [0,0]):   
    len_ = len(Px)
    err_1 = 0
    for i in range(len_-1):
        err = 0
        P_1 = np.array([Px[i], Py[i]])
        P_2 = np.array([Px[i+1], Py[i+1]])
        for j in range(len(obs)):
            data = obs[j]
            xc, yc = data[:2]
            P_3 = np.array([xc, yc])
            d = abs(np.cross(P_2-P_1, P_3-P_1)/np.linalg.norm(P_2-P_1))
            r, Kv = data[2:]
            penalty = 0.0
            if(r >= d):
                penalty = 100
            # else:
            #     penalty = 0.0
            err += np.nanmean(penalty)
        err_1 += err
    return err_1
def center_point(Px = [0,0,0,0]):
    n = int(len(Px))
    x = []
    for i in range(n-1):
        x.append(Px[i])
        x.append((Px[i] + Px[i+1])/2)
    x.append(Px[n-1])
    return x


class Path:
    def __init__(self, start=None, end=None, limits=None,nPts = None):      
        self.start = [0,0] if (start is None) else np.asarray(start)
        self.end = [10,10] if (end is None) else np.asarray(end)
        self.limits = 10 if (limits is None) else np.asarray(limits)
        self.nPts = 10 if (nPts is None) else np.asarray(nPts)
        self.obs = []

    def set_start(self, x, y):
        self.start = np.array([x,y])
        print(self.start)

    def set_end(self, x, y):
        self.end = np.array([x,y])
        print(self.end)

    def set_obs(self, x = 0.0, y = 0.0):
        r = 0.75
        Kv = 10.0
        data = (x, y, r, Kv)
        self.obs.append(data)

    # create map
    def create_map(self):
        G = GridMap(self.limits)
        grid = G.create_grid_map()
        for i in range(self.limits):
            for j in range(self.limits):
                if(grid[i][j] == 2):
                    self.set_start(i,j)
                if(grid[i][j] == 3):
                    self.set_end(i,j)
                if(grid[i][j] == 1):
                    self.set_obs(i,j)
    def build_init(self):
        xs = self.start[0]
        ys = self.start[1]
        xe = self.end[0]
        ye = self.end[1]

        Px = np.linspace(xs, xe, self.nPts+2)
        Py = np.linspace(ys, ye, self.nPts+2)

        _init = np.concatenate((Px[1:-1], Py[1:-1]))
        print(_init)

        return _init
    #function to calculate the path length
    def func(self, _init = [0,0]):
        # _init = self.build_init()
        Xs = self.start[0]
        Ys = self.start[1]
        Xe = self.end[0]
        Ye = self.end[1]
        x = np.block([Xs, _init[:self.nPts], Xe])
        y = np.block([Ys, _init[self.nPts:], Ye])

        x_new = center_point(x)
        y_new = center_point(y)
        # x_new_1 = center_point(x_new)
        # y_new_1 = center_point(y_new)
        # Path length
        dX = np.diff(x_new)
        dY = np.diff(y_new)

        L = np.sqrt(dX ** 2 + dY ** 2).sum()
        
        # err = 0
        # for i in range(int(len(x_new_1))):
        #     # error = path_penalty(self.obs, _init[i], _init[i+self.nPts])
        #     error = path_penalty(self.obs, x_new_1[i], y_new_1[i])
        #     err += error    
        
        err = path_penalty(self.obs, x_new, y_new)
        
        L = L * (1.0 + err)
        return L
    def optimize(self):
        LB = np.zeros(2*self.nPts)
        UB = np.ones(2*self.nPts)
        UB *= self.limits
        self.abc = artificial_bee_colony_optimization(min_values= LB, max_values= UB, target_func= self.func)
        # return self.abc  
    def plot_path(self):
        # Coordinates of the discretized path
        # Px = self.sol[3]
        # Py = sel.sol[4]

        # # Plot the spline
        # ax.plot(Px[0, :], Py[0, :], lw=0.50, c='r')

        # Plot the internal breakpoints
        best_sol = self.abc
        nPts = len(best_sol) // 2
        X = np.block([self.start[0], best_sol[:nPts], self.end[0]])
        Y = np.block([self.start[1], best_sol[nPts:-1], self.end[1]])
        plt.plot(X, Y, ms=4, c='b')

        # Plot start position
        plt.plot(self.start[0], self.start[1], 'o', ms=6, c='k')

        # Plot goal position
        plt.plot(self.end[0], self.end[1], '*', ms=8, c='k')

        for i in range(nPts):
            plt.plot(best_sol[i], best_sol[i+nPts], '.', ms = 9, c='k')

        ax = plt.gca()

        for i in range(len(self.obs)):
            data = self.obs[i]
            xc, yc, r = data[:3]
            element = plt.Circle((xc, yc), r, fc='wheat', color = 'r')
            
            ax.add_patch(element)                   
            ax.plot(xc, yc, 'x', ms=4, c='orange')  
        
        ax.set_xlim(-1 , self.limits)
        ax.set_ylim(-1 , self.limits)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])

        
        plt.show()
        

    
start = [0,0]
end = [10,10]
limit = 10
nPts = 5
P = Path(start, end, limit, nPts)  
P.create_map()
P.optimize()


P.plot_path()

    

    