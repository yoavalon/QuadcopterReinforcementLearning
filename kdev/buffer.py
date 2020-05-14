import numpy as np
import configs

class Buffer(object) :

    def __init__(self) :

        self.s = []
        self.a = []
        self.r = []
        self.r_dis = []

    def clear(self) :

        self.s = []
        self.a = []
        self.r = []
        self.r_dis = []

    def append(self ,s,a,r) :

        self.s.append(s)
        self.a.append(a)
        self.r.append((r+8)/8)

    def discount(self, v) :

        for r in self.r[::-1]:
            v_s_ = r + configs.GAMMA * v
            self.r_dis.append(v)
        self.r_dis.reverse()

    def format(self) :
        br = np.array(self.r_dis)[:, np.newaxis]

        return self.s, self.a, br
