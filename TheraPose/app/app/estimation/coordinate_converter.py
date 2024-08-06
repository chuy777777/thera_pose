import numpy as np

class CoordinateConverter():
    # ps (m,3)
    @staticmethod
    def world_to_graphic(ps):
        Twg=np.array([[1,0,0],[0,-1,0],[0,0,1]])
        vgs=Twg@ps.T
        return vgs.T
