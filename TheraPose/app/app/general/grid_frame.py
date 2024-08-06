import numpy as np

class GridFrame():
    def __init__(self, dim=(1,1), arr=None):
        if dim[0] < 1 or dim[1] < 1: dim=(1,1)
        self.dim=dim
        self.arr=arr   
        self.list_cad_pos=[]
        self.dict={}

        if self.arr is None:
            for i in range(self.dim[0]):
                aux=np.array([["{},{}".format(i, j)] for j in range(self.dim[1])], dtype="<U5").T
                self.arr=aux if self.arr is None else np.concatenate([self.arr, aux], axis=0)

        self.init()

    def init(self):
        h,w=self.dim
        pos=[]
        for i in range(h):
            for j in range(w):
                pos.append((i,j))
                if not self.arr[i,j] in self.list_cad_pos:
                    self.list_cad_pos.append(self.arr[i,j])
                    self.dict[self.arr[i,j]]={"columnspan": 1, "rowspan": 1}

        for p in pos:
            i,j=p
            cad=str(i)+","+str(j)
            if cad in self.list_cad_pos:
                columnspan=1
                rowspan=1
                while j+1<w and self.arr[i,j+1]==self.arr[i,j]:
                    columnspan+=1
                    j+=1
                self.dict[cad]["columnspan"]=columnspan
                i,j=p
                while i+1<h and self.arr[i+1,j]==self.arr[i,j]:
                    rowspan+=1
                    i+=1
                self.dict[cad]["rowspan"]=rowspan
                


