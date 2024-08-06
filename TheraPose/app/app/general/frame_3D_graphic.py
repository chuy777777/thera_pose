import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from general.grid_frame import GridFrame
from general.create_frame import CreateFrame

class Frame3DGraphic(CreateFrame):
    def __init__(self, master, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(1,1), arr=None), **kwargs)
        self.fig=plt.figure()
        self.canvas=FigureCanvasTkAgg(figure=self.fig, master=self)
        self.ax=plt.axes(projection='3d')
        self.ax.set(xlabel="X", ylabel="Y", zlabel="Z")

        self.insert_element(cad_pos="0,0", element=self.canvas.get_tk_widget(), padx=20, pady=20, sticky="")

        self.graphic_configuration(xlim=[-20,20], ylim=[-20,20], zlim=[0,20])

    def graphic_configuration(self, **kwargs):
        self.ax.set(**kwargs)
        self.canvas.draw()

    def clear(self):
        for line in self.ax.lines:
            line.remove()
        for collection in self.ax.collections:
            collection.remove()

    def save_plot(self, path, plot_name):
        self.fig.savefig("{}/{}.png".format(path, plot_name))

    # p1: (3,), p2: (3,)
    def plot_line(self, p1, p2, rgb_color):
        x1,y1,z1=p1
        x2,y2,z2=p2
        self.ax.plot3D([x1,x2], [y1,y2], [z1,z2], color=(rgb_color[0]/255, rgb_color[1]/255, rgb_color[2]/255))

    # ps: (m,3), connections: [[,], [,], ...]
    def plot_lines(self, ps, connections, rgb_color):
        for conn in connections:
            c1,c2=conn
            self.plot_line(p1=ps[c1], p2=ps[c2], rgb_color=rgb_color)

    # p: (3,)
    def plot_point(self, p, rgb_color):
        x,y,z=p
        self.ax.scatter3D(x, y, z, color=(rgb_color[0]/255, rgb_color[1]/255, rgb_color[2]/255), alpha=0.8, marker=".", s=50)

    # ps: (m,3)
    def plot_points(self, ps, rgb_color):
        self.ax.scatter3D(ps[:,0], ps[:,1], ps[:,2], color=(rgb_color[0]/255, rgb_color[1]/255, rgb_color[2]/255), alpha=0.8, marker=".", s=50)

    # alpha, facecolors, edgecolors 
    # xs: (n,), ys: (n,), zs: (n,)
    def plot_plane(self, xs, ys, zs, **kwargs):
        verts=[list(zip(xs,ys,zs))]
        poly=Poly3DCollection(verts, **kwargs)
        self.ax.add_collection3d(poly)

    # t: (3,1) T: (3,3)
    def plot_coordinate_system(self, t, T):
        axis_colors=['r','g','b']
        tx,ty,tz=t.flatten()
        self.ax.scatter3D([tx],[ty],[tz], edgecolor="k", color="k", s=15)
        for i in range(3):
            x,y,z=(t + T[:,[i]]).flatten()
            self.ax.plot3D([tx,x],[ty,y],[tz,z], color=axis_colors[i])

    # ts: [(3,1), ...], Ts: [(3,3), ...] 
    def plot_coordinate_systems(self, ts, Ts):
        m=len(ts)
        for i in range(m):
            t=ts[i]
            T=Ts[i]
            self.plot_coordinate_system(t=t, T=T)