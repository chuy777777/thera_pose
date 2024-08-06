from __future__ import annotations

import time
import cv2
import copy
import numpy as np
import customtkinter  as ctk
import tkinter as tk
from PIL import Image, ImageTk
import random
import os

import general.utils as utils
import general.ui_images as ui_images
from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.create_scrollable_frame import CreateScrollableFrame
from general.frame_help import FrameHelp
from general.nn import SequentialModel, StandardNormalization

from estimation.frame_estimation_configurations import FrameEstimationConfigurations
from estimation.frame_estimation import FrameEstimation

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal 

"""
    class CTkCanvas(tkinter.Canvas):
        ...

    - Sistema de coordenadas
         --> X
        |
        Y
    - Todas las creaciones devuelven un entero (un identificador)
    - https://python-course.eu/tkinter/events-and-binds-in-tkinter.php 
    - https://tkinter-docs.readthedocs.io/en/latest/widgets/canvas.html
"""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class GameFunctions():
    @staticmethod
    def point_in_box_vector(vec1, vec2, point):
        if vec1[0] >= vec2[0] and vec1[1] >= vec2[1]: return (point[0] >= vec2[0] and point[0] <= vec1[0]) and (point[1] >= vec2[1] and point[1] <= vec1[1])
        if vec1[0] >= vec2[0] and vec1[1] <= vec2[1]: return (point[0] >= vec2[0] and point[0] <= vec1[0]) and (point[1] >= vec1[1] and point[1] <= vec2[1])
        if vec1[0] <= vec2[0] and vec1[1] >= vec2[1]: return (point[0] >= vec1[0] and point[0] <= vec2[0]) and (point[1] >= vec2[1] and point[1] <= vec1[1])
        if vec1[0] <= vec2[0] and vec1[1] <= vec2[1]: return (point[0] >= vec1[0] and point[0] <= vec2[0]) and (point[1] >= vec1[1] and point[1] <= vec2[1])

    @staticmethod
    def vectors_intersect(ps, vs, ps_, vs_):
        vectors_intersect_list=[]
        points_list=[]
        # Comparamos v con v_ 
        # Siempre se cumple que v != 0 y v_ != 0
        for i in range(vs.shape[0]):
            p,v=ps[i],vs[i]
            for j in range(vs_.shape[0]):
                p_,v_=ps_[j],vs_[j]
                # for A,b in [(np.concatenate([v[:,None], -v_[:,None]], axis=1),(p_ - p)[:,None]),(np.concatenate([-v[:,None], v_[:,None]], axis=1),(p - p_)[:,None])]:
                for A,b in [(np.concatenate([v[:,None], -v_[:,None]], axis=1),(p_ - p)[:,None])]:
                    try:
                        x=(np.linalg.inv(A)@b).flatten()
                        if (x[0] >= 0 and x[0] <= 1) and (x[1] >= 0 and x[1] <= 1):
                            if sum([(vectors_intersect_list[i] == v_).all() for i in range(len(vectors_intersect_list))]) == 0:
                                vectors_intersect_list.append(v_)
                                points_list.append(p_)
                    except np.linalg.LinAlgError as e:
                        # Los vectores a1 y a2 (vectores columna de A) son linealmente dependientes (por tanto A no es invertible)
                        # Por tanto, significa que v y v_ son paralelos 
                        w=p_ - p 
                        if np.linalg.norm(np.cross(np.concatenate([w,[0]],axis=0), np.concatenate([v,[0]],axis=0))) == 0:   # Significa que v y v_ viven en la misma recta
                            u=p + v
                            u_=p_ + v_
                            # Verificamos en ambos sentidos
                            if GameFunctions.point_in_box_vector(vec1=p, vec2=u, point=p_) or GameFunctions.point_in_box_vector(vec1=p, vec2=u, point=u_) or GameFunctions.point_in_box_vector(vec1=p_, vec2=u_, point=p) or GameFunctions.point_in_box_vector(vec1=p_, vec2=u_, point=u):
                                if sum([(vectors_intersect_list[i] == v_).all() for i in range(len(vectors_intersect_list))]) == 0:
                                    vectors_intersect_list.append(v_)
                                    points_list.append(p_)
        return vectors_intersect_list,points_list

    @staticmethod
    def create_slider_container(master, title, message_format, message_size, from_, to, number_of_steps, variable):
        frame_container=CreateFrame(master=master, grid_frame=GridFrame(dim=(3,1) if message_format is not None else (2,1), arr=None), fg_color="lightpink")
        frame_container.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=frame_container, text=title, size=12, weight="bold"), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="1,0", element=ctk.CTkSlider(master=frame_container, from_=from_, to=to, number_of_steps=number_of_steps, variable=variable), padx=5, pady=5, sticky="ew")
        if message_format is not None:
            frame_container.insert_element(cad_pos="2,0", element=FrameHelp.create_label(master=frame_container, text=message_format.format(int(variable.get())), size=message_size, weight="bold", fg_color="transparent"), padx=5, pady=5, sticky="ew")
            variable.trace_add("write", lambda var, index, mode: frame_container.get_element(cad_pos="2,0").configure(text=message_format.format(int(variable.get()))))
        return frame_container
    
    @staticmethod
    def create_message_container(master, title, message_size, variable):
        frame_container=CreateFrame(master=master, grid_frame=GridFrame(dim=(2,1), arr=None), fg_color="lightpink")
        frame_container.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=frame_container, text=title, size=12, weight="bold"), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="1,0", element=FrameHelp.create_label(master=frame_container, textvariable=variable, text="", size=message_size, weight="bold", fg_color="transparent"), padx=5, pady=5, sticky="ew")
        return frame_container

    @staticmethod
    def create_progress_container(master, title, variable):
        frame_container=CreateFrame(master=master, grid_frame=GridFrame(dim=(2,1), arr=None), fg_color="lightpink")
        frame_container.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=frame_container, text=title, size=12, weight="bold"), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="1,0", element=ctk.CTkProgressBar(master=frame_container, variable=variable, width=100, height=25, orientation="horizontal"), padx=5, pady=5, sticky="ew")
        return frame_container

class GameInformation():
    def __init__(self, game, game_name, game_description):
        self.game=game
        self.game_name=game_name
        self.game_description=game_description

class FrameGameInformation(CreateFrame):
    def __init__(self, master, game_information: GameInformation, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(2,2), arr=np.array([["0,0","0,1"],["0,0","1,1"]])), **kwargs)
        self.game_information=game_information

        self.insert_element(cad_pos="0,0", element=ctk.CTkLabel(master=self, text="", image=ui_images.get_game_image(name="background_game", width=100, height=100)), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="0,1", element=FrameHelp.create_label(master=self, text=self.game_information.game_name, size=18, weight="bold"), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,1", element=ctk.CTkTextbox(master=self, fg_color="#F2F2F2", wrap="word", width=200, height=100), padx=5, pady=5, sticky="")
        
        self.get_element(cad_pos="1,1").insert("0.0", self.game_information.game_description)
        self.get_element(cad_pos="1,1").configure(state=ctk.DISABLED)

class FrameGameCountdown(CreateFrame):
    def __init__(self, master, time_s, finished_countdown_callback, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(2,1), arr=None), **kwargs)
        self.time_s=time_s
        self.finished_countdown_callback=finished_countdown_callback
        self.var_title=ctk.StringVar(value="El juego comienza en:")
        self.var_time=ctk.StringVar(value="")

        self.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self, text="", size=10, weight="bold", textvariable=self.var_title), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,0", element=FrameHelp.create_label(master=self, text="", size=50, weight="bold", textvariable=self.var_time, width=50, height=50), padx=5, pady=5, sticky="ew")
    
    def start_countdown(self):
        self.countdown(t1=self.time_s, t2=self.time_s)

    def countdown(self, t1, t2):
        t1-=0.1
        if t2 - t1 <= 1:
            self.var_time.set(value=str(int(t2)))
        else:
            t2-=1
            t1=t2
        if t2 >= 1:
            self.after(100, lambda: self.countdown(t1=t1, t2=t2))
        else:
            self.var_time.set(value="")
            self.finished_countdown_callback()

class GameCanvasTemplate(CreateFrame):
    def __init__(self, master, configurations_dict, canvas_width, canvas_height, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(2,2), arr=np.array([["0,0","0,1"],["1,0","0,1"]])), **kwargs)
        self.configurations_dict=configurations_dict
        self.canvas_width=canvas_width
        self.canvas_height=canvas_height
        self.game_loop_ms=10
        self.players_list=[]
        self.finished_game=True
        self.var_game_state=ctk.StringVar(value="")
        self.var_game_state.trace_add("write", self.game_state_callback)
        
        self.frame_visualization=CreateFrame(master=self, grid_frame=GridFrame(dim=(2,1), arr=None))
        self.frame_visualization.insert_element(cad_pos="0,0", element=ctk.CTkTextbox(master=self.frame_visualization, fg_color="#F2F2F2", wrap="word", width=120, height=120), padx=5, pady=5, sticky="ew")

        self.my_canvas=ctk.CTkCanvas(master=self, width=self.canvas_width, height=self.canvas_height, background="lightcoral")

        self.insert_element(cad_pos="1,0", element=self.frame_visualization, padx=5, pady=5, sticky="ns")
        self.insert_element(cad_pos="0,1", element=self.my_canvas, padx=5, pady=5, sticky="")
        
        self.set_input_message(message=None)
        self.game_loop()

    # Sobreescribir este metodo
    def init_game(self):
        pass

    # Sobreescribir este metodo
    def update_game(self):
        pass

    # Sobreescribir este metodo
    def estimation_data(self, filtered_estimations_results):
        pass

    def input(self, estimations_results):
        if self.configurations_dict.keys() == estimations_results.keys():
            filtered_estimations_results={}
            for algorithm_name in self.configurations_dict.keys():
                filtered_estimations_results[algorithm_name]={}
                if 'left' in self.configurations_dict[algorithm_name]:
                    filtered_estimations_results[algorithm_name]['left']={'vws': None}
                    if algorithm_name in estimations_results.keys() and estimations_results[algorithm_name]['left']['vws'] is not None:
                        filtered_estimations_results[algorithm_name]['left']['vws']=estimations_results[algorithm_name]['left']['vws'][self.configurations_dict[algorithm_name]['left']['enabled_points'],:]
                if 'right' in self.configurations_dict[algorithm_name]:
                    filtered_estimations_results[algorithm_name]['right']={'vws': None}
                    if algorithm_name in estimations_results.keys() and estimations_results[algorithm_name]['right']['vws'] is not None:
                        filtered_estimations_results[algorithm_name]['right']['vws']=estimations_results[algorithm_name]['right']['vws'][self.configurations_dict[algorithm_name]['right']['enabled_points'],:]
                if "enabled_points" in self.configurations_dict[algorithm_name]:
                    filtered_estimations_results[algorithm_name]={'vws': None}
                    if algorithm_name in estimations_results.keys() and estimations_results[algorithm_name]['vws'] is not None:
                        filtered_estimations_results[algorithm_name]['vws']=estimations_results[algorithm_name]['vws'][self.configurations_dict[algorithm_name]['enabled_points'],:]
        
            self.estimation_data(filtered_estimations_results=filtered_estimations_results)
            self.set_input_message(message="Estimaciones en curso.")

    # Sobreescribir este metodo
    def init_ctk_vars(self):
        pass

    # Sobreescribir este metodo
    def init_game_vars(self, ctk_vars=False):
        pass

    # Sobreescribir este metodo
    def init_play_vars(self):
        pass

    def vars_pause_state(self):
        if self.finished_game:
            self.var_game_state.set(value="restart")

    def vars_restart_state(self):
        self.init_game_vars(ctk_vars=True)

    def vars_play_state(self):
        self.init_play_vars()
    
    def end_game(self):
        # Configuramos variable de finalizacion
        self.finished_game=True
        # Mostramos imagen de finalizacion del juego
        end_game_player=Player(frame_game_canvas=self, name="end_game_player", primitive_shape="Image", size=(200,200), image_name="game_over", initial_position=np.array([self.canvas_width/2, self.canvas_height/2]), initial_velocity=np.zeros(2), initial_rotation_angle=0.0)
        self.players_list.append(end_game_player)

    def game_state_callback(self, var, index, mode):
        if self.var_game_state.get() == "pause":
            self.vars_pause_state()
        elif self.var_game_state.get() == "restart":
            self.vars_restart_state()
        elif self.var_game_state.get() == "play":
            self.vars_play_state()

    def set_input_message(self, message):
        self.frame_visualization.get_element(cad_pos="0,0").configure(state=ctk.NORMAL)
        self.frame_visualization.get_element(cad_pos="0,0").delete("0.0", "end") 
        self.frame_visualization.get_element(cad_pos="0,0").insert("0.0", "No se estan recibiendo datos" if message is None else message)
        self.frame_visualization.get_element(cad_pos="0,0").configure(state=ctk.DISABLED)

    def delete_all(self):
        self.my_canvas.delete('all')
    
    def init(self):
        self.delete_all()
        self.init_game()

    def game_loop(self):
        # game_state: ["play", "pause", "restart", ""]
        if self.var_game_state.get() == "play":
            self.update_game()
        elif self.var_game_state.get() == "restart":
            self.init()
            self.var_game_state.set(value="")
        self.after(self.game_loop_ms, self.game_loop)

class GameCoordinateConverter():
    def __init__(self, canvas_width, canvas_height):
        self.canvas_width=canvas_width
        self.canvas_height=canvas_height
        self.H21=np.array([[1,0,0],[0,-1,self.canvas_height],[0,0,1]])
        self.H21_inv=np.linalg.inv(self.H21)
    
    # Devuelve (m,2)
    def convert_my_system_to_canvas(self, ps, rotation_angle):
        self.H32=self.get_H32(rotation_angle=rotation_angle)
        self.H31=self.H21@self.H32
        ps_=self.H31@np.concatenate([ps, np.ones((ps.shape[0],1))],axis=1).T
        ps_=ps_[0:2,:].T
        return ps_
    
    # Devuelve (m,2)
    def convert_canvas_to_my_system(self, ps):
        ps_=self.H21_inv@np.concatenate([ps, np.ones((ps.shape[0],1))],axis=1).T
        ps_=ps_[0:2,:].T
        return ps_
    
    # Devuelve (3,3)
    def get_H32(self, rotation_angle):
        H32=np.array([[np.cos(rotation_angle),-np.sin(rotation_angle),0],[np.sin(rotation_angle),np.cos(rotation_angle),0],[0,0,1]])
        return H32

class Player(GameCoordinateConverter):
    def __init__(self, frame_game_canvas, name, primitive_shape: Literal["Rectangle", "Circle", "Image", "Window"], size, image_name, initial_position, initial_velocity, initial_rotation_angle, **kwargs):
        GameCoordinateConverter.__init__(self, canvas_width=frame_game_canvas.canvas_width, canvas_height=frame_game_canvas.canvas_height)
        self.frame_game_canvas=frame_game_canvas
        self.name=name
        self.primitive_shape=primitive_shape
        self.size=size
        self.image_name=image_name
        self.position=initial_position
        self.velocity=initial_velocity
        self.rotation_angle=initial_rotation_angle
        self.image=None

        self.create_primitive_shape(**kwargs)

    def create_primitive_shape(self, **kwargs):
        if self.primitive_shape == "Rectangle":
            self._id=self.frame_game_canvas.my_canvas.create_polygon(self.get_primitive_shape_coords(), **kwargs)
        elif self.primitive_shape == "Circle":
            self._id=self.frame_game_canvas.my_canvas.create_oval(self.get_primitive_shape_coords(), **kwargs)
        elif self.primitive_shape == "Image":
            image=ui_images.get_image_to_canvas(name=self.image_name, width=self.size[0], height=self.size[1])
            image=ImageTk.PhotoImage(image=image)
            # Tenemos que guardar la imagen en la clase porque si no no funciona
            self.image=image
            self._id=self.frame_game_canvas.my_canvas.create_image(self.get_primitive_shape_coords(), image=self.image, **kwargs)
        elif self.primitive_shape == "Window":
            self._id=self.frame_game_canvas.my_canvas.create_window(self.get_primitive_shape_coords(), **kwargs)

    def get_primitive_shape_coords(self):
        # Aqui se pasa del s.d.c. que elegimos (por comodidad) al s.d.c. original
        w,h=self.size
        if self.primitive_shape == "Rectangle":
            v1=self.position + np.array([-w/2,h/2])
            v2=self.position + np.array([-w/2,-h/2])
            v3=self.position + np.array([w/2,-h/2])
            v4=self.position + np.array([w/2,h/2])
            ps=np.concatenate([v1[None,:],v2[None,:],v3[None,:],v4[None,:]],axis=0)
            ps=self.convert_my_system_to_canvas(ps=ps, rotation_angle=self.rotation_angle) 
            return tuple(ps.flatten(order='C'))
        elif self.primitive_shape == "Circle":
            p1=self.position + np.array([-w/2, h/2])
            p2=self.position + np.array([w/2, -h/2])
            ps=np.concatenate([p1[None,:],p2[None,:]],axis=0)
            ps=self.convert_my_system_to_canvas(ps=ps, rotation_angle=0.0) 
            return tuple(ps.flatten(order='C'))
        elif self.primitive_shape == "Image" or self.primitive_shape == "Window":
            ps=self.position[None,:]
            ps=self.convert_my_system_to_canvas(ps=ps, rotation_angle=0.0) 
            return tuple(ps.flatten(order='C'))

    # Puntos: (izquierda,arriba), (izquierda,abajo), (derecha,abajo), (derecha,arriba)
    def get_vertex_points(self):
        if self.primitive_shape == "Rectangle":
            x1,y1,x2,y2,x3,y3,x4,y4=tk.Canvas.coords(self.frame_game_canvas.my_canvas, self._id)
            ps=np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
            ps=self.convert_canvas_to_my_system(ps=ps)
            return tuple(ps.flatten(order='C'))
        elif self.primitive_shape == "Circle":
            w,h=self.size
            x1,y1,x3,y3=tk.Canvas.coords(self.frame_game_canvas.my_canvas, self._id)
            ps=np.array([[x1,y1],[x1,y1 + h],[x3,y3],[x3,y3 - h]])
            ps=self.convert_canvas_to_my_system(ps=ps)
            return tuple(ps.flatten(order='C'))
        elif self.primitive_shape == "Image" or self.primitive_shape == "Window":
            w,h=self.size
            x1,y1=tk.Canvas.coords(self.frame_game_canvas.my_canvas, self._id)
            ps=np.array([[x1 - w/2,y1 - h/2],[x1 - w/2,y1 + h/2],[x1 + w/2,y1 + h/2],[x1 + w/2,y1 - h/2]])
            ps=self.convert_canvas_to_my_system(ps=ps)
            return tuple(ps.flatten(order='C'))

    # Orden: (primero,segundo), ..., (ultimo,primero)
    # Devuelve: ps (m,2), vs (m,2)
    def get_limit_vectors(self):
        ps=None
        vs=None
        if self.primitive_shape == "Rectangle" or self.primitive_shape == "Image" or self.primitive_shape == "Window":
            ps=np.reshape(self.get_vertex_points(),(4,2),order='C')
            vs=np.concatenate([ps[[i+1 if i < ps.shape[0]-1 else 0],:]-ps[[i],:] for i in range(ps.shape[0])], axis=0)
        elif self.primitive_shape == "Circle":
            m=8                     # Numero de lados
            r=self.size[0]/2        # Radio del circulo
            ps=np.zeros((m,2))
            for k in range(m):
                pk=self.position + np.array([r*np.cos((2*np.pi/m)*(k-1)), r*np.sin((2*np.pi/m)*(k-1))]) 
                ps[k,:]=pk
            vs=np.concatenate([ps[[i+1 if i < ps.shape[0]-1 else 0],:]-ps[[i],:] for i in range(ps.shape[0])], axis=0)
        return ps,vs
        
    def inside_player(self, other_player: Player):
        # Todos los puntos clave del jugador deben estar dentro del otro jugador (TODOS)
        ps,vs=self.get_limit_vectors()
        ps_,vs_=other_player.get_limit_vectors()
        # Definimos vectores que van desde la posicion del jugador hasta un punto del otro jugador (puntos limites)
        vs_=np.concatenate([(ps_[i] - self.position)[None,:] for i in range(ps_.shape[0])], axis=0)
        # Definimos puntos de donde parten dichos vectores (lo cual es el mismo punto para todos)
        ps_=np.concatenate([self.position[None,:] for i in range(vs_.shape[0])], axis=0)
        vectors_intersect_list,points_list=GameFunctions.vectors_intersect(ps=ps,vs=vs, ps_=ps_, vs_=vs_)
        return len(vectors_intersect_list) == 0

    # Devuelve: [(other_player.name, vectors_intersect_list, vectors_normal_list), ...]
    def get_collided_objects(self):
        collided_objects=[]
        players_list=self.frame_game_canvas.players_list
        # Verificar si ha colisionado con las paredes
        canvas_width,canvas_height=self.frame_game_canvas.canvas_width,self.frame_game_canvas.canvas_height
        x,y=self.position
        w,h=self.size
        if x - w/2 <= 0: collided_objects.append(('left_wall', [np.array([0,1])], [np.array([1,0])]))
        if x + w/2 >= canvas_width: collided_objects.append(('right_wall', [np.array([0,1])], [np.array([-1,0])]))
        if y - h/2 <= 0: collided_objects.append(('bottom_wall', [np.array([1,0])], [np.array([0,1])]))
        if y + h/2 >= canvas_height: collided_objects.append(('top_wall', [np.array([1,0])], [np.array([0,-1])]))
        # Verificar si ha colisionado con algun jugador
        x1,y1,x2,y2,x3,y3,x4,y4=self.get_vertex_points()
        r=np.linalg.norm(np.array([x1,y1]) - self.position)
        for other_player in players_list:
            other_player: Player
            if other_player.name != self.name:
                _x1,_y1,_x2,_y2,_x3,_y3,_x4,_y4=other_player.get_vertex_points()
                r_=np.linalg.norm(np.array([_x1,_y1]) - other_player.position)
                d=np.linalg.norm(other_player.position - self.position)
                if d <= (r + r_):
                    # Zona factible (es posible que esten colisionando)
                    # Verificar si el jugador se encuentra dentro del otro jugador y viceversa (aqui ningun limite colisiona)
                    if self.inside_player(other_player=other_player) or other_player.inside_player(other_player=self):
                        collided_objects.append((other_player.name, None, None))
                    else:
                        # Verificar si los limites del jugador colisionan con los limites del otro jugador
                        ps,vs=self.get_limit_vectors()
                        ps_,vs_=other_player.get_limit_vectors()
                        vectors_intersect_list,points_list=GameFunctions.vectors_intersect(ps=ps,vs=vs, ps_=ps_, vs_=vs_)
                        vectors_normal_list=[(points_list[i] + (1/2)*vectors_intersect_list[i]) - other_player.position for i in range(len(vectors_intersect_list))]
                        if len(vectors_intersect_list) > 0:
                            collided_objects.append((other_player.name, vectors_intersect_list, vectors_normal_list))
        return collided_objects

    def update_image_size(self, size):
        if self.image is not None:
            self.size=size
            image=ui_images.get_image_to_canvas(name=self.image_name, width=self.size[0], height=self.size[1])
            image=ImageTk.PhotoImage(image=image)
            self.image=image
            self.frame_game_canvas.my_canvas.itemconfigure(self._id, image=self.image)

    def update_player(self):
        tk.Canvas.coords(self.frame_game_canvas.my_canvas, self._id, self.get_primitive_shape_coords())
        
class FrameGameTemplate(CreateFrame):
    def __init__(self, master, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(3,3), arr=np.array([["0,0","0,1","0,2"],["0,0","0,1","1,2"],["2,0","2,0","2,0"]])), **kwargs)
        self.var_text=ctk.StringVar(value="")
        self.is_countdown=False

    def set_game(self, game_information: GameInformation, frame_game_canvas, configurations_dict):
        self.destroy_all()
        
        canvas_width=650
        canvas_height=450

        frame_buttons=CreateFrame(master=self, grid_frame=GridFrame(dim=(1,3), arr=None))
        frame_buttons.insert_element(cad_pos="0,0", element=FrameHelp.create_button(master=frame_buttons, command=lambda: self.set_game_state(game_state="play"), text="", img_name="play", img_width=50, img_height=50, width=50, height=50, fg_color="transparent"), padx=5, pady=5, sticky="")
        frame_buttons.insert_element(cad_pos="0,1", element=FrameHelp.create_button(master=frame_buttons, command=lambda: self.set_game_state(game_state="pause"), text="", img_name="pause", img_width=50, img_height=50, width=50, height=50, fg_color="transparent"), padx=5, pady=5, sticky="")
        frame_buttons.insert_element(cad_pos="0,2", element=FrameHelp.create_button(master=frame_buttons, command=lambda: self.set_game_state(game_state="restart"), text="", img_name="restart", img_width=50, img_height=50, width=50, height=50, fg_color="transparent"), padx=5, pady=5, sticky="")
        
        self.insert_element(cad_pos="0,0", element=FrameGameInformation(master=self, game_information=game_information), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="0,1", element=FrameGameCountdown(master=self, time_s=3, finished_countdown_callback=lambda: self.set_game_state(game_state="finished_countdown")), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="0,2", element=FrameHelp.create_label(master=self, text="", size=14, weight="bold", textvariable=self.var_text, fg_color="coral"), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,2", element=frame_buttons, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="2,0", element=frame_game_canvas(master=self, configurations_dict=configurations_dict, canvas_width=canvas_width, canvas_height=canvas_height), padx=5, pady=5, sticky="nsew")
        
        self.set_game_state(game_state="restart")

    def estimations_results_callback(self, estimations_results):
        if self.element_exists(cad_pos="2,0"):
            self.get_element(cad_pos="2,0").input(estimations_results=estimations_results)

    def set_game_state(self, game_state: Literal["play", "pause", "restart", "finished_countdown"]):
        frame_game_countdown=self.get_element(cad_pos="0,1")
        frame_game_canvas=self.get_element(cad_pos="2,0")
        if game_state == "play":
            if not self.is_countdown and (frame_game_canvas.var_game_state.get() == "pause" or frame_game_canvas.var_game_state.get() == ""):
                frame_game_countdown.start_countdown()
                self.var_text.set(value="El juego esta apunto de comenzar...")
                self.is_countdown=True
        elif game_state == "pause":
            if not self.is_countdown and (frame_game_canvas.var_game_state.get() == "play"):
                frame_game_canvas.var_game_state.set(value=game_state)
                self.var_text.set(value="El juego se encuentra pausado")
        elif game_state == "restart":
            if not self.is_countdown:
                frame_game_canvas.var_game_state.set(value=game_state)
                self.var_text.set(value="Presione el boton de 'play' para comenzar")
        elif game_state == "finished_countdown":
            frame_game_canvas.var_game_state.set(value="play")
            self.var_text.set(value="El juego ha comenzado")
            self.is_countdown=False

class FrameGame(CreateScrollableFrame):
    def __init__(self, master, **kwargs):
        CreateScrollableFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(3,1), arr=None), **kwargs)
        self.normal_button_color=utils.rgb_to_hex(color=(162,255,27))
        self.selected_button_color=utils.rgb_to_hex(color=(92,153,3))

        self.frame_game_buttons=CreateFrame(master=self, grid_frame=GridFrame(dim=(2,5), arr=None))
        self.cad_pos_list=["0,0","0,1","0,2","0,3","0,4"]
        self.game_information_list=[
            GameInformation(game=0, game_name="Juego 1", game_description="Este juego consiste en mover los brazos (rectangulos azules) para hacerlos coincidir con los rectangulos rojos."),
            GameInformation(game=1, game_name="Juego 2", game_description="Este juego consiste en mover las manos (las canastas) para atrapar las manzanas del color correspondiente."),
            GameInformation(game=2, game_name="Juego 3", game_description="Este juego consiste en mover un cohete por medio del recorrido de una sentadilla."),
            GameInformation(game=3, game_name="Juego 4", game_description="Este juego consiste en dibujar los contornos de las imagenes utilizando los brazos."),
            GameInformation(game=4, game_name="Juego 5", game_description="Este juego consiste en levantar los dedos que se indican en la imagen.")
        ]
        self.frame_game_canvas_list=[
            FrameGameCanvas0,
            FrameGameCanvas1,
            FrameGameCanvas2,
            FrameGameCanvas3,
            FrameGameCanvas4
        ]
        for i in range(len(self.cad_pos_list)):
            self.frame_game_buttons.insert_element(cad_pos=self.cad_pos_list[i], element=ctk.CTkButton(master=self.frame_game_buttons, command=lambda i=i: self.button_game(game=i), width=50, height=100, text=self.game_information_list[i].game_name, compound=ctk.TOP, image=ui_images.get_game_image(name="game_{}".format(i), width=50, height=50), text_color="black", fg_color=self.normal_button_color, hover_color="gray60"), padx=5, pady=5, sticky="ew")

        self.frame_game_template=FrameGameTemplate(master=self, height=600, fg_color="lightpink")

        self.frame_estimation=FrameEstimation(master=self, estimations_results_callback=self.estimations_results_callback, verification_thread_estimation_calculation=self.verification_thread_estimation_calculation, height=1900, name="FrameEstimation")
        self.frame_estimation.frame_estimation_configurations.disable_configurations()

        self.insert_element(cad_pos="0,0", element=self.frame_game_buttons, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,0", element=self.frame_game_template, padx=5, pady=5, sticky="nsew")
        self.insert_element(cad_pos="2,0", element=self.frame_estimation, padx=5, pady=5, sticky="nsew")

    def button_game(self, game):
        for i in range(len(self.cad_pos_list)):
            self.frame_game_buttons.get_element(cad_pos=self.cad_pos_list[i]).configure(text_color="black", fg_color=self.normal_button_color)
        self.frame_game_buttons.get_element(cad_pos=self.cad_pos_list[game]).configure(text_color="white", fg_color=self.selected_button_color)
        self.set_configurations_dict_from_game(game=game)

    def set_configurations_dict_from_game(self, game):
        configurations_dict={}

        if game == 0:
            configurations_dict={"mediapipe_pose_estimation": {"enabled_points": np.arange(0,FrameEstimationConfigurations.algorithm_information_list[0].number_of_points).tolist()}}
        elif game == 1:
            configurations_dict={"mediapipe_pose_estimation": {"enabled_points": np.arange(0,FrameEstimationConfigurations.algorithm_information_list[0].number_of_points).tolist()}}
        elif game == 2:
            configurations_dict={"mediapipe_pose_estimation": {"enabled_points": np.arange(0,FrameEstimationConfigurations.algorithm_information_list[0].number_of_points).tolist()}}
        elif game == 3:
            configurations_dict={"mediapipe_pose_estimation": {"enabled_points": np.arange(0,FrameEstimationConfigurations.algorithm_information_list[0].number_of_points).tolist()}}
        elif game == 4:
            configurations_dict={"mediapipe_hand_estimation": {'left': {"enabled_points": np.arange(0,FrameEstimationConfigurations.algorithm_information_list[1].number_of_points).tolist()}, 'right': {"enabled_points": np.arange(0,FrameEstimationConfigurations.algorithm_information_list[1].number_of_points).tolist()}}}

        self.frame_estimation.frame_estimation_configurations.set_configurations_dict(configurations_dict=configurations_dict)
        self.frame_estimation.button_cancel_estimation()
        self.frame_game_template.set_game(game_information=self.game_information_list[game], frame_game_canvas=self.frame_game_canvas_list[game], configurations_dict=configurations_dict)
        
    def estimations_results_callback(self, estimations_dict):
        self.frame_game_template.estimations_results_callback(estimations_results=estimations_dict['algorithms'])

    def verification_thread_estimation_calculation(self, is_alive):
        if self.frame_game_template.element_exists(cad_pos="2,0") and not is_alive:
            self.frame_game_template.get_element(cad_pos="2,0").set_input_message(message=None)
       
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class FrameGameCanvas0(GameCanvasTemplate):
    def __init__(self, master, configurations_dict, canvas_width, canvas_height, **kwargs):
        GameCanvasTemplate.__init__(self, master=master, configurations_dict=configurations_dict, canvas_width=canvas_width, canvas_height=canvas_height, **kwargs)
        self.init_ctk_vars()
        self.init_game_vars(ctk_vars=False)

        frame_container=CreateFrame(master=self, grid_frame=GridFrame(dim=(3,1), arr=None), fg_color="lightpink")
        frame_container.insert_element(cad_pos="0,0", element=GameFunctions.create_slider_container(master=frame_container, title="Numero de repeticiones", message_format="{} repeticion(es)", message_size=12, from_=1, to=50, number_of_steps=50-1, variable=self.var_number_repetitions), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="1,0", element=GameFunctions.create_message_container(master=frame_container, title="Nivel", message_size=30, variable=self.var_level), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="2,0", element=GameFunctions.create_progress_container(master=frame_container, title="Progreso del nivel actual", variable=self.var_progress), padx=5, pady=5, sticky="ew")

        self.insert_element(cad_pos="0,0", element=frame_container, padx=5, pady=5, sticky="ew")

        self.init_game()

    # Se sobreescribe este metodo
    def init_ctk_vars(self):
        self.var_number_repetitions=ctk.DoubleVar(value=5)
        self.var_level=ctk.StringVar(value="")
        self.var_progress=ctk.DoubleVar(value=0.0)

    # Se sobreescribe este metodo
    def init_game_vars(self, ctk_vars=False):
        if ctk_vars:
            self.var_number_repetitions.set(value=5)
            self.var_level.set(value="")
            self.var_progress.set(value=0.0)
        self.finished_game=True
        self.level_count=0
        self.number_repetitions=0

    # Se sobreescribe este metodo
    def init_play_vars(self):
        if self.finished_game:
            self.finished_game=False
            self.number_repetitions=int(self.var_number_repetitions.get())
            self.level_count=0
            self.var_level.set(value="{} de {}".format(self.level_count + 1, self.number_repetitions))
            self.random_positions()

    # Se sobreescribe este metodo
    def init_game(self):
        # Declaramos objetos
        player_width=30
        player_height=100
        padding=15

        self.left_random_player=Player(frame_game_canvas=self, name="left_random_player", primitive_shape="Rectangle", size=(player_width,player_height), image_name="", initial_position=np.array([padding + player_width/2,self.canvas_height/2]), initial_velocity=np.zeros(2), initial_rotation_angle=0.0, fill="red")
        self.right_random_player=Player(frame_game_canvas=self, name="right_random_player", primitive_shape="Rectangle", size=(player_width,player_height), image_name="", initial_position=np.array([self.canvas_width - (padding + player_width/2),self.canvas_height/2]), initial_velocity=np.zeros(2), initial_rotation_angle=0.0, fill="red")
        self.left_player=Player(frame_game_canvas=self, name="left_player", primitive_shape="Rectangle", size=(player_width,player_height), image_name="", initial_position=np.array([3*padding + (3/2)*player_width,self.canvas_height/2]), initial_velocity=np.zeros(2), initial_rotation_angle=0.0, fill="blue")
        self.right_player=Player(frame_game_canvas=self, name="right_player", primitive_shape="Rectangle", size=(player_width,player_height), image_name="", initial_position=np.array([self.canvas_width - (3*padding + (3/2)*player_width),self.canvas_height/2]), initial_velocity=np.zeros(2), initial_rotation_angle=0.0, fill="blue")
        
        x1,y1,x2,y2,x3,y3,x4,y4=self.left_player.get_vertex_points()
        self.left_player_top_line=Player(frame_game_canvas=self, name="left_player_top_line", primitive_shape="Rectangle", size=(x4,4), image_name="", initial_position=np.array([x4/2,y4]), initial_velocity=np.zeros(2), initial_rotation_angle=0.0, fill="black")
        self.left_player_bottom_line=Player(frame_game_canvas=self, name="left_player_bottom_line", primitive_shape="Rectangle", size=(x3,4), image_name="", initial_position=np.array([x3/2,y3]), initial_velocity=np.zeros(2), initial_rotation_angle=0.0, fill="black")
        x1,y1,x2,y2,x3,y3,x4,y4=self.right_player.get_vertex_points()
        self.right_player_top_line=Player(frame_game_canvas=self, name="right_player_top_line", primitive_shape="Rectangle", size=(self.canvas_width - x1,4), image_name="", initial_position=np.array([x1 + (self.canvas_width - x1)/2,y1]), initial_velocity=np.zeros(2), initial_rotation_angle=0.0, fill="black")
        self.right_player_bottom_line=Player(frame_game_canvas=self, name="right_player_bottom_line", primitive_shape="Rectangle", size=(self.canvas_width - x2,4), image_name="", initial_position=np.array([x2 + (self.canvas_width - x2)/2,y2]), initial_velocity=np.zeros(2), initial_rotation_angle=0.0, fill="black")
        
        self.pose_player=Player(frame_game_canvas=self, name="pose_player", primitive_shape="Image", size=(200,200), image_name="pose", initial_position=np.array([self.canvas_width/2, self.canvas_height/2]), initial_velocity=np.zeros(2), initial_rotation_angle=0.0)

        # Agregamos todos los objetos jugador en una lista
        self.players_list=[self.left_random_player, self.right_random_player, self.left_player, self.right_player]

    # Se sobreescribe este metodo
    def update_game(self):
        if not self.finished_game:
            # Actualizamos el juego       
            x1_l,y1_l,x2_l,y2_l,x3_l,y3_l,x4_l,y4_l=self.left_player.get_vertex_points()
            x1_r,y1_r,x2_r,y2_r,x3_r,y3_r,x4_r,y4_r=self.right_player.get_vertex_points()
            x1_lr,y1_lr,x2_lr,y2_lr,x3_lr,y3_lr,x4_lr,y4_lr=self.left_random_player.get_vertex_points()
            x1_rr,y1_rr,x2_rr,y2_rr,x3_rr,y3_rr,x4_rr,y4_rr=self.right_random_player.get_vertex_points()
            w,h=self.left_player.size
            p_left_error=np.abs(y1_l - y1_lr)/h
            p_right_error=np.abs(y1_r - y1_rr)/h
            self.var_progress.set(value=1 - (p_left_error + p_right_error)/2)
            if p_left_error <= 0.2 and p_right_error <= 0.2:   
                # Es aceptado un 20% de error (en cada una)
                self.next_level()

    # Se sobreescribe este metodo
    def estimation_data(self, filtered_estimations_results):
        # Actualizamos a los jugadores en funcion de la entrada recibida
        vws=filtered_estimations_results['mediapipe_pose_estimation']['vws']

        if vws is not None:
            kw=np.array([0,0,1])
            for elem in ((self.left_player, vws[15] - vws[11]), (self.right_player, vws[16] - vws[12])):
                player=elem[0]
                v=elem[1]
                v_norm=utils.normalize_vector(v)
                # Componente en z de 'p': [-1,1]
                p=(np.dot(v_norm,kw)/1)*kw        
                player_range=self.canvas_height - player.size[1]
                player.position=np.array([player.position[0], self.canvas_height/2 + (player_range/2)*p[2]])
            # Actualizamos a los jugadores
            self.left_player.update_player()
            self.right_player.update_player()
            self.plot_player_lines()

    def next_level(self):
        if not self.finished_game:
            if self.level_count < self.number_repetitions - 1:
                self.level_count+=1
                self.var_level.set(value="{} de {}".format(self.level_count + 1, self.number_repetitions))
                self.random_positions()
            else:
                self.end_game()

    def plot_player_lines(self):
        x1,y1,x2,y2,x3,y3,x4,y4=self.left_player.get_vertex_points()
        self.left_player_top_line.position=np.array([x4/2,y4])
        self.left_player_bottom_line.position=np.array([x3/2,y3])
        x1,y1,x2,y2,x3,y3,x4,y4=self.right_player.get_vertex_points()
        self.right_player_top_line.position=np.array([x1 + (self.canvas_width - x1)/2,y1])
        self.right_player_bottom_line.position=np.array([x2 + (self.canvas_width - x2)/2,y2])
        # Actualizamos 
        self.left_player_top_line.update_player()
        self.left_player_bottom_line.update_player()
        self.right_player_top_line.update_player()
        self.right_player_bottom_line.update_player()

    def random_positions(self):
        w,h=self.left_random_player.size
        self.left_random_player.position=np.array([self.left_random_player.position[0], utils.random_value_in_range(a=h/2, b=self.canvas_height - h/2)])
        self.right_random_player.position=np.array([self.right_random_player.position[0], utils.random_value_in_range(a=h/2, b=self.canvas_height - h/2)])
        self.left_random_player.update_player()
        self.right_random_player.update_player()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class FrameGameCanvas1(GameCanvasTemplate):
    def __init__(self, master, configurations_dict, canvas_width, canvas_height, **kwargs):
        GameCanvasTemplate.__init__(self, master=master, configurations_dict=configurations_dict, canvas_width=canvas_width, canvas_height=canvas_height, **kwargs)
        self.init_ctk_vars()
        self.init_game_vars(ctk_vars=False)

        frame_container=CreateFrame(master=self, grid_frame=GridFrame(dim=(4,1), arr=None), fg_color="lightpink")
        frame_container.insert_element(cad_pos="0,0", element=GameFunctions.create_slider_container(master=frame_container, title="Velocidad de las manzanas", message_format=None, message_size=0, from_=0, to=5, number_of_steps=20, variable=self.var_apple_velocity), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="1,0", element=GameFunctions.create_slider_container(master=frame_container, title="Dificultad", message_format=None, message_size=0, from_=0, to=5, number_of_steps=5, variable=self.var_difficulty), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="2,0", element=GameFunctions.create_slider_container(master=frame_container, title="Puntuacion inicial", message_format="{}", message_size=30, from_=1, to=200, number_of_steps=199, variable=self.var_initial_score), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="3,0", element=GameFunctions.create_message_container(master=frame_container, title="Puntuacion actual", message_size=30, variable=self.var_current_score), padx=5, pady=5, sticky="ew")

        self.insert_element(cad_pos="0,0", element=frame_container, padx=5, pady=5, sticky="ew")

        self.init_game()

    # Se sobreescribe este metodo
    def init_ctk_vars(self):
        self.var_apple_velocity=ctk.DoubleVar(value=1.0)
        self.var_difficulty=ctk.DoubleVar(value=1.0)
        self.var_initial_score=ctk.DoubleVar(value=50)
        self.var_current_score=ctk.StringVar(value="")

    # Se sobreescribe este metodo
    def init_game_vars(self, ctk_vars=False):
        if ctk_vars:
            self.var_apple_velocity.set(value=1.0)
            self.var_difficulty.set(value=1.0)
            self.var_initial_score.set(value=50)
            self.var_current_score.set(value="")
        self.finished_game=True
        self.difficulty=0.0
        self.apple_count=0
        self.left_color=""
        self.right_color=""
        self.colors_list=["blue", "brown", "green", "purple", "red", "yellow"]

    # Se sobreescribe este metodo
    def init_play_vars(self):
        if self.finished_game:
            self.finished_game=False
            self.var_current_score.set(value=int(self.var_initial_score.get()))
            self.difficulty=self.var_difficulty.get()
            self.apple_count=0
            self.players_list.append(self.get_random_apple_player())

    # Se sobreescribe este metodo
    def init_game(self):
        # Declaramos objetos
        self.left_color,self.right_color=random.sample(self.colors_list, 2)
        self.left_basket_player=Player(frame_game_canvas=self, name="left_basket_player_{}".format(self.left_color), primitive_shape="Image", size=(50,50), image_name="{}_basket".format(self.left_color), initial_position=np.array([self.canvas_width*(2/10), self.canvas_height/2]), initial_velocity=np.zeros(2), initial_rotation_angle=0.0)
        self.right_basket_player=Player(frame_game_canvas=self, name="right_basket_player_{}".format(self.right_color), primitive_shape="Image", size=(50,50), image_name="{}_basket".format(self.right_color), initial_position=np.array([self.canvas_width*(8/10), self.canvas_height/2]), initial_velocity=np.zeros(2), initial_rotation_angle=0.0)

        # Agregamos todos los objetos jugador en una lista
        self.players_list=[self.left_basket_player, self.right_basket_player]

    # Se sobreescribe este metodo
    def update_game(self):
        if not self.finished_game:
            # Actualizamos el juego     
            if int(self.var_current_score.get()) > 0:  
                # La dificultad hace que aparezcan menos seguido o mas seguido las manzanas
                # Tenemos 6 dificultadoes (0,1,2,3,4,5)
                # Para que aparezca una manzana o no debe de haber manzanas o todas las manzanas restantes deben estar por debajo de un umbral (el cual depende de la dificultad) 
                apple_players_list=list(filter(lambda elem: elem.name.split("_")[0] == "apple", self.players_list))
                if len(apple_players_list) == 0:
                    self.players_list.append(self.get_random_apple_player())
                else:
                    band=True
                    umbral=0 + (self.canvas_height/7)*(int(self.difficulty))
                    for i in range(len(apple_players_list)):
                        apple_player: Player=apple_players_list[i]
                        if apple_player.position[1] >= umbral:
                            band=False
                            break
                    if band:
                        self.players_list.append(self.get_random_apple_player())
                
                delete_apple_name_list=[]
                for player in self.players_list:
                    player: Player
                    if player.name.split("_")[0] == "apple":
                        collided_objects=player.get_collided_objects()
                        for collided_object_name,vectors_intersect_list,vectors_normal_list in collided_objects:
                            if 'bottom_wall' == collided_object_name:
                                # Si choca con el suelo se resta puntuacion
                                self.var_current_score.set(value=int(self.var_current_score.get()) - 1)
                                delete_apple_name_list.append(player.name)
                            elif collided_object_name not in ["left_wall", "right_wall"] and (collided_object_name.split("_")[0] == "left" or collided_object_name.split("_")[0] == "right"):
                                if collided_object_name.split("_")[3] != player.name.split("_")[2]:
                                    # Si los colores de la canasta y de la manzana no coinciden se resta puntuacion
                                    self.var_current_score.set(value=int(self.var_current_score.get()) - 1)
                                delete_apple_name_list.append(player.name)

                            if int(self.var_current_score.get()) == 0:
                                self.delete_apple_player(delete_apple_name_list=delete_apple_name_list)
                                return

                        if player.name not in delete_apple_name_list:
                            player.position=player.position + (player.velocity*self.var_apple_velocity.get())*1
                            player.update_player()
                            
                self.delete_apple_player(delete_apple_name_list=delete_apple_name_list)
            else:
                self.end_game()

    # Se sobreescribe este metodo
    def estimation_data(self, filtered_estimations_results):
        # Actualizamos a los jugadores en funcion de la entrada recibida
        vws=filtered_estimations_results['mediapipe_pose_estimation']['vws']

        if vws is not None:
            a=((np.linalg.norm(vws[13] - vws[11]) + np.linalg.norm(vws[15] - vws[13])) + (np.linalg.norm(vws[14] - vws[12]) + np.linalg.norm(vws[16] - vws[14])))/2
            b=np.linalg.norm(vws[12] - vws[11])
            if a != 0 and b != 0:
                F=np.array([[self.canvas_width/(2*a + b), 0],[0, self.canvas_height/(2*a)]])
                for player,player_type,u,v,v_ in [(self.left_basket_player,'left',vws[11] - vws[12],vws[23] - vws[11],vws[15] - vws[11]), (self.right_basket_player,'right',vws[12] - vws[11],vws[24] - vws[12],vws[16] - vws[12])]:
                    u1=utils.normalize_vector(u)
                    u2=-utils.normalize_vector(v)
                    A=np.concatenate([u1[:,None], u2[:,None]], axis=1)
                    x=A.T@v_[:,None]
                    c=np.array([self.canvas_width/2, self.canvas_height/2])[:,None]
                    u1=np.zeros(2)
                    if player_type == 'left':
                        T21=np.array([[-1, 0],[0, 1]])
                        u1=(c - (b/2)*F@np.array([1,0])[:,None] + T21@F@x).flatten()
                    elif player_type == 'right':
                        u1=(c + (b/2)*F@np.array([1,0])[:,None] + F@x).flatten()
                    player.position=u1
                # Actualizamos a los jugadores
                self.left_basket_player.update_player()
                self.right_basket_player.update_player()

    def get_random_apple_player(self):
        if self.left_color != "" and self.right_color != "":
            padding=25
            self.apple_count+=1
            index=int(utils.random_value_in_range(a=0, b=1) > 0.5)
            color=[self.left_color, self.right_color][index]
            # A continuacion trabajamos con algunas proporciones
            b=(2/9)*self.canvas_width       # Longitud hombros
            a=(7/4)*b                       # Longitud brazo
            a_random=(0 if index == 0 else self.canvas_width/2 + b/2 - a) + padding
            b_random=(self.canvas_width/2 - b/2 + a if index == 0 else self.canvas_width) - padding
            apple_player=Player(frame_game_canvas=self, name="apple_player_{}_{}".format(color,self.apple_count), primitive_shape="Image", size=(padding*2,padding*2), image_name="{}_apple".format(color), initial_position=np.array([utils.random_value_in_range(a=a_random, b=b_random), self.canvas_height]), initial_velocity=np.array([0,-1]), initial_rotation_angle=0.0)
            return apple_player

    def delete_apple_player(self, delete_apple_name_list):
        if len(delete_apple_name_list) > 0:
            temp_players_list=[]
            for i in range(len(self.players_list)):
                if self.players_list[i].name not in delete_apple_name_list:
                    temp_players_list.append(self.players_list[i])
            self.players_list=temp_players_list

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class FrameGameCanvas2(GameCanvasTemplate):
    def __init__(self, master, configurations_dict, canvas_width, canvas_height, **kwargs):
        GameCanvasTemplate.__init__(self, master=master, configurations_dict=configurations_dict, canvas_width=canvas_width, canvas_height=canvas_height, **kwargs)
        self.init_ctk_vars()
        self.init_game_vars(ctk_vars=False)

        frame_container=CreateFrame(master=self, grid_frame=GridFrame(dim=(5,1), arr=None), fg_color="lightpink")
        frame_container.insert_element(cad_pos="0,0", element=GameFunctions.create_slider_container(master=frame_container, title="Velocidad de los asteroides", message_format=None, message_size=0, from_=0, to=5, number_of_steps=20, variable=self.var_asteroid_velocity), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="1,0", element=GameFunctions.create_slider_container(master=frame_container, title="Dificultad", message_format=None, message_size=0, from_=0, to=5, number_of_steps=5, variable=self.var_difficulty), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="2,0", element=GameFunctions.create_slider_container(master=frame_container, title="Puntuacion inicial", message_format="{}", message_size=30, from_=1, to=200, number_of_steps=199, variable=self.var_initial_score), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="3,0", element=GameFunctions.create_message_container(master=frame_container, title="Puntuacion actual", message_size=30, variable=self.var_current_score), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="4,0", element=GameFunctions.create_slider_container(master=frame_container, title="Angulo minimo de flexion\n(en grados)", message_format="{}°", message_size=30, from_=10, to=170, number_of_steps=160, variable=self.var_angle), padx=5, pady=5, sticky="ew")

        self.insert_element(cad_pos="0,0", element=frame_container, padx=5, pady=5, sticky="ew")

        self.init_game()

    # Se sobreescribe este metodo
    def init_ctk_vars(self):
        self.var_asteroid_velocity=ctk.DoubleVar(value=1.0)
        self.var_difficulty=ctk.DoubleVar(value=1.0)
        self.var_initial_score=ctk.DoubleVar(value=50)
        self.var_current_score=ctk.StringVar(value="")
        self.var_angle=ctk.DoubleVar(value=90)

    # Se sobreescribe este metodo
    def init_game_vars(self, ctk_vars=False):
        if ctk_vars:
            self.var_asteroid_velocity.set(value=1.0)
            self.var_difficulty.set(value=1.0)
            self.var_initial_score.set(value=50)
            self.var_current_score.set(value="")
            self.var_angle.set(value=90)
        self.finished_game=True
        self.asteroid_count=0
        self.difficulty=0.0
        self.asteroid_names=['asteroid_1','asteroid_2','asteroid_3']
        self.x_map=np.zeros(2)

    # Se sobreescribe este metodo
    def init_play_vars(self):
        if self.finished_game:
            # Calculamos unos valores que nos serviran para mapear valores (entre angulos y posicion en pantalla del jugador)
            angle=float(self.var_angle.get())       # En grados
            angle=angle*(np.pi/180)                 # En radianes
            rw,rh=self.rocket_player.size
            A=np.array([[angle,1],[np.pi,1]])
            b=np.array([[rh/2],[self.canvas_height - rh/2]])
            # Para que exista solucion se debe cumplir lo siguiente: angle != pi (siempre se cumple)
            self.x_map=(np.linalg.inv(A)@b).flatten()

            self.finished_game=False
            self.var_current_score.set(value=int(self.var_initial_score.get()))
            self.difficulty=self.var_difficulty.get()
            self.asteroid_count=0
            self.players_list.append(self.get_random_asteroid_player())
    
     # Se sobreescribe este metodo
    def init_game(self):
        # Declaramos objetos
        self.rocket_player=Player(frame_game_canvas=self, name="rocket_player", primitive_shape="Image", size=(80,50), image_name="rocket", initial_position=np.array([self.canvas_width*(1/10), self.canvas_height/2]), initial_velocity=np.zeros(2), initial_rotation_angle=0.0)

        # Agregamos todos los objetos jugador en una lista
        self.players_list=[self.rocket_player]

    # Se sobreescribe este metodo
    def update_game(self):
        if not self.finished_game:
            # Actualizamos el juego     
            if int(self.var_current_score.get()) > 0:  
                # La dificultad hace que aparezcan menos seguido o mas seguido los asteroides
                # Tenemos 6 dificultadoes (0,1,2,3,4,5)
                asteroid_players_list=list(filter(lambda elem: elem.name.split("_")[0] == "asteroid", self.players_list))
                band=True
                umbral=0 + (self.canvas_width/7)*(int(self.difficulty))
                for i in range(len(asteroid_players_list)):
                    asteroid_player: Player=asteroid_players_list[i]
                    if asteroid_player.position[0] >= umbral:
                        band=False
                        break
                if band:
                    self.players_list.append(self.get_random_asteroid_player())
                
                delete_asteroid_name_list=[]
                for player in self.players_list:
                    player: Player
                    if player.name.split("_")[0] == "asteroid":
                        collided_objects=player.get_collided_objects()
                        for collided_object_name,vectors_intersect_list,vectors_normal_list in collided_objects:
                            if 'left_wall' == collided_object_name:
                                # Si choca con la pared izquierda se elimina el asteroide
                                delete_asteroid_name_list.append(player.name)
                            elif collided_object_name == "rocket_player":
                                #  se resta puntuacion
                                self.var_current_score.set(value=int(self.var_current_score.get()) - 1)
                                delete_asteroid_name_list.append(player.name)

                            if int(self.var_current_score.get()) == 0:
                                self.delete_asteroid_player(delete_asteroid_name_list=delete_asteroid_name_list)
                                return

                        if player.name not in delete_asteroid_name_list:
                            player.position=player.position + (player.velocity*self.var_asteroid_velocity.get())*1
                            player.update_player()
                            
                self.delete_asteroid_player(delete_asteroid_name_list=delete_asteroid_name_list)
            else:
                self.end_game()

    # Se sobreescribe este metodo
    def estimation_data(self, filtered_estimations_results):
        # Actualizamos a los jugadores en funcion de la entrada recibida
        vws=filtered_estimations_results['mediapipe_pose_estimation']['vws']

        if vws is not None:
            left_angle=np.arccos(np.dot(vws[23] - vws[25], vws[27] - vws[25])/(np.linalg.norm(vws[23] - vws[25])*np.linalg.norm(vws[27] - vws[25])))        # En radianes
            right_angle=np.arccos(np.dot(vws[24] - vws[26], vws[28] - vws[26])/(np.linalg.norm(vws[24] - vws[26])*np.linalg.norm(vws[28] - vws[26])))       # En radianes
            angle=(left_angle + right_angle)/2                                                                                                              # En radianes
            rw,rh=self.rocket_player.size
            pos=np.dot(np.array([angle, 1]), self.x_map)
            pos=0 + rh/2 if pos < 0 + rh/2 else self.canvas_height - rh/2 if pos > self.canvas_height - rh/2 else pos
            self.rocket_player.position=np.array([self.rocket_player.position[0], pos])
            # Actualizamos a los jugadores
            self.rocket_player.update_player()

    def get_random_asteroid_player(self):
        self.asteroid_count+=1
        asteroid_player=Player(frame_game_canvas=self, name="asteroid_player_{}".format(self.asteroid_count), primitive_shape="Image", size=(60,60), image_name="{}".format(random.sample(self.asteroid_names, 1)[0]), initial_position=np.array([self.canvas_width,utils.random_value_in_range(a=0 + 60/2, b=self.canvas_height - 60/2)]), initial_velocity=np.array([-1,0]), initial_rotation_angle=0.0)
        return asteroid_player

    def delete_asteroid_player(self, delete_asteroid_name_list):
        if len(delete_asteroid_name_list) > 0:
            temp_players_list=[]
            for i in range(len(self.players_list)):
                if self.players_list[i].name not in delete_asteroid_name_list:
                    temp_players_list.append(self.players_list[i])
            self.players_list=temp_players_list

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""   

class Game3Functions():
    @staticmethod
    def get_image_points(image_path, dim):
        # Por imagen, una vez obtenido los puntos se almacenan en la computadora para su posterior uso
        # De esta manera ya tenemos la informacion calculada 
        image_name="{}_{}".format(image_path.split("/")[-1:][0].split(".")[0], str(dim))
        path="{}/calculated_contour_points/{}.npy".format("/".join(image_path.split("/")[0:-1]), image_name)
        if os.path.exists(path):
            temp_ps=np.load(path)
            return temp_ps,None

        # El indicador 'cv2.IMREAD_UNCHANGED' nos permite leer el canal alfa de una imagen
        # (h,w,4)
        img=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
        img=cv2.resize(img,dim,interpolation=cv2.INTER_LINEAR)
        # (h,w)
        img=img[:,:,-1] # Nos quedamos con el ultimo canal
        _,img=cv2.threshold(img,100,255,cv2.THRESH_BINARY)

        h,w=img.shape
        img=np.pad(img, (1), 'constant', constant_values=0) 
        v_lines_img=np.zeros((h,w))
        h_lines_img=np.zeros((h,w))
        v_kernel=np.array([-1,0,1])
        h_kernel=np.array([-1,0,1])
        for i in range(h):
            for j in range(w):
                v_lines_img[i,j]=np.dot(v_kernel,img[1+i,j:j+3])
                h_lines_img[i,j]=np.dot(h_kernel,img[i:i+3,1+j])
                
        v_lines_img=np.abs(v_lines_img)
        h_lines_img=np.abs(h_lines_img)

        new_img=v_lines_img + h_lines_img
        _,new_img=cv2.threshold(new_img,100,255,cv2.THRESH_BINARY)

        # Devuelve las posiciones en la matriz donde es igual al valor que especificamos
        # En el sistema de coordenadas de la matriz
        ps=np.argwhere(new_img == 255)
        n=ps.shape[0] # Numero de puntos (todos los que forman el contorno de la imagen)
        # (n,2)
        # ps.shape 
        # print("Puntos totales (original): {}".format(n))

        """
        Reducimos el numero de puntos 
        """

        d_umbral=20 # En pixeles
        temp_ps=[ps[0][None,:]]
        for i in range(1,n):
            p=ps[i]
            band=True
            for k in range(len(temp_ps)):
                temp_p=temp_ps[k]
                d=np.linalg.norm(p - temp_p)
                if d < d_umbral:
                    band=False
                    break
            if band:
                temp_ps.append(p[None,:])
        temp_ps=np.concatenate(temp_ps, axis=0)
        k=temp_ps.shape[0]

        # print("Puntos totales (nuevo): {}".format(k))

        """
        Mostramos resultados
        """

        filter_img=np.zeros((h,w))
        for i in range(temp_ps.shape[0]):
            x,y=temp_ps[i]
            filter_img[x,y]=255

        # cv2.imshow('filter_img', filter_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Guardamos los puntos ya calculados para su posterior uso
        np.save(path, temp_ps)

        return temp_ps, filter_img
    
    # u2s (n,2)
    @staticmethod
    def convert_matrix_system_to_my_system(u2s, canvas_dim, image_dim):
        w,h=canvas_dim
        wip,hip=image_dim
        M=np.array([[0,1,w/2 - wip/2],[-1,0,h/2 + hip/2],[0,0,1]])
        u2hs=np.concatenate([u2s.T, np.ones((1,u2s.shape[0]))], axis=0)
        u1hs=M@u2hs
        u1s=u1hs[0:2,:].T
        return u1s
    
    # u2s (n,2)
    @staticmethod
    def convert_my_system_to_matrix_system(u1s, canvas_dim, image_dim):
        w,h=canvas_dim
        wip,hip=image_dim
        M=np.array([[0,1,w/2 - wip/2],[-1,0,h/2 + hip/2],[0,0,1]])
        u1hs=np.concatenate([u1s.T, np.ones((1,u1s.shape[0]))], axis=0)
        u2hs=np.linalg.inv(M)@u1hs
        u2s=u2hs[0:2,:].T
        return u2s

class FrameGameCanvas3(GameCanvasTemplate):
    def __init__(self, master, configurations_dict, canvas_width, canvas_height, **kwargs):
        GameCanvasTemplate.__init__(self, master=master, configurations_dict=configurations_dict, canvas_width=canvas_width, canvas_height=canvas_height, **kwargs)
        self.init_ctk_vars()
        self.init_game_vars(ctk_vars=False)

        frame_container=CreateFrame(master=self, grid_frame=GridFrame(dim=(7,1), arr=None), fg_color="lightpink")
        frame_container.insert_element(cad_pos="0,0", element=FrameHelp.create_option_menu(master=frame_container, size=12, weight="bold", values=["Brazo izquierdo", "Brazo derecho"], variable=self.var_arm), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="1,0", element=FrameHelp.create_option_menu(master=frame_container, size=12, weight="bold", values=["De frente", "A la izquierda", "A la derecha"], variable=self.var_plane_position), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="2,0", element=GameFunctions.create_slider_container(master=frame_container, title="Distancia al plano (en mm)", message_format="{}", message_size=30, from_=1000, to=3000, number_of_steps=4, variable=self.var_plane_distance), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="3,0", element=GameFunctions.create_slider_container(master=frame_container, title="Ancho de la imagen (en mm)", message_format="{}", message_size=30, from_=2000, to=3000, number_of_steps=5, variable=self.var_reality_image_width), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="4,0", element=GameFunctions.create_slider_container(master=frame_container, title="Radio del circulo (en pixeles)", message_format="{}", message_size=30, from_=30, to=60, number_of_steps=30, variable=self.var_circle_radius), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="5,0", element=GameFunctions.create_slider_container(master=frame_container, title="Numero de imagenes", message_format="{}", message_size=30, from_=1, to=10, number_of_steps=9, variable=self.var_number_images), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="6,0", element=GameFunctions.create_message_container(master=frame_container, title="Nivel", message_size=20, variable=self.var_level), padx=5, pady=5, sticky="ew")

        self.insert_element(cad_pos="0,0", element=frame_container, padx=5, pady=5, sticky="ew")

        self.init_game()

    # Se sobreescribe este metodo
    def init_ctk_vars(self):
        self.var_arm=ctk.StringVar(value="Brazo izquierdo")
        self.var_plane_position=ctk.StringVar(value="De frente")
        self.var_plane_distance=ctk.DoubleVar(value=1000)
        self.var_reality_image_width=ctk.DoubleVar(value=2000)
        self.var_circle_radius=ctk.DoubleVar(value=30)
        self.var_number_images=ctk.DoubleVar(value=5)
        self.var_level=ctk.StringVar(value="")

    # Se sobreescribe este metodo
    def init_game_vars(self, ctk_vars=False):
        if ctk_vars:
            self.var_arm.set(value="Brazo izquierdo")
            self.var_plane_position.set(value="De frente")
            self.var_plane_distance.set(value=1000)
            self.var_reality_image_width.set(value=2000)
            self.var_circle_radius.set(value=30)
            self.var_number_images.set(value=5)
            self.var_level.set(value="")
        self.finished_game=True
        self.game_image_names=["draw_{}".format(i) for i in range(10)]
        self.game_image_names=random.sample(self.game_image_names, len(self.game_image_names)) 
        self.image_count=0
        self.selected_image_points=[]

    # Se sobreescribe este metodo
    def init_play_vars(self):
        if self.finished_game:
            self.finished_game=False
            self.var_level.set(value="Imagen {} de {}".format(self.image_count + 1,int(self.var_number_images.get())))

    # Se sobreescribe este metodo
    def init_game(self):
        self.generate_game()

    # Se sobreescribe este metodo
    def update_game(self):
        if not self.finished_game:
            # Actualizamos el juego     
            if self.image_count < int(self.var_number_images.get()):
                if len(self.selected_image_points) < len(self.image_points):
                    ux,uy=Game3Functions.convert_my_system_to_matrix_system(u1s=self.my_point_player.position[None,:], canvas_dim=(self.canvas_width,self.canvas_height), image_dim=self.image_player.size).flatten()
                    rc=self.my_point_player.size[0]/2
                    # Verificamos si cada punto se encuentra dentro del circulo
                    for i in range(len(self.image_points)):
                        if i not in self.selected_image_points:
                            x,y=self.image_points[i]
                            if (x - ux)**2 + (y - uy)**2 <= rc**2:
                                # El punto se encuentra dentro del circulo o en el limite
                                self.selected_image_points.append(i)
                                # Cambiar de color al punto
                                self.my_canvas.itemconfigure(list(filter(lambda elem: elem.name == "point_player_{}".format(i), self.players_list))[0]._id, fill="green")
                else:
                    # Cambiar de imagen
                    self.image_count+=1
                    if self.image_count < int(self.var_number_images.get()):
                        self.var_level.set(value="Imagen {} de {}".format(self.image_count + 1,int(self.var_number_images.get())))
                        self.selected_image_points=[]
                        self.generate_game()
            else:
                self.end_game()

    # Se sobreescribe este metodo
    def estimation_data(self, filtered_estimations_results):
        # Actualizamos a los jugadores en funcion de la entrada recibida
        vws=filtered_estimations_results['mediapipe_pose_estimation']['vws']

        if vws is not None:
            """
            Aqui se toma en cuenta:
                - La eleccion del brazo
                - La eleccion del posicionamiento del plano
                - La distancia al plano
                - El ancho de la imagen en la realidad
                - El radio del circulo
            """
            try:
                # Ancho,Alto de la imagen  
                wip,hip=self.image_player.size
                # Vector que va del hombro a la muneca
                v=vws[15] - vws[11] if self.var_arm.get() == "Brazo izquierdo" else vws[16] - vws[12]
                # Posicion del hombro
                ov=vws[11] if self.var_arm.get() == "Brazo izquierdo" else vws[12]
                A=np.array([[0,1,-v[0]],[0,0,-v[1]],[-1,0,-v[2]]]) if self.var_plane_position.get() == "De frente" else np.array([[0,0,-v[0]],[0,-1,-v[1]],[-1,0,-v[2]]]) if self.var_plane_position.get() == "A la izquierda" else np.array([[0,0,-v[0]],[0,1,-v[1]],[-1,0,-v[2]]])
                t1w=np.array([0,-int(self.var_plane_distance.get()),0]) if self.var_plane_position.get() == "De frente" else np.array([-int(self.var_plane_distance.get()),0,0]) if self.var_plane_position.get() == "A la izquierda" else np.array([int(self.var_plane_distance.get()),0,0])
                b=ov[:,None] - t1w[:,None]
                alpha=int(self.var_reality_image_width.get())/wip
                wim=alpha*wip
                him=alpha*hip
                x_=np.array([wim/2,him/2,0])
                x=(np.linalg.inv(A)@(b + A@x_[:,None])).flatten()
                tp=x[2]
                u2xp,u2yp=(1/alpha)*x[0:2]
                if tp > 0:
                    # Significa que el brazo apunta hacia al plano
                    u1=Game3Functions.convert_matrix_system_to_my_system(u2s=np.array([[u2xp,u2yp]]), canvas_dim=(self.canvas_width,self.canvas_height), image_dim=self.image_player.size)
                    u1=u1.flatten()
                    # Actualizamos a los jugadores
                    if (u1[0] >= 0 and u1[0] <= self.canvas_width) and (u1[1] >= 0 and u1[1] <= self.canvas_height):
                        # Solo si el punto se encuentra dentro del canvas
                        rc=int(self.var_circle_radius.get())
                        self.my_point_player.position=u1
                        if (rc*2,rc*2) != self.my_point_player.size:
                            self.my_point_player.update_image_size(size=(rc*2,rc*2))
                        self.my_point_player.update_player()
                elif tp < 0:
                    # Significa que el brazo apunta en direccion contraria al plano
                    pass
            except np.linalg.LinAlgError:
                # Significa que A no es invertible, lo que quiere decir que la recta parametrizada que va del hombro a la muneca jamas interceptara al plano
                pass

    def generate_game(self):
        # Eliminamos todo
        self.delete_all()

        # Limpiamos la lista de jugadores
        self.players_list=[]
        
        dim=(int(self.canvas_width*(8/10)),int(self.canvas_height*(8/10)))
        self.image_player=Player(frame_game_canvas=self, name="image_player", primitive_shape="Image", size=dim, image_name=self.game_image_names[self.image_count], initial_position=np.array([self.canvas_width/2, self.canvas_height/2]), initial_velocity=np.zeros(2), initial_rotation_angle=0.0)
        self.image_points,_=Game3Functions.get_image_points(image_path=ui_images.get_game_image_path(name=self.game_image_names[self.image_count]), dim=dim)
        u1s=Game3Functions.convert_matrix_system_to_my_system(u2s=self.image_points, canvas_dim=(self.canvas_width,self.canvas_height), image_dim=self.image_player.size)
        for i in range(u1s.shape[0]):
            u1=u1s[i]
            point_player=Player(frame_game_canvas=self, name="point_player_{}".format(i), primitive_shape="Circle", size=(15,15), image_name="", initial_position=u1, initial_velocity=np.zeros(2), initial_rotation_angle=0.0, fill="red")
            self.players_list.append(point_player)
        
        self.my_point_player=Player(frame_game_canvas=self, name="my_point_player", primitive_shape="Image", size=(int(self.var_circle_radius.get()*2),int(self.var_circle_radius.get()*2)), image_name="circle_with_central_point", initial_position=np.array([self.canvas_width/2, self.canvas_height/2]), initial_velocity=np.zeros(2), initial_rotation_angle=0.0)

        self.players_list.extend([self.image_player,self.my_point_player])

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class Game4Functions():
    # vws (n,3)
    @staticmethod
    def get_local_euler_angles(vws, hand_type: Literal["left", "right"]):
        vts=Game4Functions.convert_world_system_to_traditional_system(vws=vws)
        rotation_matrices_list=Game4Functions.get_rotation_matrices_list(vts=vts, hand_type=hand_type)
        local_euler_angles=np.zeros((15,3))
        # Todos con respecto a la muñeca
        kinematic_chain_list=[(1,2,3),(5,6,7),(9,10,11),(13,14,15),(17,18,19)]
        T0t_inv=np.linalg.inv(rotation_matrices_list[0])
        count=0
        for i in range(len(kinematic_chain_list)):
            tup=kinematic_chain_list[i]
            A=T0t_inv
            for j in range(3):
                T=A @ rotation_matrices_list[tup[j]]
                if j < 2:
                    A=np.linalg.inv(T) @ A
                local_euler_angles[count,:]=utils.euler_angles_from_to_rotation_matrix(R=T)
                count+=1
        # Conversion de radianes a grados
        local_euler_angles=local_euler_angles * (180 / np.pi)
        return local_euler_angles

    # vws (n,3)
    @staticmethod
    def convert_world_system_to_traditional_system(vws):
        T=np.array([[0,1,0],[1,0,0],[0,0,1]])
        vts=T@vws.T
        vts=vts.T 
        return vts
    
    # vts (n,3)
    @staticmethod
    def get_rotation_matrices_list(vts, hand_type):
        # Matrices de rotacion para cada articulacion de la mano
        rotation_matrices_list=[]
        vectors_information_list=[0, [(1,2),(0,17)], [(2,3),(0,17)], [(3,4),(0,17)], None, [(5,6),(5,9)], [(6,7),(5,9)], [(7,8),(5,9)], None, [(9,10),(5,9)], [(10,11),(5,9)], [(11,12),(5,9)], None, [(13,14),(5,9)], [(14,15),(5,9)], [(15,16),(5,9)], None, [(17,18),(5,9)], [(18,19),(5,9)], [(19,20),(5,9)], None]
        count=0
        for elem in vectors_information_list:
            if elem is None:
                rotation_matrices_list.append(rotation_matrices_list[count-1])
            elif isinstance(elem, int):
                if elem == 0:
                    v1=(vts[5] - vts[0]) + (1/2) * (vts[17] - vts[5])
                    v2=np.cross(vts[17] - vts[0], vts[5] - vts[0])
                    v2=v2 if hand_type == "left" else -v2 
                    u1=utils.normalize_vector(v1)
                    u2=utils.normalize_vector(v2)
                    u3=np.cross(u2, u1)
                    u3=u3 if hand_type == "left" else -u3 
                    R=np.concatenate([u3[:,None], u2[:,None], u1[:,None]], axis=1)
                    rotation_matrices_list.append(R)
            else:
                t1, t2=elem
                v1=(vts[t1[1]] - vts[t1[0]])
                v2=np.cross(vts[t2[1]] - vts[t2[0]], v1)
                v2=v2 if hand_type == "left" else -v2 
                u1=utils.normalize_vector(v1)
                u2=utils.normalize_vector(v2)
                u3=np.cross(u2, u1)
                u3=u3 if hand_type == "left" else -u3 
                R=np.concatenate([u3[:,None], u2[:,None], u1[:,None]], axis=1)
                rotation_matrices_list.append(R)
            count+=1
        return rotation_matrices_list
    
    @staticmethod
    def convert_opencv_system_to_trditional_system(p, original_image_size, image_player_size, canvas_width, canvas_height):
        fw,fh=original_image_size[0]/image_player_size[0],original_image_size[1]/image_player_size[1]
        u1x,u1y=p
        u2x,u2y=u1x/fw,u1y/fh
        p_=np.array([canvas_width/2 - image_player_size[0]/2, canvas_height/2 + image_player_size[1]/2]) + np.array([u2x, -u2y])
        return p_    

class FrameGameCanvas4(GameCanvasTemplate):
    def __init__(self, master, canvas_width, canvas_height, **kwargs):
        GameCanvasTemplate.__init__(self, master=master, canvas_width=canvas_width, canvas_height=canvas_height, **kwargs)
        self.init_ctk_vars()
        self.init_game_vars(ctk_vars=False)

        # Cargamos los modelos de clasificacion
        self.load_clasification_models()

        frame_container=CreateFrame(master=self, grid_frame=GridFrame(dim=(6,1), arr=None), fg_color="lightpink")
        frame_container.insert_element(cad_pos="0,0", element=FrameHelp.create_option_menu(master=frame_container, command=self.change_hand_image, size=12, weight="bold", values=["Mano izquierda", "Mano derecha"], variable=self.var_hand), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="1,0", element=GameFunctions.create_slider_container(master=frame_container, title="Numero de repeticiones", message_format="{} repeticion(es)", message_size=16, from_=1, to=100, number_of_steps=99, variable=self.var_number_repetitions), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="2,0", element=GameFunctions.create_message_container(master=frame_container, title="Nivel", message_size=16, variable=self.var_level), padx=5, pady=5, sticky="ew")
        frame_container.insert_element(cad_pos="3,0", element=FrameHelp.create_button(master=frame_container, text="", command=self.button_restart, img_type="image", img_name="restart", img_width=50, img_height=50, fg_color="transparent"), padx=5, pady=5, sticky="")

        # Solo en etapa de obtencion de datos
        # frame_container.insert_element(cad_pos="4,0", element=FrameHelp.create_button(master=frame_container, text="COMENZAR", command=self.init_data_recollection, fg_color="coral"), padx=5, pady=5, sticky="")
        # frame_container.insert_element(cad_pos="5,0", element=FrameHelp.create_option_menu(master=frame_container, size=12, weight="bold", values=["open_hand", "closed_hand"], variable=self.var_hand_shape), padx=5, pady=5, sticky="ew")

        self.insert_element(cad_pos="0,0", element=frame_container, padx=5, pady=5, sticky="ew")

        self.init_game()

    # Se sobreescribe este metodo
    def init_ctk_vars(self):
        self.var_hand=ctk.StringVar(value="Mano izquierda")
        self.var_number_repetitions=ctk.DoubleVar(value=5)
        self.var_level=ctk.StringVar(value="")

        # Solo en etapa de obtencion de datos 
        self.var_hand_shape=ctk.StringVar(value="open_hand")

    # Se sobreescribe este metodo
    def init_game_vars(self, ctk_vars=False):
        if ctk_vars:
            self.var_hand.set(value="Mano izquierda")
            self.var_number_repetitions.set(value=5)
            self.var_level.set(value="")

            # Solo en etapa de obtencion de datos 
            self.var_hand_shape.set(value="open_hand")
        self.finished_game=True
        self.level_count=0
        self.number_repetitions=0
        self.original_image_size=(612,612)
        self.finger_names_list=["thumb_hand","index_hand","middle_hand","ring_hand","pinky_hand"]
        # thumb, index, middle, ring, pinky, palm
        self.dict_hand_principal_points={
            "Mano izquierda": np.array([[487,359],[370,142],[253,119],[164,156],[88,255],[289,385]]),
            "Mano derecha": np.array([[121,354],[239,137],[357,122],[447,159],[520,254],[332,374]])
        }
        self.current_hand_indexes_list=[]
        self.game_path="{}/game".format(os.path.dirname(os.path.abspath('__file__')))
        self.prediction_list=[]

        # Solo en etapa de obtencion de datos
        self.count_local_euler_angles=0
        self.data_recollection=False
        self.total_data=200
        self.local_euler_angles=np.zeros((15,3))

    # Se sobreescribe este metodo
    def init_play_vars(self):
        if self.finished_game:
            self.finished_game=False
            self.number_repetitions=int(self.var_number_repetitions.get())
            self.level_count=0
            self.var_level.set(value="Escenario {} de {}".format(self.level_count + 1,self.number_repetitions))
            self.select_random_fingers()

    # Se sobreescribe este metodo
    def init_game(self):
        self.change_hand_image(hand=self.var_hand.get())

    # Se sobreescribe este metodo
    def update_game(self):
        pass

    # Se sobreescribe este metodo
    def estimation_data(self, filtered_estimations_results):
        # Actualizamos a los jugadores en funcion de la entrada recibida
        l_vws=filtered_estimations_results['mediapipe_hand_estimation']['left']['vws']
        r_vws=filtered_estimations_results['mediapipe_hand_estimation']['right']['vws']

        if (self.var_hand.get() == "Mano izquierda" and l_vws is not None) or (self.var_hand.get() == "Mano derecha" and r_vws is not None):
            if self.var_hand.get() == "Mano izquierda":
                self.local_euler_angles=Game4Functions.get_local_euler_angles(vws=l_vws, hand_type="left")
                self.prediction_list=self.predict(hand="left_hand", local_euler_angles=self.local_euler_angles)
            elif self.var_hand.get() == "Mano derecha":
                self.local_euler_angles=Game4Functions.get_local_euler_angles(vws=r_vws, hand_type="right")
                self.prediction_list=self.predict(hand="right_hand", local_euler_angles=self.local_euler_angles)

            if self.data_recollection:
                self.save_local_euler_angles()

            self.draw_predictions(prediction_list=self.prediction_list)

            if self.var_game_state.get() == "play":
                if not self.finished_game:
                    # Actualizamos el juego     
                    if self.level_count < self.number_repetitions:
                        # Debemos verificar si la mano de la persona cumple con lo que se pide
                        if self.fingers_verification(prediction_list=self.prediction_list.copy()):
                            # Mostramos imagen de que fue exitoso y pausamos por un momento antes de continuar
                            finish_player=Player(frame_game_canvas=self, name="temp_finish_player", primitive_shape="Image", size=(100,100), image_name="finish", initial_position=np.array([self.canvas_width/2, self.canvas_height/2]), initial_velocity=np.zeros(2), initial_rotation_angle=0.0)
                            time.sleep(2)
                            self.my_canvas.delete(finish_player._id)
                            # Actualizamos 
                            self.level_count+=1
                            if self.level_count < self.number_repetitions:
                                self.var_level.set(value="Escenario {} de {}".format(self.level_count + 1,self.number_repetitions))
                                self.select_random_fingers()
                    else:
                        self.end_game()

    # Solo en etapa de obtencion de datos
    def init_data_recollection(self):
        # Antes de iniciar se deben eliminar todos los archivos actuales manualmente 
        self.count_local_euler_angles=0
        self.data_recollection=True

    # Solo en etapa de obtencion de datos
    def save_local_euler_angles(self):
        if self.count_local_euler_angles < self.total_data:
            self.count_local_euler_angles+=1
            path=os.path.join(self.game_path, "dataset/{}".format("left_hand" if self.var_hand.get() == "Mano izquierda" else "right_hand"))
            path=os.path.join(path, "{}_local_euler_angles_{}.npy".format(str(self.count_local_euler_angles).zfill(3), self.var_hand_shape.get()))
            np.save(path, self.local_euler_angles)
            print("Count: {}".format(self.count_local_euler_angles))
        else:
            self.data_recollection=False

    def load_clasification_models(self):
        self.dict_clasification_models={}
        for hand in ["left_hand", "right_hand"]:
            self.dict_clasification_models[hand]={}
            path="{}/clean_dataset/{}".format(self.game_path,hand)
            path_save_params="{}/models".format(path)
            for model_name in self.finger_names_list:
                model=SequentialModel()
                dict_params=model.load_params(path=path_save_params, name=model_name)
                self.dict_clasification_models[hand][model_name]={
                    "model": model,
                    "dict_params": dict_params
                }

    def predict(self, hand, local_euler_angles):
        prediction_list=[]
        for i in range(len(self.finger_names_list)):
            finger_name=self.finger_names_list[i]
            model=self.dict_clasification_models[hand][finger_name]["model"]
            dict_params=self.dict_clasification_models[hand][finger_name]["dict_params"]
            X_train_mean,X_train_std=dict_params["other_configurations"]["normalization"]
            X_new=local_euler_angles[3*i:3*(i+1),:].flatten(order='C')[None,:]
            X_new_norm,_,_=StandardNormalization.transform(data=X_new, mean_std=(X_train_mean,X_train_std))
            Y_predicted=model.predict(X=X_new_norm)
            prediction_list.append(Y_predicted[0,0])
        return prediction_list

    def fingers_verification(self, prediction_list):
        if len(self.current_hand_indexes_list) == 1 and self.current_hand_indexes_list[0] == 5:
            # Verificar que todos los dedos esten abajo
            return True if sum(prediction_list) == 0 else False
        else:
            return np.where(np.array(prediction_list) == 1)[0].tolist() == self.current_hand_indexes_list

    def draw_predictions(self, prediction_list):
        # Solo mostramos los dedos que estan levantados
        self.delete_predicted_hand_principal_points()
        ps=self.dict_hand_principal_points[self.var_hand.get()]
        for i in range(len(prediction_list)):
            band=prediction_list[i]
            if band:
                position=Game4Functions.convert_opencv_system_to_trditional_system(p=ps[i], original_image_size=self.original_image_size, image_player_size=self.image_player.size, canvas_width=self.canvas_width, canvas_height=self.canvas_height)
                predicted_hand_principal_point_player=Player(frame_game_canvas=self, name="predicted_hand_principal_point_player_{}".format(i), primitive_shape="Image", size=(40,40), image_name="circle_with_central_point", initial_position=position, initial_velocity=np.zeros(2), initial_rotation_angle=0.0)
                self.players_list.append(predicted_hand_principal_point_player)

    def change_hand_image(self, hand):
        if self.finished_game:
            self.delete_all()
            dim=(int(self.canvas_width*(8/10)),int(self.canvas_height*(8/10)))
            self.image_player=Player(frame_game_canvas=self, name="image_player", primitive_shape="Image", size=dim, image_name="left_hand" if hand == "Mano izquierda" else "right_hand", initial_position=np.array([self.canvas_width/2, self.canvas_height/2]), initial_velocity=np.zeros(2), initial_rotation_angle=0.0)
            self.players_list=[self.image_player]

    def button_restart(self):
        if not self.finished_game:
            self.select_random_fingers()

    def delete_predicted_hand_principal_points(self):
        temp_players_list=[]
        for i in range(len(self.players_list)):
            if self.players_list[i].name.split("_")[0] != "predicted":
                temp_players_list.append(self.players_list[i])
            else:
                self.my_canvas.delete(self.players_list[i]._id)
        self.players_list=temp_players_list

    def delete_hand_principal_points(self):
        temp_players_list=[]
        for i in range(len(self.players_list)):
            if self.players_list[i].name.split("_")[0] != "hand":
                temp_players_list.append(self.players_list[i])
            else:
                self.my_canvas.delete(self.players_list[i]._id)
        self.players_list=temp_players_list
        
    def select_random_fingers(self):
        self.delete_hand_principal_points()
        # (6,2)
        ps=self.dict_hand_principal_points[self.var_hand.get()]
        # n: {0,1,2,3,4,5} (numero de dedos levantados)
        n=np.random.randint(0,5+1) 
        self.current_hand_indexes_list=[5] if n == 0 else random.sample([0,1,2,3,4],n)
        self.current_hand_indexes_list.sort()
        filter_ps=ps[self.current_hand_indexes_list,:]
        for i in range(filter_ps.shape[0]):
            position=Game4Functions.convert_opencv_system_to_trditional_system(p=filter_ps[i], original_image_size=self.original_image_size, image_player_size=self.image_player.size, canvas_width=self.canvas_width, canvas_height=self.canvas_height)
            hand_principal_point_player=Player(frame_game_canvas=self, name="hand_principal_point_player_{}".format(i), primitive_shape="Circle", size=(30,30), image_name="", initial_position=position, initial_velocity=np.zeros(2), initial_rotation_angle=0.0, fill="green")
            self.players_list.append(hand_principal_point_player)

    