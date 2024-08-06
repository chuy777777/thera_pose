import numpy as np
import cv2
import customtkinter  as ctk
from PIL import Image
from io import StringIO
import pprint

def get_string_pprint(obj):
    s = StringIO()
    pprint.pprint(object=obj, stream=s)
    return s.getvalue()

def resize_image(scale_percent, img):
    if img is not None:
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        new_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return new_img
    return None

def frame_to_img(frame):
    if frame is not None:
        img=Image.fromarray(frame)
        img=ctk.CTkImage(img,size=(frame.shape[1],frame.shape[0]))
        return img
    return None

def rgb_to_hex(color):
    return '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

def random_value_in_range(a,b):
    return a + np.random.random()*(b-a)

# Angulo en radianes
def Rx(psi):
    R=np.array([[1,0,0],[0,np.cos(psi),-np.sin(psi)],[0,np.sin(psi),np.cos(psi)]])
    return R

# Angulo en radianes
def Ry(theta):
    R=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    return R

# Angulo en radianes
def Rz(phi):
    R=np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]])
    return R

# Angulo en radianes
# Rotation en X, despues en Y, y por ultimo en Z
def RzRyRx(psi, theta, phi):
    R=Rz(phi=phi)@Ry(theta=theta)@Rx(psi=psi)
    return R

"""
R=RzRyRx
(rotation en X, despues en Y, y por ultimo en Z)
(X,    Y,     Z)
(psi, theta, phi)
"""
# R: (3,3)
# Devuelve: (3,) (en radianes)
def euler_angles_from_to_rotation_matrix(R):
    R11,R12,R13,R21,R22,R23,R31,R32,R33=R.flatten(order='C')
    euler_angles=np.zeros(3)
    if R31 == -1 or R31 == 1:
        if R31 == -1:
            theta=np.pi/2
            phi=0  
            psi=np.arctan2(R12, R13) + phi
            euler_angles=np.array([psi, theta, phi])
        else:
            theta=-np.pi/2
            phi=0  
            psi=np.arctan2(-R12, -R13) - phi
            euler_angles=np.array([psi, theta, phi])
    elif R31 != -1 and R31 != 1:
        theta_1=np.arcsin(-R31)
        theta_2=np.pi - theta_1
        psi_1=np.arctan2(R32 / np.cos(theta_1), R33 / np.cos(theta_1))
        psi_2=np.arctan2(R32 / np.cos(theta_2), R33 / np.cos(theta_2))
        phi_1=np.arctan2(R21 / np.cos(theta_1), R11 / np.cos(theta_1))
        phi_2=np.arctan2(R21 / np.cos(theta_2), R11 / np.cos(theta_2))
        euler_angles=np.array([psi_1, theta_1, phi_1])
    return euler_angles

# def euler_angles_from_to_rotation_matrix(R):
#     R11,R12,R13,R21,R22,R23,R31,R32,R33=R.flatten()
#     euler_angles=np.zeros(3)
#     if R31 != -1 and R31 != 1:
#         theta_1=-np.arcsin(R31)
#         theta_2=np.pi-theta_1
#         psi_1=np.arctan2(R32/np.cos(theta_1), R33/np.cos(theta_1))
#         psi_2=np.arctan2(R32/np.cos(theta_2), R33/np.cos(theta_2))
#         phi_1=np.arctan2(R21/np.cos(theta_1), R11/np.cos(theta_1))
#         phi_2=np.arctan2(R21/np.cos(theta_2), R11/np.cos(theta_2))
#         euler_angles=np.array([psi_1, theta_1, phi_1])
#     else:
#         phi=0  
#         if R31 == -1:
#             theta=np.pi/2
#             psi=phi+np.arctan2(R12, R13)
#             euler_angles=np.array([psi, theta, phi])
#         else:
#             theta=-np.pi/2
#             psi=-phi+np.arctan2(-R12, -R13)
#             euler_angles=np.array([psi, theta, phi])
#     return euler_angles
