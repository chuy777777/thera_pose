#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4l2-ctl --list-devices

    Synaptics RMI4 Touch Sensor (rmi4:rmi4-00.fn54):
    	/dev/v4l-touch0
    
    USB PHY 2.0: USB CAMERA (usb-0000:00:14.0-3.3):
    	/dev/video4
    	/dev/video5
    
    HX-USB Camera: HX-USB Camera (usb-0000:00:14.0-3.4):
    	/dev/video2
    	/dev/video3
    
    EasyCamera: EasyCamera (usb-0000:00:14.0-8):
    	/dev/video0
    	/dev/video1


EL SIGUIENTE ANALISIS SE HARA CON RESPECTO A LOS APUNTES QUE HE REALIZADO
"""

import cv2
import requests
import urllib
import numpy as np
import time
import glob
import scipy
from scipy.optimize import root, minimize, fsolve, least_squares

"""
Ruta donde se encuentran las imagenes
"""
path_calibration_images='/home/chuy/Practicas/Python/Project/UI15/test_calibration_images'

"""
Captura de imagenes del tablero de ajedrez
"""
# count=0
# camera_path="/dev/video2"
# cap=cv2.VideoCapture(camera_path)
# while True:
#     if cap.isOpened():
#         # ret: Booleano
#         # frame: Imagen capturada
#         ret,frame=cap.read()
#         if ret:
#             gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#             cv2.imshow("Image",gray)
#             # Espera el tiempo dado para destruir la ventana (en ms)
#             key=cv2.waitKey(1)
#             if key==ord('t'):
#                 name=path_calibration_images+'/img{}.png'.format(str(count).zfill(2))
#                 cv2.imwrite(name,gray)
#                 count=count+1
#             if key==ord('q'):
#                 break  
# # Destruye todas las ventanas abiertas 
# cv2.destroyAllWindows()
# # Cierra el flujo de captura de video
# cap.release()  

"""
Definicion de funciones
"""

# Parametros: params (2,), v3p (2,), K_ (3,3), q (5,1)
# Devuelve (2,)
def fun_undistorted_points(params, v3p, K_, q):
    v1_=params[:,None]
    x1_,y1_=v1_.flatten()
    r=np.linalg.norm(v1_)
    Md=np.array([[x1_*r**2,x1_*r**4,x1_*r**6,2*x1_*y1_,r**2+2*x1_**2],[y1_*r**2,y1_*r**4,y1_*r**6,r**2+2*y1_**2,2*x1_*y1_]])
    d=Md@q
    v1__=v1_+d
    v1h__=np.concatenate([v1__, np.ones((1,1))], axis=0)
    v3ph=np.concatenate([v3p[:,None], np.ones((1,1))], axis=0)
    f=(v3ph - K_@v1h__).flatten()
    f=f[0:2]
    return f
    
# Parametros:  K (3,3), q (5,1), v3ps (m,2)
# Devuelve (m,2)
def get_undistorted_points(K, q, v3ps):
    m=v3ps.shape[0]
    v3ps_undistorted=np.zeros((m,2))
    k11,k12,k13,_,k22,k23,_,_,_=K.flatten()
    K_=np.array([[-k11,-k12,k13],[0,-k22,k23],[0,0,1]])
    K__inv=np.linalg.inv(K_)
    for i in range(m):
        v3p=v3ps[i]
        v3ph=np.concatenate([v3p[:,None], np.ones((1,1))], axis=0)
        initial_params=(K__inv@v3ph).flatten()
        initial_params=initial_params[0:2]
        res=root(lambda params: fun_undistorted_points(params=params, v3p=v3p, K_=K_, q=q), initial_params, method='hybr', jac=False)
        v1_=res.x
        
        v1h_=np.concatenate([v1_[:,None], np.ones((1,1))], axis=0)
        v3ph_undistorted=(K_@v1h_).flatten()
        v3p_undistorted=v3ph_undistorted[0:2]
        v3ps_undistorted[i]=v3p_undistorted
    return v3ps_undistorted

# Parametros: K (3,3), q (5,1), v3ps (n,2), vws (n,3)
# Devuelve (3,4)
# NOTA: SE NECESITAN MINIMO 4 PARES DE PUNTOS CORRESPONDIENTES PARA DETERMINAR LAS 8 INCOGNITAS DE G
def get_Q(K, q, v3ps, vws):
    v3ps_undistorted=get_undistorted_points(K=K, q=q, v3ps=v3ps)
    G=get_G(v3ps=v3ps_undistorted, vws=vws)
    Gs=G[None,:,:]
    Qs=get_Qs(K=K, Gs=Gs)
    Q=Qs[0]
    return Q

# Parametros: K (3,3), Q (3,4), q (5,1), vws (n,3)
# Devuelve (n,2)
def get_2D_points(K, Q, q, vws):
    n=vws.shape[0]
    v3ps=np.zeros((n,2))
    k11,k12,k13,_,k22,k23,_,_,_=K.flatten()
    K_=np.array([[-k11,-k12,k13],[0,-k22,k23],[0,0,1]])
    for i in range(n):
        vw=vws[i]
        vwh=np.concatenate([vw[:,None], np.ones((1,1))], axis=0)
        vc=(Q@vwh).flatten()
        xc,yc,zc=vc
        v1_=np.array([[-xc/zc],[-yc/zc]])
        x1_,y1_=v1_.flatten()
        r=np.linalg.norm(v1_)
        Md=np.array([[x1_*r**2,x1_*r**4,x1_*r**6,2*x1_*y1_,r**2+2*x1_**2],[y1_*r**2,y1_*r**4,y1_*r**6,r**2+2*y1_**2,2*x1_*y1_]])
        d=Md@q
        v1__=v1_ + d
        v1h__=np.concatenate([v1__, np.ones((1,1))], axis=0)
        v3ph=(K_@v1h__).flatten()
        v3p=v3ph[0:2]
        v3ps[i]=v3p
    return v3ps
    
# Parametros: K1 (3,3), Q1 (3,4), q1 (5,1), K2 (3,3), Q2 (3,4), q2 (5,1), v3ps1 (n,2), v3ps2 (n,2)
# Devuelve (n,3)
# OBTENER PUNTOS 3D A PARTIR DE PUNTOS 2D CORRESPONDIENTES DE 2 CAMARAS YA CALIBRADAS
def get_3D_points(K1, Q1, q1, K2, Q2, q2, v3ps1, v3ps2):
    n=v3ps1.shape[0]
    vws=np.zeros((n,3))
    P1=K1@Q1
    P2=K2@Q2
    p11_1,p12_1,p13_1,p14_1,p21_1,p22_1,p23_1,p24_1,p31_1,p32_1,p33_1,p34_1=P1.flatten(order='C')
    p11_2,p12_2,p13_2,p14_2,p21_2,p22_2,p23_2,p24_2,p31_2,p32_2,p33_2,p34_2=P2.flatten(order='C')
    v3ps1_undistorted=get_undistorted_points(K=K1, q=q1, v3ps=v3ps1)
    v3ps2_undistorted=get_undistorted_points(K=K2, q=q2, v3ps=v3ps2)
    for i in range(n):
        v3p1_undistorted=v3ps1_undistorted[i]
        v3p2_undistorted=v3ps2_undistorted[i]
        x3p_1,y3p_1=v3p1_undistorted
        x3p_2,y3p_2=v3p2_undistorted
        A=np.array([[p31_1*x3p_1-p11_1,p32_1*x3p_1-p12_1,p33_1*x3p_1-p13_1],[p31_1*y3p_1-p21_1,p32_1*y3p_1-p22_1,p33_1*y3p_1-p23_1],[p31_2*x3p_2-p11_2,p32_2*x3p_2-p12_2,p33_2*x3p_2-p13_2],[p31_2*y3p_2-p21_2,p32_2*y3p_2-p22_2,p33_2*y3p_2-p23_2]])
        b=np.array([[p14_1-p34_1*x3p_1],[p24_1-p34_1*y3p_1],[p14_2-p34_2*x3p_2],[p24_2-p34_2*y3p_2]])
        
        x=np.linalg(A.T@A)@A.T@b
        vw=x.flatten()
        vws[i]=vw
    return vws
    
# Parametros: img (imagen de la camara), K (3,3), q (5,1), board_dimensions (tupla), square_size (escalar) 
# Devuelve (imagen de la camara con dibujos)
def draw_3D_cube_on_chessboard(img, K, q, board_dimensions=(9,6), square_size=26.5):
    indexes=[0,9-1,9*5,9*6-1]
    v3ps=get_v3ps_from_chessboard_image(img=img, board_dimensions=(9,6))
    if v3ps is not None:
        v3ps=v3ps[indexes,:]
        vws=get_vws_from_chessboard(board_dimensions=(9,6), square_size=26.5)[indexes,:]
        Q=get_Q(K=K, q=q, v3ps=v3ps, vws=vws)
        
        new_img=img.copy()
        vws=np.concatenate([vws, vws + np.concatenate([np.zeros((4,1)), np.zeros((4,1)), 80*np.ones((4,1))], axis=1)], axis=0)
        conn=[(0,1),(1,3),(3,2),(2,0),(0,4),(1,5),(3,7),(2,6),(4,5),(5,7),(7,6),(6,4)]
        v3ps=get_2D_points(K=K, Q=Q, q=q, vws=vws)
        n=v3ps.shape[0]
        for i in range(n):
            x3p,y3p=v3ps[i].astype(int)
            color=(0,0,255) # BGR
            new_img=cv2.circle(new_img, (x3p,y3p), 5, color, -1)
        for i in range(len(conn)):
            p1,p2=conn[i]
            x3p_1,y3p_1=v3ps[p1].astype(int)
            x3p_2,y3p_2=v3ps[p2].astype(int)
            color=(0,255,0) # BGR
            new_img=cv2.line(new_img, (x3p_1,y3p_1), (x3p_2,y3p_2), color, 2)
        return new_img
    return None

# Parametros: Q (3,4)
# Devuelve 
def print_info_Q(Q):
  print("r1*r1: {:.5f} r2*r2: {:.5f} r3*r3: {:.5f} r1*r2: {:.5f} r1*r3: {:.5f} r2*r3: {:.5f}".format(np.dot(Q[:,0],Q[:,0]), np.dot(Q[:,1],Q[:,1]), np.dot(Q[:,2],Q[:,2]), np.dot(Q[:,0],Q[:,1]), np.dot(Q[:,0],Q[:,2]), np.dot(Q[:,1],Q[:,2])))

# Parametros: img (imagen tomada de camara), board_dimensions (tuple)
# Devuelve (m,2)
def get_v3ps_from_chessboard_image(img, board_dimensions=(9,6)):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Las detecciones se hacen de izquierda a derecha, de arriba a abajo
    ret,corners=cv2.findChessboardCorners(gray,board_dimensions,None,flags=cv2.CALIB_CB_FAST_CHECK)
    if ret:
        # print(corners.shape) # (m, 1, 2)
        corners=corners.squeeze()
        return corners
    return None

# Parametros: board_dimensions (tuple), square_size (en mm)
# Devuelve (m,3)
def get_vws_from_chessboard(board_dimensions=(9,6), square_size=26.5):
    m=board_dimensions[0]*board_dimensions[1]
    vws=np.zeros((m,3))
    count=0
    T=np.eye(2)*square_size
    for y in range(board_dimensions[1]):
        for x in range(board_dimensions[0]):
            v=np.array([[x],[y]])
            u=(T@v).flatten()
            vws[count]=np.array([u[0],u[1],0])
            count+=1
    return vws

# Parametros: v3ps (m,2), vws (m,3)
# Devuelve (3,3)
# NOTA: SE NECESITAN MINIMO 4 PARES DE PUNTOS CORRESPONDIENTES PARA DETERMINAR LAS 8 INCOGNITAS DE G
def get_G(v3ps, vws):
    m=v3ps.shape[0]
    A=np.zeros((2*m,8))
    b=np.zeros((2*m,1))
    for i in range(m):
        x3p,y3p=v3ps[i]
        xw,yw,_=vws[i]
        A_=np.array([[xw, yw, 1, 0, 0, 0, -xw*x3p, -yw*x3p],[0, 0, 0, xw, yw, 1, -xw*y3p, -yw*y3p]])
        b_=np.array([[x3p],[y3p]])
        A[2*i:2*(i+1),:]=A_
        b[2*i:2*(i+1),:]=b_
    x=np.linalg.inv(A.T@A)@A.T@b 
    x=np.concatenate([x, np.ones((1,1))], axis=0)
    G=np.reshape(x, (3,3), order='C')
    return G

# Parametros: Gs (l,3,3)
# Devuelve (3,3)
# NOTA: SE NECESITAN MINIMO 3 MATRICES G PARA DETERMINAR LAS 6 INCOGNITAS DE B
def get_B(Gs):
    # Descomposicion SVD de una matriz
    # u,s,vh=np.linalg.svd(A)
    # The rows of vh are the eigenvectors of A'A and the columns of u are the eigenvectors of AA'. 
    # In both cases the corresponding (possibly non-zero) eigenvalues are given by s**2.
    l=Gs.shape[0]
    A=np.zeros((2*l,6))
    for i in range(l):
        G=Gs[i]
        g11,g12,g13,g21,g22,g23,g31,g32,_=G.flatten(order='C')
        A_=np.array([[g11*g12, g12*g21+g11*g22, g12*g31+g11*g32, g21*g22, g22*g31+g21*g32, g31*g32],[g11**2-g12**2, 2*g11*g21-2*g12*g22, 2*g11*g31-2*g12*g32, g21**2-g22**2, 2*g21*g31-2*g22*g32, g31**2-g32**2]])
        A[2*i:2*(i+1),:]=A_
    u,s,vh=np.linalg.svd(A)
    h=np.reshape(vh[-1],(6,1))
    a,b,c,d,e,f=h.flatten()
    B=np.array([[a,b,c],[b,d,e],[c,e,f]])
    return B
    
# Parametros: B (3,3)
# Devuelve (3,3)
def get_K(B):
    L=np.linalg.cholesky(B)
    K=np.linalg.inv(L.T)
    return K

# Parametros: K (3,3), Gs (l,3,3)
# Devuelve (l,3,4)
def get_Qs(K, Gs):
    K_inv=np.linalg.inv(K)
    l=Gs.shape[0]
    Qs=np.zeros((l,3,4))
    for i in range(l):
        G=Gs[i]
        g1=G[:,[0]]
        g2=G[:,[1]]
        g3=G[:,[2]]
        h33=1/np.linalg.norm(K_inv@g1)
        iw=h33*K_inv@g1
        jw=h33*K_inv@g2
        kw=-np.cross(iw.flatten(),jw.flatten())[:,None]
        tw=h33*K_inv@g3
        Q=np.concatenate([iw,jw,kw,tw], axis=1)
        Qs[i]=Q
    return Qs

# Parametros: params (5+5+9*l,), l (escalar), v3ps_real_list [(m,2),...], vws (m,3)
# Devuelve (escalar)
def reprojection_error(params, l, v3ps_real_list, vws):
    error=0
    K=np.array([[[params[0],params[1],params[2]],[0,params[3],params[4]],[0,0,1]]])
    q=params[5:10][:,None]
    Qs_=np.reshape(params[10:10+9*l], (l,3,3), order='C')
    
    k11,k12,k13,_,k22,k23,_,_,_=K.flatten()
    K_=np.array([[-k11,-k12,k13],[0,-k22,k23],[0,0,1]])
    m=vws.shape[0]
    for i in range(l):
        v3ps_real=v3ps_real_list[i]
        Q_=Qs_[i]
        for j in range(m):
            v3p_real=v3ps_real[j]
            vwh_=np.concatenate([vws[j,0:2][:,None], np.ones((1,1))], axis=0)
            
            xc,yc,zc=(Q_@vwh_).flatten()
            v1_=np.array([[-xc/zc],[-yc/zc]])
            r=np.linalg.norm(v1_)
            x1_,y1_=v1_.flatten()
            Md=np.array([[x1_*r**2,x1_*r**4,x1_*r**6,2*x1_*y1_,r**2+2*x1_**2],[y1_*r**2,y1_*r**4,y1_*r**6,r**2+2*y1_**2,2*x1_*y1_]])
            d=Md@q
            v1__=v1_+d
            v1h__=np.concatenate([v1__, np.ones((1,1))], axis=0)
            v3ph=(K_@v1h__).flatten()
            v3p_predicted=v3ph[0:2]
            
            error+=np.linalg.norm(v3p_real-v3p_predicted)**2
    return error

# Parametros: params (5+5+9*l,), l (escalar), v3ps_real_list [(m,2),...], vws (m,3)
# Devuelve (5+5+9*l,)
def jac_reprojection_error(params, l, v3ps_real_list, vws):
    jac=np.zeros(5+5+9*l)
    error=0
    
    K=np.array([[[params[0],params[1],params[2]],[0,params[3],params[4]],[0,0,1]]])
    q=params[5:10][:,None]
    Qs_=np.reshape(params[10:10+9*l], (l,3,3), order='C')
    
    k11,k12,k13,_,k22,k23,_,_,_=K.flatten()
    K_=np.array([[-k11,-k12,k13],[0,-k22,k23],[0,0,1]])
    for i in range(l):
        v3ps_real=v3ps_real_list[i]
        Q_=Qs_[i]
        dfdQ_=np.zeros((1,9))
        m=v3ps_real.shape[0]
        for j in range(m):
            v3p_real=v3ps_real[j]
            vwh_=np.concatenate([vws[j,0:2][:,None], np.ones((1,1))], axis=0)
            xw,yw=vws[j,0:2]
            
            xc,yc,zc=(Q_@vwh_).flatten()
            v1_=np.array([[-xc/zc],[-yc/zc]])
            x1_,y1_=v1_.flatten()
            r=np.linalg.norm(v1_)
            Md=np.array([[x1_*r**2,x1_*r**4,x1_*r**6,2*x1_*y1_,r**2+2*x1_**2],[y1_*r**2,y1_*r**4,y1_*r**6,r**2+2*y1_**2,2*x1_*y1_]])
            d=Md@q
            v1__=v1_+d
            x1__,y1__=v1__.flatten()
            v1h__=np.concatenate([v1__, np.ones((1,1))], axis=0)
            v3ph=(K_@v1h__).flatten()
            v3p_predicted=v3ph[0:2]
            u=(v3p_real-v3p_predicted)[:,None]
            
            error+=np.linalg.norm(u)**2
            
            dfdK=(2*u.T)@(-np.eye(2))@(np.array([[-x1__,-y1__,1,0,0],[0,0,0,-y1__,1]]))
            dfdq=(2*u.T)@(-np.eye(2))@(np.array([[-k11,-k12],[0,-k22]]))@(np.eye(2))@(Md)
            
            dv1_dQ_=(np.array([[-1/zc,0,xc/zc**2],[0,-1/zc,yc/zc**2]]))@(np.kron(np.eye(3), np.array([[xw,yw,1]])))
            dMddv1_=np.array([[r**2+2*x1_**2,2*x1_*y1_],[r**4+4*x1_**2*r**2,4*x1_*y1_*r**2],[r**6+6*x1_**2*r**4,6*x1_*y1_*r**4],[2*y1_,2*x1_],[6*x1_,2*y1_],[2*x1_*y1_,r**2+2*y1_**2],[4*x1_*y1_*r**2,r**4+4*y1_**2*r**2],[6*x1_*y1_*r**4,r**6+6*y1_**2*r**4],[2*x1_,6*y1_],[2*y1_,2*x1_]])
            dfdQ_+=(2*u.T)@(-np.eye(2))@(np.array([[-k11,-k12],[0,-k22]]))@((dv1_dQ_) + (np.kron(np.eye(2), q.T))@(dMddv1_)@(dv1_dQ_))
            
            jac[0:5]+=dfdK.flatten()
            jac[5:10]+=dfdq.flatten()
            
        jac[10+9*i:10+9*(i+1)]=dfdQ_.flatten()
    return error,jac

# Parametros: params (5+5+9*l+3*l,), l (escalar), v3ps_real_list [(m,2),...], vws (m,3)
# Devuelve (5+5+9*l+3*l,)
def jac_reprojection_error_with_constraints(params, l, v3ps_real_list, vws):
    jac=np.zeros(5+5+9*l+3*l)
    error=0
    
    K=np.array([[[params[0],params[1],params[2]],[0,params[3],params[4]],[0,0,1]]])
    q=params[5:10][:,None]
    Qs_=np.reshape(params[10:10+9*l], (l,3,3), order='C')
    lambdas=params[10+9*l:]
    
    k11,k12,k13,_,k22,k23,_,_,_=K.flatten()
    K_=np.array([[-k11,-k12,k13],[0,-k22,k23],[0,0,1]])
    for i in range(l):
        v3ps_real=v3ps_real_list[i]
        Q_=Qs_[i]
        iw=Q_[:,0]
        jw=Q_[:,1]
        dfdQ_=np.zeros((1,9))
        m=v3ps_real.shape[0]
        for j in range(m):
            v3p_real=v3ps_real[j]
            vwh_=np.concatenate([vws[j,0:2][:,None], np.ones((1,1))], axis=0)
            xw,yw=vws[j,0:2]
            
            xc,yc,zc=(Q_@vwh_).flatten()
            v1_=np.array([[-xc/zc],[-yc/zc]])
            x1_,y1_=v1_.flatten()
            r=np.linalg.norm(v1_)
            Md=np.array([[x1_*r**2,x1_*r**4,x1_*r**6,2*x1_*y1_,r**2+2*x1_**2],[y1_*r**2,y1_*r**4,y1_*r**6,r**2+2*y1_**2,2*x1_*y1_]])
            d=Md@q
            v1__=v1_+d
            x1__,y1__=v1__.flatten()
            v1h__=np.concatenate([v1__, np.ones((1,1))], axis=0)
            v3ph=(K_@v1h__).flatten()
            v3p_predicted=v3ph[0:2]
            u=(v3p_real-v3p_predicted)[:,None]
            
            error+=np.linalg.norm(u)**2
            
            dfdK=(2*u.T)@(-np.eye(2))@(np.array([[-x1__,-y1__,1,0,0],[0,0,0,-y1__,1]]))
            dfdq=(2*u.T)@(-np.eye(2))@(np.array([[-k11,-k12],[0,-k22]]))@(np.eye(2))@(Md)
            
            dv1_dQ_=(np.array([[-1/zc,0,xc/zc**2],[0,-1/zc,yc/zc**2]]))@(np.kron(np.eye(3), np.array([[xw,yw,1]])))
            dMddv1_=np.array([[r**2+2*x1_**2,2*x1_*y1_],[r**4+4*x1_**2*r**2,4*x1_*y1_*r**2],[r**6+6*x1_**2*r**4,6*x1_*y1_*r**4],[2*y1_,2*x1_],[6*x1_,2*y1_],[2*x1_*y1_,r**2+2*y1_**2],[4*x1_*y1_*r**2,r**4+4*y1_**2*r**2],[6*x1_*y1_*r**4,r**6+6*y1_**2*r**4],[2*x1_,6*y1_],[2*y1_,2*x1_]])
            dfdQ_+=(2*u.T)@(-np.eye(2))@(np.array([[-k11,-k12],[0,-k22]]))@((dv1_dQ_) + (np.kron(np.eye(2), q.T))@(dMddv1_)@(dv1_dQ_))
            
            jac[0:5]+=dfdK.flatten()
            jac[5:10]+=dfdq.flatten()
        
        lambda1,lambda2,lambda3=lambdas[3*i:3*i+3]
        dfdQ_+=-np.array([[2*lambda1*iw[0]+lambda3*jw[0],2*lambda2*jw[0]+lambda3*iw[0],0,2*lambda1*iw[1]+lambda3*jw[1],2*lambda2*jw[1]+lambda3*iw[1],0,2*lambda1*iw[2]+lambda3*jw[2],2*lambda2*jw[2]+lambda3*iw[2],0]])
        
        error+=-(lambda1*(np.dot(iw,iw)-1) + lambda2*(np.dot(jw,jw)-1) + lambda3*np.dot(iw,jw))
        
        jac[10+9*i:10+9*(i+1)]=dfdQ_.flatten()
        jac[10+9*l+3*i:10+9*l+3*(i+1)]=-np.array([np.dot(iw,iw)-1,np.dot(jw,jw)-1,np.dot(iw,jw)])
    return jac

"""
Proceso de calibracion
"""
image_names=glob.glob(path_calibration_images+'/*.png')
image_names.sort()

board_dimensions=(9,6)
square_size=26.5
n=len(image_names)
m=board_dimensions[0]*board_dimensions[1]
vws=get_vws_from_chessboard(board_dimensions=board_dimensions, square_size=square_size)

correct_images=[]
G_list=[]
for image_name in image_names:
    img=cv2.imread(image_name)
    v3ps=get_v3ps_from_chessboard_image(img=img, board_dimensions=board_dimensions)
    if v3ps is not None:
        correct_images.append((image_name, v3ps))
        G=get_G(v3ps=v3ps, vws=vws)
        G_list.append(G)
        
l=len(G_list)
Gs=np.zeros((l,3,3))
for i in range(l):
    Gs[i]=G_list[i]
    
B=get_B(Gs=Gs)
# La K aqui es una aproximacion a la real (aun falta el proceso de refinamiento)
K=get_K(B=B)
# Esto ayuda al metodo numerico a converger mas rapido
K=K/K[-1,-1] # Ya que sabemos que K33=1 
Qs=get_Qs(K=K, Gs=Gs)

print(K)

"""
Proceso de refinamiento

Hasta aqui los parametros obtenidos no son los reales.

Estos parametros serviran de valores iniciales para el proceso de refinamiento
de parametros.
    - K
        Se refinara
    - Qs
        Se refinaran
    - q (parametros de distorsion)
        Se encontraran
"""

# Puntos 2D reales con los que se medira el error de nuestro modelo
# Estos puntos son obtenidos por algoritmos de dereccion de esquinas para el tablero de ajedrez
v3ps_real_list=list(map(lambda elem: elem[1], correct_images))

# Parametros iniciales
initial_params=np.concatenate([[K[0,0], K[0,1], K[0,2], K[1,1], K[1,2]], np.zeros(5), Qs[:,:,[0,1,3]].flatten(order='C'), np.zeros(3*l)], axis=0)
error=reprojection_error(params=initial_params, l=l, v3ps_real_list=v3ps_real_list, vws=vws)
jac=jac_reprojection_error_with_constraints(params=initial_params, l=l, v3ps_real_list=v3ps_real_list, vws=vws)
print("Initial error: {}".format(error)) # Initial error: 70.56501206852091
print("Initial norm jac: {}".format(np.linalg.norm(jac))) # Initial norm jac: 211.55790947395602
# Tardo alrededor de 4 minutos
res=root(lambda params: jac_reprojection_error_with_constraints(params=params, l=l, v3ps_real_list=v3ps_real_list, vws=vws), initial_params, method='hybr', jac=False)
params_opt=res.x
error=reprojection_error(params=params_opt, l=l, v3ps_real_list=v3ps_real_list, vws=vws)
jac=jac_reprojection_error_with_constraints(params=params_opt, l=l, v3ps_real_list=v3ps_real_list, vws=vws)
print("Final error: {}".format(error)) # Final error: 71.01147301662229
print("Final norm jac: {}".format(np.linalg.norm(jac))) # Final norm jac: 0.0001074339529036248

K_opt=np.array([[params_opt[0],params_opt[1],params_opt[2]],[0,params_opt[3],params_opt[4]],[0,0,1]])
q_opt=params_opt[5:10][:,None]
Qs__opt=np.reshape(params_opt[10:10+9*l], (l,3,3), order='C')
lambdas_opt=params_opt[10+9*l:][:,None]
Qs_opt=np.zeros((l,3,4))
for i in range(l):
    Q__opt=Qs__opt[i]
    Qs_opt[i]=np.concatenate([Q__opt[:,0:2],-np.cross(Q__opt[:,0],Q__opt[:,1])[:,None],Q__opt[:,[2]]], axis=1)

print(K_opt)
print(q_opt)

# Se cumplen todas las restricciones que impusimos sobre la matriz Q
for i in range(l):
    print_info_Q(Qs_opt[i])
    
"""
Guardamos parametros obtenidos 
"""
# Camara: HX-USB    
np.save(path_calibration_images+"/K.npy", K_opt)
np.save(path_calibration_images+"/q.npy", q_opt) 
np.save(path_calibration_images+"/Qs.npy", Qs_opt) 
    
"""
Cargar parametros
"""
K=np.load(path_calibration_images+"/K.npy")
q=np.load(path_calibration_images+"/q.npy") 
Qs=np.load(path_calibration_images+"/Qs.npy") 

"""
Correccion de imagen distorsionada a imagen no distorsionada
"""
# img=cv2.imread(image_names[0])
# distorted_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("Distorted image", distorted_image) 
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 

# undistorted_image=np.zeros(distorted_image.shape)
# for y in range(distorted_image.shape[0]):
#     print("{} of {}".format(y+1, distorted_image.shape[0]))
#     for x in range(distorted_image.shape[1]):
#         v3ps_undistorted=get_undistorted_points(K=K, q=q, v3ps=np.array([[x,y]]))
#         x_,y_=v3ps_undistorted.flatten().astype(int)
#         if (x_ >= 0 and x_ < distorted_image.shape[1]) and (y_ >= 0 and y_ < distorted_image.shape[0]):
#             undistorted_image[y_,x_]=distorted_image[y,x]

# cv2.imwrite(path_calibration_images+'/undistorted_image_00.jpg', undistorted_image)

"""
Realidad aumentada
"""
# En una foto
# img=cv2.imread(image_names[5])
# new_img=draw_3D_cube_on_chessboard(img=img, K=K, q=q, board_dimensions=(9,6), square_size=26.5) 
# cv2.imshow("New image", new_img) 
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 

# En video
camera_path="/dev/video4"
cap=cv2.VideoCapture(camera_path)
while True:
    if cap.isOpened():
        # ret: Booleano
        # frame: Imagen capturada
        ret,frame=cap.read()
        if ret:
            new_img=draw_3D_cube_on_chessboard(img=frame, K=K, q=q, board_dimensions=(9,6), square_size=26.5) 
            if new_img is not None:
                cv2.imshow("New image", new_img) 
            else:
                cv2.imshow("New image", frame) 
            # Espera el tiempo dado para destruir la ventana (en ms)
            key=cv2.waitKey(1)
            if key==ord('q'):
                break  
# Cierra el flujo de captura de video
cap.release()  
# Destruye todas las ventanas abiertas 
cv2.destroyAllWindows()









