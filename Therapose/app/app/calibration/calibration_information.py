from __future__ import annotations

import numpy as np
import os

from general.template_copy import TemplateCopy

global_path_app=os.path.dirname(os.path.abspath('__file__'))
global_path_app_calibration_files=os.path.join(global_path_app, "calibration/files")

class CalibrationInformation(TemplateCopy):
    def __init__(self, board_dimensions, square_size, number_images, timer_time, K, q, Qs, cost, jac_norm):
        TemplateCopy.__init__(self)
        self.board_dimensions=board_dimensions          # Tupla  
        self.square_size=square_size                    # Escalar
        self.number_images=number_images                # Escalar
        self.timer_time=timer_time                      # Escalar
        self.K=K                                        # (3,3)
        self.q=q                                        # (5,1)
        self.Qs=Qs                                      # (<number_images>,3,4)
        self.cost=cost                                  # Escalar
        self.jac_norm=jac_norm                          # Escalar    

    def save(self, camera_name):
        if camera_name != "":
            path=global_path_app_calibration_files+"/"+camera_name
            if not os.path.exists(path):
                os.mkdir(path)
            np.save("{}/board_dimensions.npy".format(path), self.board_dimensions)
            np.save("{}/square_size.npy".format(path), self.square_size)
            np.save("{}/number_images.npy".format(path), self.number_images)
            np.save("{}/timer_time.npy".format(path), self.timer_time)
            np.save("{}/K.npy".format(path), self.K)
            np.save("{}/q.npy".format(path), self.q)
            np.save("{}/Qs.npy".format(path), self.Qs)
            np.save("{}/cost.npy".format(path), self.cost)
            np.save("{}/jac_norm.npy".format(path), self.jac_norm)

    @staticmethod
    def load(camera_name):
        if camera_name != "":
            path=global_path_app_calibration_files+"/"+camera_name
            if os.path.exists(path):
                board_dimensions=np.load("{}/board_dimensions.npy".format(path))
                square_size=np.load("{}/square_size.npy".format(path))
                number_images=np.load("{}/number_images.npy".format(path))
                timer_time=np.load("{}/timer_time.npy".format(path))
                K=np.load("{}/K.npy".format(path))
                q=np.load("{}/q.npy".format(path))
                Qs=np.load("{}/Qs.npy".format(path))
                cost=np.load("{}/cost.npy".format(path))
                jac_norm=np.load("{}/jac_norm.npy".format(path))
                
                return CalibrationInformation(board_dimensions=board_dimensions, square_size=square_size, number_images=number_images, timer_time=timer_time, K=K, q=q, Qs=Qs, cost=cost, jac_norm=jac_norm)
        return None
        
    
