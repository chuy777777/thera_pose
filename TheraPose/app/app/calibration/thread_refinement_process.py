import numpy as np
from threading import Thread

from calibration.calibration_functions import CalibrationFunctions

class ThreadRefinementProcess(Thread):
    def __init__(self, v3ps_real_list, board_dimensions, square_size, **kwargs):
        Thread.__init__(self, **kwargs)
        self.v3ps_real_list=v3ps_real_list
        self.board_dimensions=board_dimensions
        self.square_size=square_size
        self.exception_message=""
        self.refinement_process={}
        self.is_ok=False

    def run(self):
        self.loop_refinement_process()
        print("End ThreadRefinementProcess")    

    def loop_refinement_process(self):
        try:
            K, q, Qs, cost, jac_norm=CalibrationFunctions.refinement_process(v3ps_real_list=self.v3ps_real_list, board_dimensions=self.board_dimensions, square_size=self.square_size)
            self.refinement_process={
                "K": K,
                "q": q,
                "Qs": Qs,
                "cost": cost,
                "jac_norm": jac_norm
            }
            self.is_ok=True
        except np.linalg.LinAlgError as error:
            self.exception_message="(LinAlgError)\n\nHa ocurrido un error en el proceso de refinamiento de los parametros de la camara:\n\n'{}'".format(error)
        except Exception as error:
            self.exception_message="(Exception)\n\nHa ocurrido un error en el proceso de refinamiento de los parametros de la camara:\n\n'{}'".format(error)
        