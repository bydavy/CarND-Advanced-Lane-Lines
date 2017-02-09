# contains calibration information
class Camera:
    def __init__(self, mtx=None, dist=None):
        self.mtx = mtx
        self.dist = dist
