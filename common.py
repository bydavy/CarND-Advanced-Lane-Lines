class Camera:
    """Contains calibration meta-data"""
    def __init__(self, mtx=None, dist=None):
        self.mtx = mtx
        self.dist = dist
