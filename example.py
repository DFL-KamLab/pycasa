import pycasa as pc

self = pc.io.load_default_data()
self.detection.yolo()
self.tracking.sort()
self.motility.kinematic_parameters()
self.motility.casa_parameters()
