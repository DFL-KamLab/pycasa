import pycasa as pc

self = pc.io.load_default_data()
self.detection.yolo()
self.tracking.sort()
self.motility.standard_motility_parameters()
