from roboflow import Roboflow

rf = Roboflow(api_key="v1WILCzEjTSxMlZ4WBbd")

project = rf.workspace("ahmat-bachir").project("chicken-xynhh")
dataset = project.version(1).download("yolov8")