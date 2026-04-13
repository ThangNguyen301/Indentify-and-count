from roboflow import Roboflow

rf = Roboflow(api_key="v1WILCzEjTSxMlZ4WBbd")

project = rf.workspace("sages").project("meat-a9qkz")
dataset = project.version(1).download("yolov8")