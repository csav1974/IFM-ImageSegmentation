# IFM-ImageSegmentation
This repository uses a custom trained ultralytics yolo model to segment images of solarcells captured with a IFM. The model currently finds and marks 3 different kind of defects (Whiskers, Chipping, Scratching).

The model can be used as a stand-alone program (in this case, update constants in inference.py as needed) or by calling `run_image_segmentation()`
