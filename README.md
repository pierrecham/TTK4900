# TTK4900
Code used for my Master thesis in 2019.

## Detection
The "detection"-folder contains a Google Colab Notebook that can be opened in Google Colab. First, the Notebook downloads the trained models of the detection networks from a public Google Drive folder. This includes configuration files, data files and trained weights files. Then, the Notebook clones darknet from a Github repository, moves the models of the detection network to darknet and makes darknet. Finally, the Notebook tests the models on an image and on a video.

## Segmentation
The "segmentation"-folder contains two python program files, "main.py" and "segmentation.py": 
- "main.py" reads images from a "images"-folder, segments them and draws proposed bounding boxes on them before saving the images to a "results"-folder. 
- "segmentation.py" contains the functions used by "main.py" to segment and detect the strawberries in each image.

## Labeling
The "labeling"-folder contains three python program files, "main.py", "segmentation.py" and "utilities.py": 
- "main.py" first reads images from a "images"-folder, segments them and draws proposed bounding boxes on them. Then, it waits for the user to decide which bounding boxes to save and discard with keyboard inputs before letting the user draw additional bounding boxes on the image. Finally, "main.py" saves the bounding boxes in YOLO-format in a "labels"-folder, one labels text file per image. The filename and path of every image is saved in "images.txt". 
- "segmentation.py" is the same as in the "segmentation"-folder. 
- "utilities.py" contains the functions used by "main.py" to receive user input and save bounding boxes to files.
