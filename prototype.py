import numpy as np

bbox = np.array([0.550969, 0.852614, 0.117573, 0.0611791])
       
b_width = bbox[2,]
b_height = bbox[3,]
x_max = bbox[0,] + b_width
y_max = bbox[1,] + b_height

bbox[0,] = ((x_max + bbox[0,]) / 2)  / 1
bbox[1,] = ((y_max + bbox[1,]) / 2)  / 1
bbox[2,] = bbox[2,] / 1
bbox[3,] = bbox[3,] / 1
        
bobox = bbox[0,], bbox[1,], bbox[2,], bbox[3,] # normalized x_center, y_center, box_width and box_height

print(bobox)