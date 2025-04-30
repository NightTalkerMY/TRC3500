from distance_model import *
from material_model import *
from height_model import *

dataset_directory = "dataset" 

classifier, data = classify_distance_drops(dataset_directory, retrain=True)
classifier, data = classify_height_drops(dataset_directory, retrain=True)
classifier, data = classify_material_drops(dataset_directory, retrain=True)

print("Classification complete!")