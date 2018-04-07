from preprocessing import *
from frontend import YOLO
import keras

os.environ["CUDA_VISIBLE_DEVICES"] = ""
weights_path = './full_yolo/weights.24-0.05.h5'
model_save_path = './full_yolo/full_yolo.h5'
input_size = 608
backend_name = 'Full Yolo'


yolo = YOLO(backend=backend_name,
            input_size=input_size,
            labels=['person'],
            max_box_per_image=100,
            anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828])
yolo.load_weights(weights_path)
keras.models.save_model(yolo.model, model_save_path)
