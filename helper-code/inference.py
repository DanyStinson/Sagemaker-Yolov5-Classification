import argparse
import os
import platform
import sys
import torch
import json
import numpy as np
import cv2
import torch.nn.functional as F



def model_fn(model_dir):

    os.system("pip install dill")
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    model = torch.hub.load("ultralytics/yolov5", "custom", path="/opt/ml/model/exp/weights/best.pt", force_reload=True)
    print("Model Loaded")
    return model


def input_fn(input_data, content_type):

    if content_type in ['image/png', 'image/jpeg']:
        img = np.frombuffer(input_data, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)[..., ::-1]
        img = cv2.resize(img, [244,244])
        img = img / 255.0  # Scale pixel values to the range [0, 1]
        img = np.transpose(img, (2, 0, 1))  # Change the shape from HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        # Convert the NumPy array to a PyTorch tensor
        img_tensor = torch.tensor(img, dtype=torch.float32)
        return img_tensor
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return


def predict_fn(input_data, model):
    print("Making inference")
    results = model(input_data)
    probabilities = F.softmax(results, dim=1)
    print("Res", results)
    print("Probs", probabilities)
    # Convert the PyTorch tensor to a NumPy array
    numpy_array = probabilities.numpy()
    # Convert the NumPy array to a JSON string
    json_data = json.dumps(numpy_array.tolist())
    return json_data