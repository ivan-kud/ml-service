import base64
from enum import Enum
import io

from PIL import Image
import PIL.ImageOps
import torch
from torch import nn
import torchvision


IMG_WIDTH, IMG_HEIGHT = 28, 28
CLASSES = 10
CHANNELS = 1
DATA_MEAN, DATA_STD = 0.13, 0.31
DATA_SUBZERO = (0 - DATA_MEAN) / DATA_STD
MODEL_PATH = 'ml-models/'


class LogRegModel(nn.Module):
    """Logistic Regression model"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(IMG_HEIGHT * IMG_WIDTH, CLASSES)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear(x)
        return logits


class Dense3Model(nn.Module):
    """Dense model with 3 fully connected layers"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(IMG_HEIGHT * IMG_WIDTH, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, CLASSES)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits


class Conv3Model(nn.Module):
    """Convolutional model with 2 conv and 1 FC layers"""
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            # Conv 1 layer
            nn.ConstantPad2d(1, DATA_SUBZERO),
            nn.Conv2d(CHANNELS, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(4),

            # Conv 2 layer
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64 * 7 * 7, CLASSES)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits


class Conv5Model(nn.Module):
    """Convolutional model with 4 conv and 1 FC layers"""
    def __init__(self, ch=(8, 16, 32, 64)):
        super().__init__()
        self.conv_stack = nn.Sequential(
            # Conv 1 layer
            nn.ConstantPad2d(1, DATA_SUBZERO),
            nn.Conv2d(CHANNELS, ch[0], 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv 2 layer
            nn.Conv2d(ch[0], ch[1], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv 3 layer
            nn.Conv2d(ch[1], ch[2], 3, padding=1),
            nn.ReLU(),
            nn.ConstantPad2d((0, 1, 0, 1), 0),
            nn.MaxPool2d(2),

            # Conv 4 layer
            nn.Conv2d(ch[2], ch[3], 4),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(ch[3], CLASSES)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits


class ModelName(str, Enum):
    logreg = 'logreg'
    dense3 = 'dense3'
    conv3 = 'conv3'
    conv5 = 'conv5'
    all = 'all'


# Load models
models = {ModelName.logreg: LogRegModel(),
          ModelName.dense3: Dense3Model(),
          ModelName.conv3: Conv3Model(),
          ModelName.conv5: Conv5Model((32, 32, 32, 64))}
for name, model in models.items():
    path = MODEL_PATH + 'digit_' + name + '.pt'
    model.load_state_dict(torch.load(path))


def preprocess_image(img: str) -> torch.Tensor:
    # Convert to PIL image (RGBA)
    image_base64 = img.split(';base64,')[1]
    image_bytes = base64.b64decode(image_base64)

    # Convert to PIL, convert to grayscale, invert
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = pil_image.convert('L')
    pil_image = PIL.ImageOps.invert(pil_image)

    # Transform to Tensor, normalize, add dimension
    image_tensor = torchvision.transforms.functional.to_tensor(pil_image)
    image_tensor = (image_tensor - DATA_MEAN) / DATA_STD
    image_tensor = torch.unsqueeze(image_tensor, dim=0)

    return image_tensor


def predict(model: nn.Module, image: torch.Tensor) -> tuple[float, int]:
    model.eval()
    with torch.no_grad():
        probabilities = nn.Softmax(dim=1)(model(image))[0]
        proba = probabilities.max().item()
        label = probabilities.argmax().item()

    return proba, label


def get_response_data(model_name: ModelName, image: str) -> dict:
    # Preprocess image to use it as model input
    image_tensor = preprocess_image(image)

    # Predict
    result = {}
    for name, model in models.items():
        if model_name is ModelName.all or model_name == name:
            result[name] = predict(model, image_tensor)

    # Form result as strings
    if len(result) == 1:
        result_str_1 = f'{result[model_name][1]}'
        result_str_2 = f'{100*result[model_name][0]:.2f} %'
    else:
        result_str_1 = '; '.join([f'{name} - {value[1]}      '
                                  for name, value in result.items()])
        result_str_2 = '; '.join([f'{name} - {100*value[0]:.2f} %'
                                  for name, value in result.items()])

    return {
        'model_name': model_name,
        'image': image,
        'output1': 'Label: ' + result_str_1,
        'output2': 'Confidence: ' + result_str_2,
    }
