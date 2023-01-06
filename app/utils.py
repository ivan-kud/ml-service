import base64
import io

from PIL import Image
import PIL.ImageOps
import torch
from torch import nn
import torchvision


DIGIT_WIDTH, DIGIT_HEIGHT = 28, 28
DIGIT_CLASSES = 10
DIGIT_CHANNELS = 1
DIGIT_MEAN, DIGIT_STD = 0.13, 0.31
DIGIT_SUBZERO = (0 - DIGIT_MEAN) / DIGIT_STD
MODEL_PATH = 'app/models/'


class LogRegDigitModel(nn.Module):
    """Logistic Regression model"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(DIGIT_HEIGHT * DIGIT_WIDTH, DIGIT_CLASSES)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear(x)
        return logits


class FullyConn3DigitModel(nn.Module):
    """Fully Connected model with 3 layers"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(DIGIT_HEIGHT * DIGIT_WIDTH, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, DIGIT_CLASSES)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits


class Convl2DigitModel(nn.Module):
    """Convolutional model with 2 conv layers"""
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            # Conv 1 layer
            nn.ConstantPad2d(1, DIGIT_SUBZERO),
            nn.Conv2d(DIGIT_CHANNELS, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(4),

            # Conv 2 layer
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64 * 7 * 7, DIGIT_CLASSES)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits


class Convl4DigitModel(nn.Module):
    """Convolutional model with 4 conv layers"""
    def __init__(self, ch=(8, 16, 32, 64)):
        super().__init__()
        self.conv_stack = nn.Sequential(
            # Conv 1 layer
            nn.ConstantPad2d(1, DIGIT_SUBZERO),
            nn.Conv2d(DIGIT_CHANNELS, ch[0], 3),
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
        self.linear = nn.Linear(ch[3], DIGIT_CLASSES)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits


# Load models
digit_models = {'logreg': LogRegDigitModel(),
                'fconn3': FullyConn3DigitModel(),
                'convl2fconn1': Convl2DigitModel(),
                'convl4fconn1': Convl4DigitModel((32, 32, 32, 64))}
for name in digit_models:
    path = MODEL_PATH + 'digit_' + name + '.pt'
    digit_models[name].load_state_dict(torch.load(path))


def preprocess_digit(img: str):
    # Convert to PIL image (RGBA)
    image_base64 = img.split(';base64,')[1]
    image_bytes = base64.b64decode(image_base64)

    # Convert to PIL, remove alpha, convert to grayscale, resize, invert
    pil_image = Image.open(io.BytesIO(image_bytes))
    background = Image.new('RGBA', pil_image.size, (255, 255, 255, 255))
    pil_image = Image.alpha_composite(background, pil_image)
    pil_image = pil_image.convert('L')
    pil_image = pil_image.resize((DIGIT_WIDTH, DIGIT_HEIGHT))
    pil_image = PIL.ImageOps.invert(pil_image)

    # Transform to Tensor, normalize, add dimension
    image_tensor = torchvision.transforms.functional.to_tensor(pil_image)
    image_tensor = (image_tensor - DIGIT_MEAN) / DIGIT_STD
    image_tensor = torch.unsqueeze(image_tensor, dim=0)

    return image_tensor


def predict_digit(model: nn.Module, image: torch.Tensor):
    model.eval()
    with torch.no_grad():
        probabilities = nn.Softmax(dim=1)(model(image))[0]
        proba = probabilities.max().item()
        label = probabilities.argmax().item()

    return proba, label
