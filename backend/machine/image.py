import base64
from io import BytesIO
import re

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


from openaiclient import generate_response


classes = [
    "aloevera",
    "banana",
    "bilimbi",
    "cantaloupe",
    "cassava",
    "coconut",
    "corn",
    "cucumber",
    "curcuma",
    "eggplant",
    "galangal",
    "ginger",
    "guava",
    "kale",
    "longbeans",
    "mango",
    "melon",
    "orange",
    "paddy",
    "papaya",
    "peperchili",
    "pineapple",
    "pomelo",
    "shallot",
    "soybeans",
    "spinach",
    "sweetpotatoes",
    "tobacco",
    "waterapple",
    "watermelon",
]


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        # First block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third block (final convolution layer)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self._initialize_weights()
        self.fc1 = nn.Linear(self.fc_input_size, 256)  # Reduced size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 30)  # Output classes

    def _initialize_weights(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            output = F.relu(self.bn1(self.conv1(dummy_input)))
            output = self.pool1(output)
            output = F.relu(self.bn2(self.conv2(output)))
            output = self.pool2(output)
            output = F.relu(self.bn3(self.conv3(output)))
            output = self.pool3(output)
            self.fc_input_size = output.numel()

    def forward(self, input):
        # First block
        output = F.relu(self.bn1(self.conv1(input)))
        output = self.pool1(output)

        # Second block
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool2(output)

        # Third block
        output = F.relu(self.bn3(self.conv3(output)))
        output = self.pool3(output)

        # Flatten and fully connected layers
        output = output.view(-1, self.fc_input_size)
        output = F.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)

        return output


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
    transforms.RandomRotation(degrees=15),  # Randomly rotate images
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


PROMPT = """
I have done an analysis of an image and found that it is a plant: {plant}
The plants that I have checked are: {classes} Generate a short response [70 - 100 words] based on the result of my analisys.
""".replace("{classes}", str(classes))


def predict_image(body:dict):
    base64_image = body.get("image")

    if base64_image is None:
        raise ValueError("Invalid data keys")

    model = Network()

    path = "./machine/models/cnn.pth"

    model.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device('cpu')), strict=False)
    model.eval()

    image_data = re.sub('^data:image/.+;base64,', '', base64_image)
    image_data = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_data)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0)

    # Perform prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, result = torch.max(outputs, 1)  # Get class index with the highest score
        result = result.item()  # Convert tensor to integer

    plant = classes[result]

    response = generate_response(PROMPT.replace("{plant}", plant))

    return {
        "response": response,
        "plant": plant
    }
