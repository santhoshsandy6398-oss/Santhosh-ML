from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# Define the same TinyCNN as in train.py
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 14 * 14, 32)  # must match train.py
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained TinyCNN model
model = TinyCNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((32, 32)),   # must match train.py
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5])
])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img = Image.open(file).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    classes = ["Cat", "Dog"]
    return jsonify({"class": classes[predicted.item()]})

if __name__ == "__main__":
    import os
    port=int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
