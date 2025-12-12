import json
import pandas as pd
from torch.utils.data import Dataset
import pickle
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score
import speech_recognition as sr
import os
import re
import joblib
from werkzeug.utils import secure_filename

from flask import Flask, render_template, flash, redirect, url_for, request, session, jsonify



app = Flask(__name__)
app.secret_key = 'your_secret_key'
# Function to load data from a .jsonl file


@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/index')
def index():
    return render_template('imageprediction.html')



@app.route('/cnn')
def cnn_route():  # Changed function name to avoid conflict
    return render_template('imageprediction.html')

@app.route('/rnn')
def rnn_route():  # Changed function name to avoid conflict
    return render_template('rnnprediction.html')


def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# Load training and testing datasets
train_df = load_jsonl('trainnew.jsonl')
test_df = load_jsonl('test.jsonl')

print("Train Data:")
print(train_df.head())
print("Test Data:")
print(test_df.head())

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing images to 224x224 for CNN
    transforms.ToTensor(),  # Convert image to PyTorch Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalizing with ImageNet statistics
])

# Function to load and preprocess an image
def load_image(img_path):
    # Load the image
    image = Image.open(img_path)

    # Convert to RGB if it has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization for pre-trained models
    ])
    
    # Apply transformations
    image = transform(image)
    return image

class MemeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = self.df.iloc[idx]['label']
        img_path = self.df.iloc[idx]['img']

        # Load and transform image
        image = load_image(img_path)

        return text, image, torch.tensor(label, dtype=torch.float32)

class MultiModalCNNModel(nn.Module):
    def __init__(self):
        super(MultiModalCNNModel, self).__init__()
        
        # Pre-trained CNN for image data (ResNet18)
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 256)  # Final layer outputs 256 features
        
        # Text feature processing (simple, could use embeddings later)
        self.text_fc = nn.Linear(300, 256)  # Assuming 300-dimensional text embeddings

        # Combine the text and image features
        self.fc = nn.Linear(512, 1)  # Combine the 256 image features + 256 text features

    def forward(self, text_features, image):
        # Image features from CNN
        img_features = self.cnn(image)
        
        # Text features (dummy embedding, replace with real embeddings)
        text_features = self.text_fc(text_features)

        # Concatenate image and text features
        combined = torch.cat((img_features, text_features), dim=1)

        # Final classification layer
        out = self.fc(combined)
        return torch.sigmoid(out)

# Initialize the model
model = MultiModalCNNModel()

# Dummy function for text embedding (replace this with GloVe, BERT, etc.)
def extract_text_features(text):
    # For now, we simulate text embeddings with a random 300-dimensional vector
    return torch.randn(300)
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for text_data, images, labels in train_loader:
            # Extract text embeddings
            text_features = torch.stack([extract_text_features(text) for text in text_data])
            text_features = text_features.to(device)
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(text_features, images).squeeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Backpropagation
            loss.backward()
            optimizer.step()

            # Collect predictions and true labels
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy for this epoch
        accuracy = accuracy_score(all_labels, all_preds)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}, Accuracy: {accuracy:.4f}')

# Set up DataLoader
train_loader = DataLoader(MemeDataset(train_df), batch_size=2, shuffle=True)

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=5)

torch.save(model.state_dict(), 'multi_modal_cnn_model_new.pth')
# Save the model
# with open('multi_modal_cnn_model.pkl', 'wb') as f:
#     pickle.dump(model, f)

print("Model saved.")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == 'admin':
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')
# Load the model
class MultiModalCNNModel(nn.Module):
    def __init__(self):
        super(MultiModalCNNModel, self).__init__()
        
        # Pre-trained CNN for image data (ResNet18)
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 256)  # Final layer outputs 256 features
        
        # Text feature processing (assuming 300-dimensional text embeddings)
        self.text_fc = nn.Linear(300, 256)

        # Combine the text and image features
        self.fc = nn.Linear(512, 1)  # Combine the 256 image features + 256 text features

    def forward(self, text_features, image):
        # Image features from CNN
        img_features = self.cnn(image)
        
        # Text features (dummy embedding, replace with real embeddings)
        text_features = self.text_fc(text_features)

        # Concatenate image and text features
        combined = torch.cat((img_features, text_features), dim=1)

        # Final classification layer
        out = self.fc(combined)
        return torch.sigmoid(out)

# Load the trained model's state
model = MultiModalCNNModel()
model.load_state_dict(torch.load('multi_modal_cnn_model.pth'))
model.eval()  # Set to evaluation mode

# Image transformation
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to process the uploaded image and make predictions
def predict_image(image):
    # Transform the image
    image = image_transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(torch.zeros(1, 300), image)  # Placeholder for text features
        prediction = output.item()

    # Return True if prediction > 0.5 (hateful)
    return prediction > 0.5

# Flask route to handle image upload
@app.route('/', methods=['GET', 'POST'])
def cnn_prediction():
    if request.method == 'POST':
        file = request.files['image']  # Get the uploaded image
        image = Image.open(file.stream)  # Open the image

        # Predict whether the image is hateful or not
        is_hateful = predict_image(image)
        return jsonify({'is_hateful': is_hateful})

    return render_template('imageprediction.html')  # Render the HTML template

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"wav"}

modelnew = joblib.load("hate_speech_model.pkl")
# Function to check allowed file type
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to convert audio to text
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)  # Record audio
        try:
            text = recognizer.recognize_google(audio_data)  # Convert speech to text
            return text.lower()
        except sr.UnknownValueError:
            return "Error: Could not understand audio"
        except sr.RequestError:
            return "Error: Speech recognition service unavailable"

# Function to clean text before prediction
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text.lower()

# Function to predict hate speech
def predict_hate_speech(text):
    cleaned_text = clean_text(text)
    prediction = modelnew.predict([cleaned_text])[0]
    categories = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
    return categories[prediction]

# Flask Routes
@app.route("/hatespeech", methods=["GET", "POST"])
def hatespeech():
    if request.method == "POST":
        # Check if file is uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"})

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Convert audio to text
            transcribed_text = transcribe_audio(filepath)

            # If transcription failed
            if "Error" in transcribed_text:
                return jsonify({"error": transcribed_text})

            # Predict hate speech
            prediction = predict_hate_speech(transcribed_text)

            return jsonify({"text": transcribed_text, "prediction": prediction})

    return render_template("hatespeech.html")

if __name__ == '__main__':
    app.run(debug=True)