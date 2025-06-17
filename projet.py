import glob
from PIL import Image
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms,models
from torchvision.datasets import ImageFolder

basePath = r"C:\Users\chadi\Desktop\Kagglehub" #chemin ou se trouve les données à changer avec votre chemin de données
device = 'cuda' if torch.cuda.is_available() else 'cpu' #pour choisir travailler sur le processeur ou la carte graphique s'il existe
print("device:",device)
df = pd.read_csv(basePath + r'\train-scene classification\train.csv') #lire les données a partir d'un fichier excel
print(df.head())
rename={
    0:"Buildings",
    1:"Forests",
    2:"Montains",
    3:"Glacier",
    4:"Street",
    5:"Sea"
} #liste des noms de nouveaux labels
df['label']=df['label'].map(rename)  #pour renommer les labels
class_counts= df['label'].value_counts() #calculer le nombre d'image par classe
print(class_counts)
classes = df['label'].unique()
data_by_class={}
for cls in classes:
    data_by_class[cls] = df[df['label']==cls]
print(data_by_class["Buildings"].head())
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) #application des filtres pour faire l'equivalence avec resnet18
train_dataset = ImageFolder(root=basePath+r'\train-scene classification\sorted_images', transform=transform)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size]) #diviser les données en train et validation
# Créer les DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
# Afficher les classes
classes = train_dataset.classes
print(f"Classes : {classes}")
model = models.resnet18(pretrained=True) #appel au model resnet18 pretrained
# Modifier la dernière couche pour correspondre au nombre de classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(classes))

# Déplacer le modèle sur le dispositif GPU/CPU
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Réinitialiser les gradients
        outputs = model(images)  # Passage avant
        loss = criterion(outputs, labels)  # Calcul de la perte
        loss.backward()  # Backpropagation
        optimizer.step()  # Mise à jour des poids

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Validation Accuracy: {100 * correct / total:.2f}%")
predicted_images={}
for filename in glob.glob(r'C:\Users\chadi\Desktop\Kagglehub\train-scene classification\train\*.jpg'):
    img_arr = Image.open(filename)
# Pass the image for preprocessing and the image preprocessed
    img_preprocessed = transform(img_arr)
# Reshape, crop, and normalize the input tensor for feeding into network for evaluation
    batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)
    model.eval()
    out = model(batch_img_tensor)
    _, predicted = torch.max(out, 1)
    """
    logits = torch.tensor(out)
    probabilities = F.softmax(logits, dim=1)
    print(probabilities)
    predicted_class = torch.argmax(logits, dim=1).item()
    print(predicted_class)
    print(filename)
    """
    predicted_images[filename[66:]]=predicted

    if filename==r"C:\Users\chadi\Desktop\Kagglehub\train-scene classification\train\9993.jpg":
        break
label=0
correct = 0
i=0
for cle,valeur in predicted_images.items():
    if cle in df["image_name"].values:
        label_value = df.loc[df["image_name"] == cle, "label"].values[0]
        match label_value:
            case 'Buildings':
                label=0
            case 'Forests':
                label = 1
            case 'Glacier':
                label = 3
            case 'Mountains':
                label = 2
            case 'Sea':
                label = 5
            case 'Street':
                label = 4

    if valeur==label:
        correct+=1
    i += 1
print(i)
print(f"test accuracy: {100 * correct / i:.2f}%")