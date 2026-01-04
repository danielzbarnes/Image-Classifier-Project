
#imports
import argparse # For command line arguments
import torch

from torch import nn # Neural network module
from torchvision import datasets, transforms, models # Datasets, transformations, and pre-trained models

print("Pytorch Version: %s" % (torch.__version__))
print("Cuda is available: %s" % (torch.cuda.is_available()))

# prompt for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='flowers', help='Directory of the dataset')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Path to save the checkpoint')
parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture: vgg16 or densenet121')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
args = parser.parse_args()



# Data directories
train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'

# transforms for the training and validation sets
train_transforms = transforms.Compose([
    transforms.RandomRotation(224),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dir_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# Define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_dir_data, batch_size=64)

# Load a pre-trained model, checks for vgg16, vgg13, or densenet121
if (args.arch == 'vgg16'):
    model = models.vgg16(pretrained=True)
    input_size = 25088
elif (args.arch == 'vgg13'):
    model = models.vgg13(pretrained=True)
    input_size = 25088
elif (args.arch == 'densenet121'):
    model = models.densenet121(pretrained=True)
    input_size = 1024
else:
    print("Unsupported architecture. Please choose either 'vgg16', 'vgg13', or 'densenet121'.")
    exit()
    
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
# Define a new, untrained feed-forward classifier as a nn.Sequential model
classifier = nn.Sequential(
    nn.Linear(input_size, args.hidden_units), # input layer
    nn.ReLU(),                                # relu activation
    nn.Dropout(0.2),                          # dropout layer
    nn.Linear(args.hidden_units, 102),        # output layer
    nn.LogSoftmax(dim=1)                      # log softmax activation
)

# assign the classifier to the model
model.classifier = classifier

# Define the loss function and optimizer
criterion = nn.NLLLoss() # Negative log likelihood loss
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.learning_rate) 

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)

print("Training on device: %s" % device)


for epoch in range(args.epochs):
    
    training_loss = 0
    model.train()
    
    # Training loop
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad() # Zero the gradients
        
        logps = model(inputs) # Forward pass
        loss = criterion(logps, labels) # Calculate the loss
        loss.backward() # Backward pass
        optimizer.step() # Update the weights
        
        training_loss += loss.item() # Accumulate the loss
        

    model.eval()
    valid_loss = 0
    accuracy = 0
    
    # Validation loop
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs) # Forward pass
            
            valid_loss += criterion(logps, labels).item() # Calculate the validation loss
            
            ps = torch.exp(logps) # Convert log probabilities to probabilities
            top_p, top_class = ps.topk(1, dim=1) # Get the top class predictions
            equals = top_class == labels.view(*top_class.shape) # Check for correct predictions
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item() # Calculate accuracy
        
        print(f"Epoch {epoch+1}/{args.epochs}.. "
              f"Train loss: {training_loss/len(trainloader):.3f}.. "
              f"Validation loss: {valid_loss/len(validloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(validloader):.3f}")

# Save the checkpoint
model.class_to_idx = train_data.class_to_idx

# Create the checkpoint dictionary
checkpoint = {
    'arch': args.arch,                  # Model architecture
    'classifier': model.classifier,     # Classifier
    'class_to_idx': model.class_to_idx, # Class to index mapping
    'state_dict': model.state_dict()    # Model state dictionary
}

torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
print("Checkpoint saved to %s/checkpoint.pth" % args.save_dir)

