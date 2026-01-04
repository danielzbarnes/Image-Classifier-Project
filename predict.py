# imports
import argparse # For command line arguments
import torch
import json
from torchvision import models, transforms # Pre-trained models and transformations
from PIL import Image # For image processing

print("Pytorch Version: %s" % (torch.__version__))
print("Cuda is available: %s" % (torch.cuda.is_available()))

# prompt for command line arguments
parser = argparse.ArgumentParser() # Initialize the parser
parser.add_argument('--image_path', type=str, help='Path to images') # Image path
parser.add_argument('--checkpoint', type=str, help='Path to checkpoint') # Checkpoint path
parser.add_argument('--top_k', type=int, default=5, help='Top K Classes') # Top K classes
parser.add_argument('--category_name', type=str, help='JSON file for category names') # Category names
parser.add_argument('--gpu', action='store_true', help='use GPU, if available') # Use GPU if available
args = parser.parse_args()

def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    image = Image.open(image)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transform(image)

def predict(image_path, model, topk=5, device=any):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.eval() # Set model to evaluation mode
    model.to(device) # Move model to device
    
    image = process_image(image_path) # Process the image
    image = image.unsqueeze(0) # Add batch dimension
    image = image.to(device) # Move image to device
    
    # Disable gradient calculation
    with torch.no_grad(): 
        output = model(image)
    
    ps = torch.exp(output) # Get probabilities
    top_p, top_class = ps.topk(topk, dim=1) # Get top K probabilities and classes
    
    # Convert to numpy array
    top_p = top_p.to('cpu').numpy()[0]
    top_class = top_class.to('cpu').numpy()[0]
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()} # Invert class_to_idx
    top_classes = [idx_to_class[i] for i in top_class] # Map indices to classes
    
    return top_p, top_classes

print("Loading checkpoint from %s" % args.checkpoint)
model = load_checkpoint(args.checkpoint)

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
print("Predicting on device: %s" % device)

# Make predictions
probs, classes = predict(args.image_path, model, args.top_k, device)

# If category names provided, map classes to names
if args.category_name:
    import json
    with open(args.category_name, 'r') as f:
        cat_to_name = json.load(f)
    classes = [cat_to_name[c] for c in classes]
else:
    classes = classes

print(f"\nTop {args.top_k} Predictions for image '{args.image_path}':")

for i in range(len(classes)):
    print(f"{i+1}: Class = {classes[i]}, Probability = {probs[i]:.4f}")
