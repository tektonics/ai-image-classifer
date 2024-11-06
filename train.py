import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from app.models.classifier import ImageClassifier
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm

def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and transform CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        print("Loading CIFAR-10 dataset...")
        trainset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True,
            download=False,
            transform=transform
        )
        print("Dataset loaded successfully!")
        
        # Configure DataLoader with consistent batch size
        batch_size = 32
        trainloader = DataLoader(
            trainset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=0,
            drop_last=True,  # Drop the last incomplete batch
            pin_memory=False  # Don't use pinned memory since we're on CPU
        )
        print(f"DataLoader configured with batch size: {batch_size}")

    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        sys.exit(1)

    # Initialize model, loss function, and optimizer
    model = ImageClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    print(f"Training on {len(trainset)} images")
    
    # Training loop
    num_epochs = 10
    try:
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Create progress bar
            pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for inputs, targets in pbar:
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{running_loss/total:.3f}',
                    'acc': f'{100.*correct/total:.1f}%'
                })
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1} Summary:')
            print(f'Average Loss: {running_loss/len(trainloader):.3f}')
            print(f'Accuracy: {100.*correct/total:.1f}%\n')

        print('Finished Training')

        # Save the final model weights
        torch.save(model.state_dict(), 'model_weights.pth')
        print("Model saved successfully!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    train_model()
