import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

# Define the customized network
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(500, 21)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc1(torch.flatten(x, 1)))

        return x

# Define the customized LeNet-5 network
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 21)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def imshow(img):
    """
    Visualize the given image 

    Parameters
    ----------
    `img`: image to be visualized

    Returns
    -------
    `None`: 
    """
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    return

def train(model, model_path, device, data_path, data_transform, batch_size, n_epochs, lr, classes):
    """
    Train the given model with the given parameters and dataset

    Parameters
    ----------
    `model`: network model to be trained
    `model_path`: path to save the model weights after the training
    `device`: device to be used for training
    `data_path`: path to the dataset location
    `data_transform`: defined transforms to be conducted on the input images
    `batch_size`: training batch size
    `n_epochs`: number of epochs to train
    `lr`: learning rate
    `classes`: list of class label names

    Returns
    -------
    `None`: 
    """
    trainset = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=data_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    # # Visualize some random training images
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # # Show the images and their labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    # imshow(torchvision.utils.make_grid(images))

    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_log = []
    writer = SummaryWriter()
    # Train the model for a number of epochs
    for epoch in range(n_epochs):
        running_loss = 0.0
        count = 0
        for i, data in enumerate(trainloader, 0):
            # Load a batch of training images
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            # Feed this batch into the model and predict
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Update loss count
            count += 1
            running_loss += loss.item()

        # Record the loss for each epoch trained
        loss_log.append(running_loss/count)
        print('Epoch [%d] loss: %.3f' % (epoch + 1, loss_log[-1]))
        # Plot the loss curve
        writer.add_scalar('Loss/train', running_loss/count, epoch)

    writer.close()
    print('Finished Training')

    # Save model weights
    torch.save(model.state_dict(), model_path)
    return

def test(model, model_path, device, data_path, data_transform, batch_size, classes):
    """
    Test the given model with the given weights and dataset

    Parameters
    ----------
    `model`: network model to be trained
    `model_path`: path to load the model weights from
    `device`: device to be used for testing
    `data_path`: path to the dataset location
    `data_transform`: defined transforms to be conducted on the input images
    `batch_size`: testing batch size
    `classes`: list of class label names

    Returns
    -------
    `None`: 
    """
    testset = datasets.ImageFolder(root=os.path.join(data_path, 'test'), transform=data_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Load the trained model
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    
    # Testing on some random testset images
    dataiter = iter(testloader)
    data = dataiter.next()
    images, labels = data[0].to(device), data[1].to(device)
    # Load this batch into the model and predict
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Visualize the images and their labels
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))
    imshow(torchvision.utils.make_grid(images))

    # Compute the overall and class-wise error rates on the testset
    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    with torch.no_grad():
        # Load each batch into the model and predict
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # Collect overall results
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Collect the correct predictions for each class
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                       accuracy))

    print('Error rate of the network on the %d test images: %.4f %%' % (total, 100*(total - correct)/total))

    return

def main():
    # Set destination paths
    repo_path = pathlib.Path(__file__).parent.parent.resolve()
    data_path = os.path.join(repo_path, 'data/cnn')
    model_path = os.path.join(repo_path, 'model/cnn.pth')

    # Define the device to be used for training and testing
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # Collect class names
    classes = [x for x in os.listdir(os.path.join(data_path, 'train')) if x!='.DS_Store']
    print(classes)
    # Define data argmentation processes
    data_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5), (0.5)),
    ])

    # Hyperparameters for training
    training    = True
    batch_size  = 256
    n_epochs    = 1000
    lr          = 3e-4
    
    # Define model
    # model = SimpleCNN()
    model = LeNet5()

    # Training Process
    if training:
        train(model, model_path, device, data_path, data_transform, batch_size, n_epochs, lr, classes)
    # Testing Process
    test(model, model_path, device, data_path, data_transform, batch_size, classes)

if __name__ == "__main__":
    main()