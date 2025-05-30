
import torch
import torch.nn as nn #Base class for all neural network modules.
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
# A basic convolutional neural network
# the CNN inherits from torch.nn.Module
class CNN(nn.Module):
    
    def __init__(self, num_classes = 10): # there are 10 classes in CIFAR10
        super(CNN, self).__init__() # we super the parent's init (nn.module)
        self.convolutional_layer = nn.Conv2d(in_channels = 3, # 3 = R, G, B
                                             out_channels = 16, # Filters, increase for better accuracy.
                                             kernel_size = 3, # 3x3 square to look at
                                             stride = 1, # how much we move the kernel each time
                                             padding = 1) # padding on the outside of the image
        self.relu = nn.ReLU() # rectified linear unit function
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fully_connected = nn.Linear(16*16*16, num_classes)

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x

#training method for the CNN
def train_CNN():
    batch_size = 32
    threads = 2
    learning_rate = 0.001

    # composition of transforms functions to be used on CIFAR10 for ease of use.
    transform = transforms.Compose([ # creates a sequential pipeline of transformations
        transforms.ToTensor(), # Converts a PIL image or NumPy array to a PyTorch tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalizes each channel R, G, B

    # CIFAR10 DATA SET : TRAINING
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                          download=True, transform=transform)
    #sets up a data loader in PyTorch to feed training data into CNN
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                             shuffle=True, num_workers=threads)

    # CIFAR10 DATA SET : TESTING
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                         download=False, transform=transform)
    # same as trainloader but for test data. 
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                            shuffle=False, num_workers=threads)

    model = CNN()
    loss_function = nn.CrossEntropyLoss()
    # Adaptive Moment Estimation
    adam_opt = optim.Adam(model.parameters(), lr=learning_rate)

    #training loop
    print("Training...")
    losses = []
    for full_pass in range(5):
        total_loss = 0
        for i,(inputs, labels) in enumerate(trainloader):
            adam_opt.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            adam_opt.step()

            total_loss += loss.item()

            if i % 100 == 99: # reset every 10
                print(f'Epoch {full_pass+1}, Batch {i+1}: Loss = {total_loss/100:.4f}')
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch {full_pass+1}: Test Accuracy={accuracy:.2f}%')
        losses.append(total_loss)





    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title('CNN Training Loss')
    plt.xlabel('Large Batch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    
    print("done.")

if __name__ == "__main__":
    train_CNN()



