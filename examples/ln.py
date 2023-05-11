import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(1, 1, bias=False)
        self.layer2 = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# Initialize the neural network and set the weight decay
net = SimpleNet()
weight_decay = 0.1

# Use the Mean Squared Error (MSE) loss function
criterion = nn.MSELoss()

# Use Stochastic Gradient Descent (SGD) optimizer without weight decay
optimizer = optim.SGD(net.parameters(), lr=0.001)

# Train the network for a number of epochs
num_epochs = 10000

param_log_1 = []
param_log_2 = []
loss_log = []

param_log_1.append(net.layer1.weight.detach().numpy().item())
param_log_2.append(net.layer2.weight.detach().numpy().item())

for epoch in range(num_epochs):
    # Forward pass
    input_tensor = torch.tensor([[1.0]])
    output = net(input_tensor)

    # Calculate the loss
    loss = criterion(output, input_tensor)

    # Add the weight decay for the second layer (middle layer to output layer)
    layer2_weights = net.layer2.weight
    loss += weight_decay * torch.sum(layer2_weights ** 2) / 2

    # Zero the gradients
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update the weights
    optimizer.step()

    param_log_1.append(net.layer1.weight.detach().numpy().item())
    param_log_2.append(net.layer2.weight.detach().numpy().item())
    loss_log.append(loss.item())

    # Print the loss every 100 epochs
    params = [x for x in net.parameters()]
    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')
        print(param_log_1[-1], param_log_2[-1], params)

fig, [ax1, ax2] = plt.subplots(nrows=2)
ax1.plot(np.array(param_log_1), label="param_in")
ax1.plot(np.array(param_log_2), label="param_out")
ax2.plot(loss_log, label="loss")
ax1.legend()
ax2.legend()
ax2.set_xlabel("Epoch")
plt.savefig("params")

# Test the trained network with input 1
test_input = torch.tensor([[1.0]])
output = net(test_input)
print("Output:", output.item())