# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="828" height="830" alt="image" src="https://github.com/user-attachments/assets/05f5086b-8689-47cb-af02-be363231bfa6" />




## DESIGN STEPS
### STEP 1: 
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

### STEP 2:
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.

### STEP 3:
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.

### STEP 4:
Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### STEP 5:
Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

### STEP 6:
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.


## PROGRAM

### Name: SANTHIYA R

### Register Number: 212223230192

```python
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        #Include your code here
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4)

    def forward(self, x):
      #Include your code here
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x
        
# Initialize model
model = PeopleClassifier(input_size = X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
  #Include your code here
  model.train()
  for epoch in range(epochs):
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

```

### Dataset Information
<img width="1169" height="242" alt="Screenshot 2026-05-11 092943" src="https://github.com/user-attachments/assets/a75a1dbc-909a-46ac-bd4e-64eeca9c9caa" />



### OUTPUT

## Confusion Matrix
<img width="616" height="522" alt="Screenshot 2026-05-11 093146" src="https://github.com/user-attachments/assets/a0d60d82-5fa8-4247-b97c-1915a9aca688" />



## Classification Report
<img width="655" height="395" alt="Screenshot 2026-05-11 093302" src="https://github.com/user-attachments/assets/37d5d0d3-fa31-4e6f-8b3d-007d09b480ef" />



### New Sample Data Prediction
<img width="481" height="96" alt="Screenshot 2026-05-11 093338" src="https://github.com/user-attachments/assets/1d22ab83-2c15-42da-aef1-689275cff436" />



## RESULT
A neural network classification model was successfully developed and tested on the given dataset with satisfactory classification performance.
