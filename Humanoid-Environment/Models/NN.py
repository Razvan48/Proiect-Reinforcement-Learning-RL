
import torch




class NN(torch.nn.Module):



    def __init__(self, INPUT_DIM, OUTPUT_DIM):
        super(NN, self).__init__()

        self.INPUT_DIM = INPUT_DIM
        self.OUTPUT_DIM = OUTPUT_DIM

        self.fc1 = torch.nn.Linear(self.INPUT_DIM, 2 * self.INPUT_DIM)
        self.fc2 = torch.nn.Linear(2 * self.INPUT_DIM, 2 * self.INPUT_DIM)
        self.fc3 = torch.nn.Linear(2 * self.INPUT_DIM, 2 * self.INPUT_DIM)
        self.fc4 = torch.nn.Linear(2 * self.INPUT_DIM, self.OUTPUT_DIM)


        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()



    def forward(self, x):
        self.eval() # Seteaza modelul in modul de evaluare
        x0 = torch.tensor(x, dtype=torch.float32)
        x1 = torch.relu(self.fc1(x0))
        x2 = torch.relu(self.fc2(x1))
        x3 = torch.relu(self.fc3(x2))
        x4 = self.fc4(x3)
        return x4



    def fit(self, x, y, NUM_ITERATIONS=100):
        self.train() # Seteaza modelul in modul de antrenare
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        minLoss = float('inf')
        MAXIMUM_PATIENCE = 10
        currentPatience = 0
        LOSS_DELTA = 0.01

        for iteration in range(NUM_ITERATIONS):
            self.optimizer.zero_grad()
            predictions = self.forward(x)
            loss = self.criterion(predictions, y)
            loss.backward()

            self.optimizer.step() # Modificare ponderi

            if loss.item() > minLoss + LOSS_DELTA:
                currentPatience += 1
                if currentPatience == MAXIMUM_PATIENCE:
                    break
            else:
                currentPatience = 0

            minLoss = min(minLoss, loss.item())

            # Afisare loss
            # print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Loss: {loss.item():.4f}")



    def clone(self):
        clone = NN(self.INPUT_DIM, self.OUTPUT_DIM)
        clone.load_state_dict(self.state_dict())
        return clone



    def save_model(self, directory_path : str, file_name : str):
        torch.save(self.state_dict(), directory_path + file_name)



    def load_model(self, directory_path : str, file_name : str):
        self.load_state_dict(torch.load(directory_path + file_name))


