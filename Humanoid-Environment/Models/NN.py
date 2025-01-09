import torch




class NN(torch.nn.Module):



    def __init__(self, INPUT_DIM, OUTPUT_DIM):
        super(NN, self).__init__()

        self.INPUT_DIM = INPUT_DIM
        self.OUTPUT_DIM = OUTPUT_DIM

        self.fc1 = torch.nn.Linear(self.INPUT_DIM, 2 * self.INPUT_DIM)
        self.fc2 = torch.nn.Linear(2 * self.INPUT_DIM, 2 * self.INPUT_DIM)
        self.fc3 = torch.nn.Linear(2 * self.INPUT_DIM, self.OUTPUT_DIM)


        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()



    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



    def fit(self, x, y, NUM_ITERATIONS=10):

        for iteration in range(NUM_ITERATIONS):
            self.train() # Seteaza modelul in modul de antrenare

            self.optimizer.zero_grad()
            predictions = self.forward(x)
            loss = self.criterion(predictions, y)
            loss.backward()

            self.optimizer.step() # Modificare ponderi

            # Afisare loss
            print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Loss: {loss.item():.4f}")



    def clone(self):
        clone = NN(self.INPUT_DIM, self.OUTPUT_DIM)
        clone.load_state_dict(self.state_dict())
        return clone



    def save_model(self, directory_path : str, file_name : str):
        torch.save(self.state_dict(), directory_path + file_name)



    def load_model(self, directory_path : str, file_name : str):
        self.load_state_dict(torch.load(directory_path + file_name))


