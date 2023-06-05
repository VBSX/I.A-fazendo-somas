import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Preparar os dados de treinamento
input_data = np.array([[2, 3], [4, 1], [5, 6], [7, 2],[8, 4], [9, 5],[10,10]], dtype=np.float32)
target_data = np.array([5, 5, 11, 9, 12, 14, 20], dtype=np.float32)

# Converter os dados para tensores do PyTorch
inputs = torch.from_numpy(input_data)
targets = torch.from_numpy(target_data)

# Definir a arquitetura do modelo
model = nn.Linear(2, 1)  # Uma camada linear com 2 entradas e 1 saída

# Definir a função de perda e o otimizador
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


# Variáveis para acompanhar a melhor geração
best_loss = float('inf')
best_generation = None
# Lista para armazenar as perdas de cada geração
losses = []
# Loop de treinamento
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs.squeeze(), targets)
    
    # Verificar se é a melhor geração até agora
    if loss < best_loss:
        best_loss = loss
        best_generation = model.state_dict().copy()  # Salvar os pesos do modelo

    # Backward pass e otimização
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Salvar a perda da geração atual
    losses.append(loss.item())
    
    # Imprimir a perda a cada 100 épocas
    if (epoch + 1) % 100 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')
        
# Salvar o modelo da melhor geração
torch.save(best_generation, 'modelo.pt')

# Carregar o modelo da melhor geração
model.load_state_dict(torch.load('modelo.pt'))

# Fazer previsões em novos exemplos
test_input = torch.tensor([[3, 4], [6, 2], [7, 4], [10,11], [1,1],[325,635]], dtype=torch.float32)
predicted = model(test_input)
print("Predictions:")
for i in range(len(test_input)):
    print(f"Input: {test_input[i].numpy()}, Predicted Sum: {predicted[i].item()}")


# Plotar o gráfico das perdas por geração
plt.plot(range(1, num_epochs + 1), losses)
plt.xlabel('Época')
plt.ylabel('Perda')
plt.title('Evolução da perda durante o treinamento')
plt.show()
