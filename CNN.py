import torch
import os 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random

class DatasetMNIST:
    def __init__(self):
        self.pasta_de_dados = './data'

        if not os.path.exists(self.pasta_de_dados):
            print("Pasta ./data não encontrada. Baixando o dataset MNIST...")
        
        self.transformacao = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.setDeTreino = datasets.MNIST(root=self.pasta_de_dados, train=True, download=True, transform=self.transformacao)
        self.setDeTeste = datasets.MNIST(root=self.pasta_de_dados, train=False, download=True, transform=self.transformacao)

        self.carregadorDeTreino = DataLoader(self.setDeTreino, batch_size=64, shuffle=True)
        self.carregadorDeTeste = DataLoader(self.setDeTeste, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        
        x = torch.flatten(x, 1)
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DatasetMNIST()
    modelo = CNN().to(device)
    funcaoDePerda = nn.CrossEntropyLoss()
    otimizador = optim.Adam(modelo.parameters(), lr=0.001)
    numeroDeEpocas = 20

    acertos = [0] * 10
    total = [0] * 10
    previsoes = []

    fig, eixos = plt.subplots(1, 5, figsize=(10, 3))

    print("Iniciando treinamento ...\n")
    

    for epoca in range(numeroDeEpocas):
        perdaAcumulada = 0.0

        for entradas, rotulos in dataset.carregadorDeTreino:
            otimizador.zero_grad()
            saidas = modelo(entradas)
            perda = funcaoDePerda(saidas, rotulos)
            perda.backward()
            otimizador.step()
            
            perdaAcumulada += perda.item()
        
        print(f"Época {epoca+1}, Perda: {perdaAcumulada/len(dataset.carregadorDeTreino)}")

    with torch.no_grad():
        for entradas, rotulos in dataset.carregadorDeTeste:
            saidas = modelo(entradas)
            _, previsto = torch.max(saidas, 1)

            for i in range(rotulos.size(0)):
                rotulo = rotulos[i].item()
                rotuloPrevisto = previsto[i].item()

                if rotulo == rotuloPrevisto:
                    acertos[rotulo] += 1

                total[rotulo] += 1
                previsoes.append((rotulo, rotuloPrevisto))

    print("\nPrecisões por número:\n")
    for i in range(10):
        precisao = 100 * acertos[i] / total[i]
        print(f"Precisão pro {i}: {precisao:.2f}%")

    for i in range(5):
        indiceAleatorio = random.randint(0, len(dataset.setDeTeste) - 1)
        imagem, rotulo = dataset.setDeTeste[indiceAleatorio]
        eixos[i].imshow(imagem.squeeze(), cmap='gray')
        imagem = imagem.unsqueeze(0) #?

        with torch.no_grad():
            saidas = modelo(imagem)
            _, previsto = torch.max(saidas, 1)

        previsao = previsto.item()
        
        eixos[i].set_title(f"Nº {rotulo}, CNN: {previsao}")
        eixos[i].axis('off') 

    plt.show()
    

if __name__ == "__main__":
    main()
