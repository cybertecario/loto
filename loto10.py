import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit, Aer, execute
import random

# Carregar os dados históricos da Lotofácil
def load_data(file_path):
    data = pd.read_csv(file_path, sep=';', header=0)
    return data.iloc[:, 1:].values  # Ignorar a coluna "Concurso"

# Pré-processamento dos dados
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Criar sequências para o modelo GRU
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Modelo GRU
def build_gru_model(input_shape):
    model = Sequential([
        GRU(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(15, activation='sigmoid')  # 15 números da Lotofácil
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Simulação Quântica
def quantum_simulation(predictions, num_qubits=15):
    circuit = QuantumCircuit(num_qubits)
    for i, prob in enumerate(predictions):
        if random.random() < prob:  # Probabilidade de ativação baseada na saída do GRU
            circuit.x(i)  # Aplicar porta X (ativar o qubit)
    simulator = Aer.get_backend('statevector_simulator')
    result = execute(circuit, simulator).result()
    statevector = result.get_statevector()
    return np.abs(statevector) ** 2  # Probabilidades finais

# GAN Generator
def build_generator(latent_dim):
    model = Sequential([
        Dense(64, input_dim=latent_dim, activation='relu'),
        Dense(128, activation='relu'),
        Dense(15, activation='sigmoid')  # Gerar 15 números
    ])
    return model

# GAN Discriminator
def build_discriminator():
    model = Sequential([
        Dense(128, input_dim=15, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Classificar como real/falso
    ])
    model.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Treinar GAN
def train_gan(generator, discriminator, gan, X_train, epochs=1000, batch_size=32):
    for epoch in range(epochs):
        # Gerar dados falsos
        noise = np.random.normal(0, 1, (batch_size, generator.input_shape[1]))
        fake_data = generator.predict(noise)
        
        # Selecionar dados reais
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_data = X_train[idx]
        
        # Labels
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        # Treinar discriminador
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Treinar gerador
        noise = np.random.normal(0, 1, (batch_size, generator.input_shape[1]))
        g_loss = gan.train_on_batch(noise, real_labels)
        
        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]")

# Função principal
def main():
    # Parâmetros
    file_path = 'base_Lotofacil.csv'
    seq_length = 10
    latent_dim = 100
    epochs = 1000
    batch_size = 32
    
    # Carregar e pré-processar dados
    raw_data = load_data(file_path)
    scaled_data, scaler = preprocess_data(raw_data)
    
    # Criar sequências para o GRU
    X, y = create_sequences(scaled_data, seq_length)
    
    # Dividir em treino e teste
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Construir e treinar o modelo GRU
    gru_model = build_gru_model((X_train.shape[1], X_train.shape[2]))
    gru_model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_data=(X_test, y_test))
    
    # Fazer previsões iniciais com o GRU
    predictions = gru_model.predict(X_test[-1].reshape(1, seq_length, 15))
    
    # Refinar previsões com simulação quântica
    quantum_probs = quantum_simulation(predictions.flatten())
    
    # Construir GAN
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()
    gan = Sequential([generator, discriminator])
    gan.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')
    
    # Treinar GAN
    train_gan(generator, discriminator, gan, X_train, epochs=epochs, batch_size=batch_size)
    
    # Gerar previsão final com GAN
    noise = np.random.normal(0, 1, (1, latent_dim))
    final_prediction = generator.predict(noise)
    
    # Combinar resultados
    combined_prediction = (quantum_probs + final_prediction.flatten()) / 2
    predicted_numbers = np.argsort(combined_prediction)[-15:] + 1  # Índices das 15 maiores probabilidades
    
    print("Previsão Final:", sorted(predicted_numbers))

if __name__ == "__main__":
    main()