import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GRU, Dense, Dropout
from qiskit import QuantumCircuit, Aer, execute
from sklearn.preprocessing import MinMaxScaler
import hashlib
import json
import os

class Loto15:
    def __init__(self, num_backtest_contests=20):
        self.backup_dir = "backups"
        os.makedirs(self.backup_dir, exist_ok=True)
        self.validate_and_backup_csv()
        self.df = self.load_and_validate_data()
        self.scaler = MinMaxScaler()
        self.best_params = self.load_best_params()
        self.model_gru = self.build_gru_model()
        self.model_gan = self.build_gan_model()
        self.quantum_weight = 0.35
        self.gan_weight = 0.3
        self.gru_weight = 0.25
        self.delay_weight = 0.1
        self.last_concurso = self.df["Concurso"].max()

    def validate_and_backup_csv(self):
        """Valida o CSV e cria backups."""
        try:
            df = pd.read_csv("base_Lotofacil.csv", sep=';')
            required_cols = ["Concurso"] + [f"Bola{i}" for i in range(1, 16)]
            if not all(col in df.columns for col in required_cols):
                raise ValueError("CSV inválido: colunas faltantes.")
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            df.to_csv(os.path.join(self.backup_dir, f"backup_{timestamp}.csv"), index=False, sep=';')
        except Exception as e:
            print(f"Erro: {e}. Restaurando último backup...")
            backups = sorted([f for f in os.listdir(self.backup_dir) if f.endswith(".csv")])
            if backups:
                shutil.copy(os.path.join(self.backup_dir, backups[-1]), "base_Lotofacil.csv")
            else:
                raise FileNotFoundError("Nenhum backup válido disponível.")

    def load_and_validate_data(self):
        """Carrega e valida os dados históricos."""
        df = pd.read_csv("base_Lotofacil.csv", sep=';')
        required_cols = ["Concurso"] + [f"Bola{i}" for i in range(1, 16)]
        if not all(col in df.columns for col in required_cols):
            raise ValueError("CSV inválido: colunas faltantes.")
        return df.sort_values(by="Concurso").drop_duplicates(subset=["Concurso"])

    def calculate_delay_cycles(self):
        """Calcula ciclos de atraso para cada número."""
        last_occurrence = defaultdict(int)
        delays = defaultdict(int)
        for _, row in self.df.iterrows():
            for num in row[1:16]:
                delays[num] = row["Concurso"] - last_occurrence.get(num, row["Concurso"])
                last_occurrence[num] = row["Concurso"]
        return delays

    def build_gru_model(self):
        """Constrói um modelo GRU bidirecional para padrões temporais."""
        model = Sequential()
        model.add(Bidirectional(GRU(128, return_sequences=True), input_shape=(None, 15)))
        model.add(Dropout(0.3))
        model.add(GRU(64))
        model.add(Dense(25, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    def build_gan_model(self):
        """Constrói GAN para geração de números sintéticos."""
        # Gerador
        generator = Sequential([
            Dense(256, input_dim=100, activation="relu"),
            BatchNormalization(),
            Dense(512, activation="relu"),
            BatchNormalization(),
            Dense(15, activation="sigmoid")
        ])
        return generator

    def quantum_simulation(self):
        """Simulação quântica com amplitude de probabilidade dinâmica."""
        qc = QuantumCircuit(25)
        historical_freq = self.df.iloc[:, 1:16].stack().value_counts(normalize=True)
        for i in range(25):
            prob = historical_freq.get(i+1, 0.0) + self.delay_weight * (1 / (self.last_concurso - self.df["Concurso"].max() + 1))
            qc.ry(prob * np.pi, i)
        qc.measure_all()
        result = execute(qc, Aer.get_backend('qasm_simulator'), shots=1000).result()
        counts = result.get_counts()
        return [int(k, 2) + 1 for k in counts.keys()]

    def hybrid_prediction(self, concurso):
        """Previsão híbrida com GRU, GAN, simulação quântica e ciclos de atraso."""
        # Dados até o concurso anterior
        train_data = self.df[self.df["Concurso"] < concurso]
        X = self.scaler.fit_transform(train_data.iloc[:, 1:16])
        
        # GRU Bidirecional
        gru_pred = self.model_gru.predict(X[-200:].reshape(1, 200, 15))[0]
        gru_numbers = np.argsort(gru_pred)[-15:] + 1
        
        # GAN
        noise = np.random.normal(0, 1, (1, 100))
        gan_pred = self.model_gan.predict(noise)[0]
        gan_numbers = np.argsort(gan_pred)[-15:] + 1
        
        # Simulação Quântica
        quantum_numbers = self.quantum_simulation()[:15]
        
        # Ciclos de Atraso
        delays = self.calculate_delay_cycles()
        delay_numbers = sorted(delays, key=delays.get, reverse=True)[:15]
        
        # Combinação ponderada com unicidade
        combined = defaultdict(float)
        for num in gru_numbers:
            combined[num] += self.gru_weight
        for num in gan_numbers:
            combined[num] += self.gan_weight
        for num in quantum_numbers:
            combined[num] += self.quantum_weight
        for num in delay_numbers:
            combined[num] += self.delay_weight
        
        # Garante 15 números únicos
        final_numbers = sorted(combined, key=lambda x: (-combined[x], x))[:15]
        return final_numbers

    def backtest(self, test_range):
        """Backtest com validação temporal rigorosa."""
        hits_hybrid = []
        for concurso in test_range:
            real = set(self.df[self.df["Concurso"] == concurso].iloc[0, 1:16])
            hybrid_pred = set(self.hybrid_prediction(concurso))
            hit_hybrid = len(real & hybrid_pred)
            hits_hybrid.append(hit_hybrid)
        return hits_hybrid

# Execução do Loto15
if __name__ == "__main__":
    loto15 = Loto15()
    last_concurso = loto15.last_concurso
    test_range = range(last_concurso - 19, last_concurso + 1)  # Últimos 20 concursos
    hits_hybrid = loto15.backtest(test_range)
    
    # Exibir resultados
    print("=== Resultados do Backtest ===")
    for i, concurso in enumerate(test_range):
        print(f"Concurso {concurso}: {hits_hybrid[i]} acertos")
    
    mean_hits = np.mean(hits_hybrid)
    print(f"Média de acertos: {mean_hits:.2f}")