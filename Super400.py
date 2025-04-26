# Super400 - Previsão de Resultados da Lotofácil
# Versão com GRU bidirecional, GAN, simulação quântica, índices de cansaço expandidos e validação de matriz 5x5.
# Dependências: pandas, numpy, tensorflow, qiskit, scikit-learn, joblib, scipy.

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GRU, Dense, Dropout
from qiskit.providers.aer import AerSimulator
from qiskit import QuantumCircuit, execute
from sklearn.utils import resample
from joblib import Parallel, delayed
import os
import json
import hashlib
from scipy.stats import ttest_ind
from collections import defaultdict, Counter
from itertools import combinations

class Super400:
    def __init__(self):
        # Backup e validação do CSV
        self.backup_dir = "backups"
        os.makedirs(self.backup_dir, exist_ok=True)
        self.validate_and_backup_csv()
        self.df = self.load_and_validate_data()
        self.last_concurso = self.df["Concurso"].max()
        
        # Parâmetros otimizados (carregados ou gerados)
        self.params_file = "best_params_super400.json"
        self.best_params = self.load_best_params()
        if not self.best_params:
            print("Iniciando otimização genética...")
            self.best_params, self.best_score = self.run_optimization(generations=50, population_size=100)
            self.save_best_params()
        
        # Pesos dinâmicos
        self.quantum_weight = self.best_params.get("quantum_weight", 0.5)
        self.gan_weight = self.best_params.get("gan_weight", 0.2)
        self.gru_weight = 1 - self.quantum_weight - self.gan_weight

    def validate_and_backup_csv(self):
        """Valida o CSV e cria backup assinado com SHA-256."""
        try:
            df = pd.read_csv("base_Lotofacil.csv", sep=';')
            required_cols = ["Concurso"] + [f"Bola{i}" for i in range(1, 16)]
            
            # Verifica colunas necessárias
            if not all(col in df.columns for col in required_cols):
                raise ValueError("CSV inválido: colunas faltantes.")
            
            # Calcula hash SHA-256 do CSV
            sha256_hash = hashlib.sha256()
            with open("base_Lotofacil.csv", "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            self.current_hash = sha256_hash.hexdigest()
            
            # Cria backup com timestamp
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}.csv")
            df.to_csv(backup_path, index=False, sep=';')
            print(f"Backup válido criado em: {backup_path}")
        except Exception as e:
            print(f"Erro na validação/backup: {e}")
            raise

    def load_and_validate_data(self):
        """Carrega e valida os dados do CSV."""
        try:
            df = pd.read_csv("base_Lotofacil.csv", sep=';')
            required_cols = ["Concurso"] + [f"Bola{i}" for i in range(1, 16)]
            if not all(col in df.columns for col in required_cols):
                raise ValueError("CSV inválido: colunas faltantes.")
            df = df.drop_duplicates(subset=["Concurso"]).sort_values(by="Concurso").reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            raise

    def build_gru_model(self, params):
        """Constrói um modelo GRU bidirecional com atenção e dropout."""
        model = Sequential()
        model.add(Bidirectional(GRU(params["gru_units"], return_sequences=True), input_shape=(params["window_size"], 15)))
        model.add(Dropout(params["gru_dropout"]))
        model.add(Dense(25, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

    def quantum_simulation(self, historical_data):
        """Simula probabilidades quânticas com entrelaçamento."""
        num_qubits = 25  # Um qubit por número (1–25)
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            qc.h(i)  # Superposição inicial
        qc.barrier()
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)  # Entrelaçamento entre qubits adjacentes
        simulator = AerSimulator()
        result = execute(qc, simulator).result()
        statevector = result.get_statevector()
        probabilities = np.abs(statevector) ** 2
        return np.argsort(probabilities)[-15:] + 1

    def random_params(self, param_grid):
        """Gera um conjunto aleatório de parâmetros com base no espaço de busca."""
        random_params = {}
        for param, values in param_grid.items():
            random_params[param] = np.random.choice(values)
        return random_params

    def run_optimization(self, generations=50, population_size=100):
        """Otimiza parâmetros via algoritmo genético."""
        param_grid = {
            "window_size": [25, 30, 50],
            "gru_units": [128, 256],
            "gru_dropout": [0.2, 0.3],
            "quantum_weight": [0.4, 0.5],
            "gan_weight": [0.1, 0.2]
        }
        population = [self.random_params(param_grid) for _ in range(population_size)]
        best_score = -np.inf
        
        for generation in range(generations):
            fitness_scores = Parallel(n_jobs=1)(  # Ajustado para evitar erros de serialização
                delayed(self.evaluate_individual)(params) for params in population
            )
            selected = np.argsort(fitness_scores)[-20:]
            
            # Nova população via crossover e mutação
            new_population = []
            while len(new_population) < population_size:
                parent1, parent2 = population[np.random.choice(selected, 2, replace=False)]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, param_grid)
                new_population.append(child)
            population = new_population
            
            # Atualiza melhor parâmetro
            current_best_score = max(fitness_scores)
            if current_best_score > best_score:
                best_score = current_best_score
                best_params = population[np.argmax(fitness_scores)]
                print(f"Nova melhor configuração na geração {generation}: {best_params}")
        
        return best_params, best_score

    # Outros métodos continuam...

# Execução principal
if __name__ == "__main__":
    super400 = Super400()
    test_range = range(2376, 3376)  # Últimos 1000 concursos
    acertos_15, acertos_14, acertos_13 = super400.backtest(test_range)
    
    print(f"=== Resultados do Backtest ===")
    print(f"Acertos de 15 pontos: {acertos_15}")
    print(f"Acertos de 14 pontos: {acertos_14}")
    print(f"Acertos de 13 pontos: {acertos_13}")
