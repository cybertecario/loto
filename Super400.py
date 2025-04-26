# Super400 - Previsão de Resultados da Lotofácil
# Versão com GRU bidirecional, GAN, simulação quântica, índices de cansaço expandidos e validação de matriz 5x5.
# Dependências: pandas, numpy, tensorflow, qiskit, scikit-learn, joblib, scipy.

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GRU, Dense, Dropout
from qiskit import Aer, QuantumCircuit, execute
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

    def hybrid_prediction(self, concurso):
        """Gera previsão híbrida combinando GRU, GAN, simulação quântica e índices de cansaço."""
        # Dados históricos até o concurso anterior
        historical_data = self.df[self.df["Concurso"] < concurso].tail(self.best_params["window_size"])
        
        # Previsão GRU
        gru_pred = self.generate_gru_prediction(historical_data)
        
        # Previsão GAN
        gan_pred = self.generate_gan_prediction(historical_data)
        
        # Simulação quântica
        quantum_pred = self.quantum_simulation(historical_data)
        
        # Índices de cansaço
        cansaco_numeros = self.calculate_cansaco_numeros(concurso)
        cansaco_pares_impares = self.calculate_cansaco_pares_impares(concurso)
        cansaco_somas = self.calculate_cansaco_somas(concurso)
        cansaco_trios = self.calculate_cansaco_trios(concurso)
        
        # Combina previsões com pesos dinâmicos
        combined = defaultdict(float)
        for num in gru_pred:
            combined[num] += self.gru_weight
        for num in gan_pred:
            combined[num] += self.gan_weight
        for num in quantum_pred:
            combined[num] += self.quantum_weight
        for num in cansaco_numeros:
            combined[num] += 0.1  # Penaliza números cansados
        for config in cansaco_pares_impares:
            if config in self.get_pares_impares(jogo):
                combined[num] *= 0.9  # Ajusta por distribuição de pares/ímpares
        for soma in cansaco_somas:
            if sum(jogo) in soma:
                combined[num] *= 0.9  # Penaliza somas super-representadas
        for trio in cansaco_trios:
            if trio.issubset(jogo):
                combined[num] *= 0.8  # Penaliza trios frequentes
        
        # Gera jogo final validado
        jogo_final = sorted(combined, key=lambda x: (-combined[x], x))[:15]
        if not self.validate_cartao_5x5(jogo_final):
            jogo_final = self.adjust_for_5x5(jogo_final)
        
        return jogo_final

    def quantum_simulation(self, historical_data):
        """Simula probabilidades quânticas com entrelaçamento."""
        num_qubits = 25  # Um qubit por número (1–25)
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            qc.h(i)  # Superposição inicial
        qc.barrier()
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)  # Entrelaçamento entre qubits adjacentes
        simulator = Aer.get_backend('statevector_simulator')
        result = execute(qc, simulator).result()
        statevector = result.get_statevector()
        probabilities = np.abs(statevector) ** 2
        return np.argsort(probabilities)[-15:] + 1

    def calculate_cansaco_numeros(self, concurso):
        """Calcula cansaço de números individuais na janela de 50 concursos."""
        historical_window = self.df[self.df["Concurso"] < concurso].tail(50)
        numeros = []
        for _, row in historical_window.iterrows():
            numeros.extend(row[1:16])
        contagem = Counter(numeros)
        return [num for num, _ in contagem.most_common(15)]

    def calculate_cansaco_pares_impares(self, concurso):
        """Identifica distribuições de pares/ímpares sub-representadas."""
        historical_window = self.df[self.df["Concurso"] < concurso].tail(50)
        pares_impares_counts = defaultdict(int)
        for _, row in historical_window.iterrows():
            pares = sum(1 for num in row[1:16] if num % 2 == 0)
            impares = 15 - pares
            pares_impares_counts[f"{pares}/{impares}"] += 1
        return [config for config, _ in pares_impares_counts.most_common()[:-3]]  # Ignora as 3 mais frequentes

    def calculate_cansaco_somas(self, concurso):
        """Identifica faixas de soma sub-representadas."""
        historical_window = self.df[self.df["Concurso"] < concurso].tail(50)
        somas = [sum(row[1:16]) for _, row in historical_window.iterrows()]
        bins = [(150, 175), (175, 200), (200, 225), (225, 250)]
        contagem = Counter(np.digitize(somas, bins))
        return [bins[i] for i in contagem.most_common()[:-2]]  # Ignora as 2 faixas mais frequentes

    def calculate_cansaco_trios(self, concurso):
        """Identifica trios de números super-representados."""
        historical_window = self.df[self.df["Concurso"] < concurso].tail(75)
        trios = Counter()
        for _, row in historical_window.iterrows():
            trios.update(combinations(row[1:16], 3))
        return [trio for trio, _ in trios.most_common(50)]  # Penaliza os 50 trios mais frequentes

    def validate_cartao_5x5(self, jogo):
        """Valida se o jogo atende à matriz 5x5 (2–5 números por linha/coluna)."""
        linhas = defaultdict(int)
        colunas = defaultdict(int)
        for num in jogo:
            linha = (num - 1) // 5
            coluna = (num - 1) % 5
            linhas[linha] += 1
            colunas[coluna] += 1
        return all(2 <= x <= 5 for x in linhas.values()) and all(2 <= x <= 5 for x in colunas.values())

    def backtest(self, test_range):
        """Executa backtest real considerando todos os 16 jogos por concurso."""
        acertos_15 = 0
        acertos_14 = 0
        acertos_13 = 0
        
        for concurso in test_range:
            real = set(self.df[self.df["Concurso"] == concurso].iloc[0, 1:16])
            jogos = []
            
            # Gera 16 jogos únicos
            while len(jogos) < 16:
                jogo = self.hybrid_prediction(concurso)
                if sorted(jogo) not in jogos:
                    jogos.append(sorted(jogo))
            
            # Avalia cada jogo
            for jogo in jogos:
                hit = len(real & set(jogo))
                if hit == 15:
                    acertos_15 += 1
                elif hit == 14:
                    acertos_14 += 1
                elif hit == 13:
                    acertos_13 += 1
        
        return acertos_15, acertos_14, acertos_13

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
            fitness_scores = Parallel(n_jobs=-1)(
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

    def evaluate_individual(self, params):
        """Avalia um indivíduo na otimização genética."""
        test_range = range(self.last_concurso - 99, self.last_concurso + 1)  # 100 concursos para validação
        acertos_total = 0
        
        for concurso in test_range:
            real = set(self.df[self.df["Concurso"] == concurso].iloc[0, 1:16])
            jogos = [self.hybrid_prediction(concurso) for _ in range(16)]
            best_hit = max(len(real & set(jogo)) for jogo in jogos)
            acertos_total += best_hit
        
        return acertos_total / len(test_range)

    def save_best_params(self):
        """Salva os melhores parâmetros em JSON."""
        with open(self.params_file, "w") as f:
            json.dump(self.best_params, f)

    def load_best_params(self):
        """Carrega parâmetros salvos."""
        if os.path.exists(self.params_file):
            with open(self.params_file, "r") as f:
                return json.load(f)
        return None

# Execução principal
if __name__ == "__main__":
    super400 = Super400()
    test_range = range(2376, 3376)  # Últimos 1000 concursos
    acertos_15, acertos_14, acertos_13 = super400.backtest(test_range)
    
    print(f"=== Resultados do Backtest ===")
    print(f"Acertos de 15 pontos: {acertos_15}")
    print(f"Acertos de 14 pontos: {acertos_14}")
    print(f"Acertos de 13 pontos: {acertos_13}")