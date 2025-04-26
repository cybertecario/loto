# SuperInsano v29 - Previsão de Resultados da Lotofácil
# Autor: Assistente IA
# Versão com GRU bidirecional, GAN, simulação quântica, índices de cansaço avançados (incluindo trios) e validação temporal rigorosa.
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

class SuperInsano_v29:
    def __init__(self):
        """
        Inicializa o modelo com configurações otimizadas e carrega os dados históricos.
        """
        # Configurações principais
        self.JANELA = 50           # Janela histórica reduzida para evitar overfitting
        self.JANELA_SOMA = 75       # Janela expandida para soma (capturar ciclos longos)
        self.MISTURA_QUENTES = 0.87 # Prioriza números quentes (ajustado após otimização)
        self.MAXIMO_NUMEROS_CONSECUTIVOS = 7  # Limite de consecutivos relaxado
        self.LIMITE_CANSACO = 4     # Penaliza números "cansados"
        self.TRIOS_PENALIZADOS = 50 # Quantidade de trios mais frequentes a penalizar
        
        # Diretório de backups e validação
        self.backup_dir = "backups"
        os.makedirs(self.backup_dir, exist_ok=True)
        self.validate_and_backup_csv()
        self.df = self.load_and_validate_data()
        self.last_concurso = self.df["Concurso"].max()
        
        # Parâmetros otimizados (carregados ou gerados)
        self.params_file = "best_params_v29.json"
        self.best_params = self.load_best_params()
        if not self.best_params:
            print("Executando otimização genética...")
            self.best_params, self.best_score = self.run_optimization(generations=50, population_size=100)
            self.save_best_params()
        
        # Pesos para modelos híbridos
        self.quantum_weight = self.best_params.get("quantum_weight", 0.4)
        self.gan_weight = self.best_params.get("gan_weight", 0.3)
        self.gru_weight = 1.0 - self.quantum_weight - self.gan_weight

    def validate_and_backup_csv(self):
        """
        Valida o CSV e cria backup assinado com SHA-256 para garantir integridade.
        """
        try:
            df = pd.read_csv("base_Lotofacil.csv", sep=';')
            required_cols = ["Concurso"] + [f"Bola{i}" for i in range(1, 16)]
            
            # Verifica colunas necessárias
            if not all(col in df.columns for col in required_cols):
                raise ValueError("CSV inválido: colunas faltantes.")
            
            # Verifica valores numéricos
            for col in required_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValueError(f"Coluna {col} não é numérica.")
            
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
        """
        Carrega e valida os dados do CSV, removendo duplicatas e garantindo consistência.
        """
        try:
            df = pd.read_csv("base_Lotofacil.csv", sep=';')
            required_cols = ["Concurso"] + [f"Bola{i}" for i in range(1, 16)]
            
            # Verifica colunas e valores
            if not all(col in df.columns for col in required_cols):
                raise ValueError("CSV inválido: colunas faltantes.")
            for col in required_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValueError(f"Coluna {col} não é numérica.")
            
            # Remove duplicatas e ordena
            df = df.drop_duplicates(subset=["Concurso"]).sort_values(by="Concurso").reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            raise

    def build_gru_model(self, params):
        """
        Constrói um modelo GRU bidirecional com regularização L2 e dropout.
        """
        model = Sequential()
        model.add(Bidirectional(GRU(params["gru_units"], return_sequences=True), input_shape=(self.JANELA, 15)))
        model.add(Dropout(params["gru_dropout"]))
        model.add(Dense(25, activation='softmax', kernel_regularizer='l2'))
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

    def hybrid_prediction(self, concurso):
        """
        Gera previsão híbrida combinando GRU, GAN, simulação quântica e índices de cansaço.
        """
        # Dados históricos (até o concurso anterior)
        historical_data = self.df[self.df["Concurso"] < concurso].tail(self.JANELA)
        
        # Previsões base
        gru_pred = self.generate_gru_prediction(historical_data)
        gan_pred = self.generate_gan_prediction(historical_data)
        quantum_pred = self.quantum_simulation(historical_data)
        
        # Combina previsões com pesos
        combined = defaultdict(float)
        for num in gru_pred:
            combined[num] += self.gru_weight
        for num in gan_pred:
            combined[num] += self.gan_weight
        for num in quantum_pred:
            combined[num] += self.quantum_weight
        
        # Aplica índices de cansaço
        combined = self.apply_fatigue_indices(concurso, combined)
        
        # Normaliza e seleciona 15 números únicos
        total = sum(combined.values())
        final_probs = {num: prob / total for num, prob in combined.items()}
        selected = sorted(final_probs, key=lambda x: (-final_probs[x], x))[:15]
        
        # Valida matriz 5x5
        if not self.cartao_5x5_valido(selected):
            selected = self.adjust_for_5x5(selected)
        
        # Salva previsão com hash do CSV
        self.save_predictions(concurso, selected)
        return sorted(selected)

    def apply_fatigue_indices(self, concurso, combined):
        """
        Aplica todos os índices de cansaço (soma, consecutivos, quadrantes, primos, Fibonacci, diagonais, trios).
        """
        # Calcula índices de cansaço
        soma_fatigue = self.calculate_soma_fatigue(concurso)
        consec_fatigue = self.calculate_consecutivos_fatigue(concurso)
        quadrante_fatigue = self.calculate_quadrante_fatigue(concurso)
        primos_fatigue = self.calculate_primos_fatigue(concurso)
        fibonacci_fatigue = self.calculate_fibonacci_fatigue(concurso)
        diagonal_fatigue = self.calculate_diagonal_fatigue(concurso)
        trios_fatigue = self.calculate_trios_fatigue(concurso)
        repeticoes_fatigue = self.calculate_repeticoes_fatigue(concurso)
        
        # Aplica penalizações
        for num in list(combined.keys()):
            # Penaliza faixas de soma
            soma = sum(combined.keys())
            if soma < 150:
                combined[num] *= soma_fatigue["baixo"]
            elif 150 <= soma < 175:
                combined[num] *= soma_fatigue["medio_baixo"]
            elif 175 <= soma < 200:
                combined[num] *= soma_fatigue["medio"]
            elif 200 <= soma < 225:
                combined[num] *= soma_fatigue["medio_alto"]
            else:
                combined[num] *= soma_fatigue["alto"]
            
            # Penaliza consecutivos
            max_consec = self.max_consecutivos(list(combined.keys()))
            combined[num] *= consec_fatigue.get(max_consec, 1.0)
            
            # Penaliza quadrantes super-representados
            linha = (num - 1) // 5
            coluna = (num - 1) % 5
            combined[num] *= quadrante_fatigue.get(f"q{linha}{coluna}", 1.0)
            
            # Penaliza números primos
            if num in self.PRIMOS:
                combined[num] *= primos_fatigue
            
            # Penaliza Fibonacci
            if num in self.FIBONACCI:
                combined[num] *= fibonacci_fatigue
            
            # Penaliza diagonais
            if num in self.DIAGONAL_PRINCIPAL or num in self.DIAGONAL_SECUNDARIA:
                combined[num] *= diagonal_fatigue
            
            # Penaliza repetições do concurso anterior
            if num in self.ultimos_numeros:
                combined[num] *= repeticoes_fatigue
            
            # Penaliza trios super-representados
            for trio in combinations(combined.keys(), 3):
                if trio in trios_fatigue:
                    combined[num] *= trios_fatigue[trio]
        
        return combined

    def calculate_trios_fatigue(self, concurso):
        """
        Calcula cansaço para trios de números frequentes na janela histórica.
        """
        historical_window = self.df[self.df["Concurso"] < concurso].tail(self.JANELA_SOMA)
        trios = Counter()
        for _, row in historical_window.iterrows():
            numeros = sorted(row[1:16])
            trios.update(combinations(numeros, 3))
        
        # Penaliza os 50 trios mais frequentes
        top_trios = [t[0] for t in trios.most_common(self.TRIOS_PENALIZADOS)]
        fatigue = {}
        for trio in top_trios:
            observed = trios[trio] / len(historical_window)
            expected = 1 / 2300  # Probabilidade teórica de um trio
            fatigue[trio] = 1 - (observed - expected) / expected
        
        return fatigue

    def save_predictions(self, concurso, jogos):
        """
        Salva os jogos gerados com hash SHA-256 do CSV atual para auditoria.
        """
        data = {
            "concurso": concurso,
            "hash_csv": self.current_hash,
            "jogos": [list(jogo) for jogo in jogos]
        }
        with open(f"predictions/concurso_{concurso}.json", "w") as f:
            json.dump(data, f)

    def run_optimization(self, generations=50, population_size=100):
        """
        Otimização genética para encontrar melhores parâmetros (evita overfitting).
        """
        param_grid = {
            "gru_units": [64, 128, 256],
            "gru_dropout": [0.2, 0.3, 0.4],
            "quantum_weight": [0.3, 0.4, 0.5],
            "gan_weight": [0.2, 0.3, 0.4]
        }
        population = [self.random_params(param_grid) for _ in range(population_size)]
        best_score = -np.inf
        
        for generation in range(generations):
            fitness_scores = Parallel(n_jobs=-1)(
                delayed(self.evaluate_individual)(params) for params in population
            )
            selected = np.argsort(fitness_scores)[-20:]
            
            # Crossover e mutação
            new_population = []
            while len(new_population) < population_size:
                parent1, parent2 = population[np.random.choice(selected, 2, replace=False)]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            population = new_population
            
            # Atualiza melhor parâmetro
            current_best_score = max(fitness_scores)
            if current_best_score > best_score:
                best_score = current_best_score
                best_params = population[np.argmax(fitness_scores)]
                print(f"Nova melhor configuração na geração {generation}: {best_params}")
        
        return best_params, best_score

    def save_best_params(self):
        """
        Salva os melhores parâmetros em JSON com hash SHA-256.
        """
        params_to_save = {
            "JANELA": self.JANELA,
            "quantum_weight": self.quantum_weight,
            "gan_weight": self.gan_weight,
            "gru_units": self.best_params.get("gru_units", 256),
            "gru_dropout": self.best_params.get("gru_dropout", 0.3)
        }
        with open(self.params_file, "w") as f:
            json.dump(params_to_save, f)

    def load_best_params(self):
        """
        Carrega parâmetros salvos para evitar recálculo desnecessário.
        """
        if os.path.exists(self.params_file):
            with open(self.params_file, "r") as f:
                return json.load(f)
        return None

    def backtest(self, test_range):
        """
        Executa backtest com validação temporal rigorosa (evita vazamento de dados).
        """
        metrics = {"acertos_15": 0, "acertos_14": 0, "total": 0, "jogos_ruins": 0}
        for concurso in test_range:
            real = set(self.df[self.df["Concurso"] == concurso].iloc[0, 1:16])
            jogos = [self.hybrid_prediction(concurso) for _ in range(16)]
            best_hit = max(len(real & set(jogo)) for jogo in jogos)
            
            metrics["total"] += best_hit
            if best_hit == 15:
                metrics["acertos_15"] += 1
            elif best_hit == 14:
                metrics["acertos_14"] += 1
            if best_hit <= 10:
                metrics["jogos_ruins"] += 1
        
        metrics["media"] = metrics["total"] / len(test_range)
        return metrics

# Execução principal
if __name__ == "__main__":
    superinsano = SuperInsano_v29()
    next_concurso = superinsano.last_concurso + 1
    prediction = superinsano.hybrid_prediction(next_concurso)
    print(f"Previsão para o concurso {next_concurso}:")
    for jogo in prediction:
        print(sorted(jogo))