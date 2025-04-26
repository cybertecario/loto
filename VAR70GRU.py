import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import os
import shutil
import time
from collections import defaultdict
from sklearn.utils import resample


class VAR70GRU:
    def __init__(self):
        # Backup inicial e validação do arquivo CSV
        self.backup_dir = "backups"
        os.makedirs(self.backup_dir, exist_ok=True)
        self.validate_and_backup_csv()

        # Carregar dados
        self.df = self.load_and_validate_data()
        self.last_concurso = self.df["Concurso"].max()

        # Parâmetros iniciais ou recuperação de backup
        self.params_file = "best_params.json"
        self.log_file = "training_log.txt"
        self.best_score = -np.inf
        self.best_params = self.load_best_params()

        # Executar otimização genética automaticamente
        if not self.best_params:
            print("Executando otimização genética...")
            self.best_params, self.best_score = self.run_optimization(generations=30, population_size=50)
            self.save_best_params()

        # Modelo GRU com os melhores parâmetros
        self.model = self.build_model(self.best_params)

    def validate_and_backup_csv(self):
        """Valida o arquivo CSV e cria um backup se estiver íntegro."""
        try:
            df = pd.read_csv("base_Lotofacil.csv", sep=';')
            required_cols = ["Concurso"] + [f"Bola{i}" for i in range(1, 16)]
            if not all(col in df.columns for col in required_cols):
                raise ValueError("CSV inválido: colunas faltantes.")

            # Criar backup com timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}.csv")
            shutil.copy("base_Lotofacil.csv", backup_path)
        except Exception as e:
            print(f"Erro ao validar CSV: {e}")
            # Recuperar último backup
            latest_backup = max(
                [os.path.join(self.backup_dir, f) for f in os.listdir(self.backup_dir)],
                key=os.path.getctime,
                default=None
            )
            if latest_backup:
                shutil.copy(latest_backup, "base_Lotofacil.csv")
                print("Último backup restaurado.")
            else:
                raise FileNotFoundError("Nenhum backup disponível.")

    def load_and_validate_data(self):
        """Carrega e valida os dados do CSV."""
        df = pd.read_csv("base_Lotofacil.csv", sep=';')
        required_cols = ["Concurso"] + [f"Bola{i}" for i in range(1, 16)]
        if not all(col in df.columns for col in required_cols):
            raise ValueError("CSV inválido: colunas faltantes.")
        df = df.drop_duplicates(subset=["Concurso"]).sort_values(by="Concurso")
        return df

    def load_best_params(self):
        """Carrega os melhores parâmetros de treinamento salvos."""
        if os.path.exists(self.params_file):
            with open(self.params_file, "r") as f:
                import json
                return json.load(f)
        return None

    def save_best_params(self):
        """Salva os melhores parâmetros no arquivo JSON."""
        with open(self.params_file, "w") as f:
            import json
            json.dump(self.best_params, f)

    def build_model(self, params):
        """Constrói o modelo GRU com os parâmetros fornecidos."""
        model = Sequential()
        for _ in range(params["num_layers"]):
            model.add(GRU(params["units"], return_sequences=True, input_shape=(100, 15)))
            model.add(Dropout(params["dropout"]))
        model.add(Dense(25, activation='softmax'))
        optimizer = Adam(learning_rate=params["learning_rate"])
        loss = self.get_loss_function(params["loss_function"])
        model.compile(loss=loss, optimizer=optimizer)
        return model

    def get_loss_function(self, loss_name):
        """Retorna a função de perda adequada."""
        from tensorflow.keras.losses import CategoricalCrossentropy
        if loss_name == "categorical_crossentropy":
            return CategoricalCrossentropy()
        elif loss_name == "focal_loss":
            from focal_loss import SparseCategoricalFocalLoss
            return SparseCategoricalFocalLoss(gamma=2)
        elif loss_name == "weighted_categorical_crossentropy":
            from tensorflow.keras.losses import CategoricalCrossentropy
            return CategoricalCrossentropy()
        else:
            raise ValueError(f"Função de perda desconhecida: {loss_name}")

    def train_ml_model(self, historical_data):
        """Treina o modelo com validação cruzada."""
        X = historical_data.iloc[:, 1:].values.reshape(-1, 100, 15)
        y = np.eye(25)[historical_data.iloc[:, 1:].values.flatten()]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            self.model.fit(X_train, y_train, batch_size=self.best_params["batch_size"], epochs=10, verbose=0)
            score = self.model.evaluate(X_val, y_val, verbose=0)
            scores.append(score)

        avg_score = np.mean(scores)
        if avg_score > self.best_score:
            self.best_score = avg_score
            self.save_best_params()
        return avg_score

    def generate_prediction_ml(self, concurso):
        """Gera previsões usando o modelo treinado."""
        historical_data = self.df[self.df["Concurso"] < concurso].tail(100)
        X = historical_data.iloc[:, 1:].values.reshape(1, 100, 15)
        probs = self.model.predict(X)[-1]
        selected = np.argsort(probs)[-15:]
        return sorted(selected + 1)

    def backtest(self, test_range):
        """Executa backtest com bootstrapping."""
        hits = []
        for concurso in test_range:
            real = set(self.df[self.df["Concurso"] == concurso].iloc[0, 1:16])
            predicted = set(self.generate_prediction_ml(concurso))
            hits.append(len(real & predicted))

            # Bootstrapping para robustez
            bootstrap_hits = []
            for _ in range(10):
                bootstrap_sample = resample(self.df[self.df["Concurso"] < concurso].iloc[:, 1:].values, replace=True)
                bootstrap_predicted = set(self.generate_prediction_ml(concurso))
                bootstrap_hits.append(len(set(bootstrap_sample.flatten()) & bootstrap_predicted))
            avg_bootstrap_hit = np.mean(bootstrap_hits)
            hits[-1] = (hits[-1] + avg_bootstrap_hit) / 2

            # Log de progresso
            self.log_progress(concurso, hits[-1])

        return hits

    def log_progress(self, concurso, hit):
        """Registra o progresso em um arquivo de log."""
        with open(self.log_file, "a") as f:
            f.write(f"Concurso: {concurso}, Hits: {hit}\n")

    def reinforcement_learning_update(self, real):
        """Atualiza os pesos via reinforcement learning."""
        predicted = set(self.generate_prediction_ml(self.last_concurso))
        reward = len(real & predicted)
        self.best_params["quantum_weight"] += 0.01 * reward
        self.best_params["quantum_weight"] = min(max(self.best_params["quantum_weight"], 0.1), 0.9)

    def run_optimization(self, generations=30, population_size=50):
        """Executa otimização genética para ajustar os parâmetros."""
        param_grid = {
            "num_layers": [1, 2, 3],
            "units": [64, 128, 256],
            "dropout": [0.2, 0.3, 0.4],
            "learning_rate": [0.001, 0.0005, 0.0001],
            "batch_size": [16, 32, 64],
            "quantum_weight": [0.4, 0.5, 0.6],
            "loss_function": ["categorical_crossentropy", "focal_loss", "weighted_categorical_crossentropy"]
        }

        population = [self.random_params(param_grid) for _ in range(population_size)]

        for generation in range(generations):
            print(f"Generation {generation + 1}/{generations}")
            fitness_scores = Parallel(n_jobs=-1)(
                delayed(self.train_ml_model)(self.df[self.df["Concurso"] < self.last_concurso]) for _ in population
            )
            best_index = np.argmax(fitness_scores)
            if fitness_scores[best_index] > self.best_score:
                self.best_score = fitness_scores[best_index]
                self.best_params = population[best_index]
                print(f"New Best Score: {self.best_score}, Params: {self.best_params}")
                self.save_best_params()

            selected_indices = np.argsort(fitness_scores)[-int(population_size / 2):]
            selected_population = [population[i] for i in selected_indices]

            new_population = []
            while len(new_population) < population_size:
                parent1, parent2 = np.random.choice(selected_population, size=2, replace=False)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        return self.best_params, self.best_score

    def random_params(self, param_grid):
        """Gera parâmetros aleatórios."""
        return {key: np.random.choice(values) for key, values in param_grid.items()}

    def crossover(self, parent1, parent2):
        """Realiza crossover entre dois pais."""
        child = {}
        for key in parent1:
            child[key] = parent1[key] if np.random.rand() < 0.5 else parent2[key]
        return child

    def mutate(self, params):
        """Realiza mutação nos parâmetros."""
        mutated = params.copy()
        for key in mutated:
            if isinstance(mutated[key], int):
                mutated[key] += np.random.randint(-1, 2)
                mutated[key] = max(1, mutated[key])
            elif isinstance(mutated[key], float):
                mutated[key] += np.random.uniform(-0.1, 0.1)
                mutated[key] = np.clip(mutated[key], 0.0, 1.0)
        return mutated


# Executar otimização automaticamente ao instanciar a classe
if __name__ == "__main__":
    optimizer = VAR70GRU()

    # Backtest nos últimos 20 concursos
    last_20_concursos = range(optimizer.last_concurso - 19, optimizer.last_concurso + 1)
    hits = optimizer.backtest(last_20_concursos)

    # Exibir resultados
    results_df = pd.DataFrame({
        "Concurso": last_20_concursos,
        "Acertos": hits
    })

    print("=== Resultados dos Últimos 20 Concursos ===")
    print(results_df)

    # Listar concursos com 15 pontos
    concursos_com_15_pontos = results_df[results_df["Acertos"] == 15]
    print("\n=== Concursos com 15 Pontos ===")
    if not concursos_com_15_pontos.empty:
        print(concursos_com_15_pontos)
    else:
        print("Nenhum concurso com 15 pontos encontrado.")