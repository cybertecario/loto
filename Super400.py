# Super400 - Previsão de Resultados da Lotofácil
# ...

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
            print("Validando o arquivo CSV...")
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
            print(f"SHA-256 do CSV: {self.current_hash}")
            
            # Cria pasta de backups com caminho absoluto
            script_dir = os.path.dirname(os.path.abspath(__file__))
            backup_dir = os.path.join(script_dir, self.backup_dir)
            os.makedirs(backup_dir, exist_ok=True)
            print(f"Pasta de backups criada/verificada: {backup_dir}")
            
            # Cria backup com timestamp
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"backup_{timestamp}.csv")
            df.to_csv(backup_path, index=False, sep=';')
            print(f"Backup salvo em: {backup_path}")
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

    def load_best_params(self):
        """Carrega parâmetros salvos."""
        if os.path.exists(self.params_file):
            with open(self.params_file, "r") as f:
                return json.load(f)
        return None

    def save_best_params(self):
        """Salva os melhores parâmetros em JSON."""
        with open(self.params_file, "w") as f:
            json.dump(self.best_params, f)

    # Outros métodos do script continuam...
