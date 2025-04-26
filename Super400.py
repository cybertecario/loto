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

    # Restante do código continua...
