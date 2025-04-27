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
import logging

# Configuração do logger
logging.basicConfig(
    level=logging.DEBUG,  # Nível mínimo de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("super400.log", mode="w"),  # Grava os logs em um arquivo
        logging.StreamHandler()  # Exibe os logs no console
    ]
)

class Super400:
    def __init__(self):
        # Backup e validação do CSV
        logging.info("Inicializando a classe Super400.")
        self.backup_dir = "backups"
        os.makedirs(self.backup_dir, exist_ok=True)
        self.validate_and_backup_csv()
        self.df = self.load_and_validate_data()
        self.last_concurso = self.df["Concurso"].max()
        
        # Parâmetros otimizados (carregados ou gerados)
        self.params_file = "best_params_super400.json"
        self.best_params = self.load_best_params()
        if not self.best_params:
            logging.info("Iniciando otimização genética.")
            self.best_params, self.best_score = self.run_optimization(generations=50, population_size=100)
            self.save_best_params()
        
        # Pesos dinâmicos
        self.quantum_weight = self.best_params.get("quantum_weight", 0.5)
        self.gan_weight = self.best_params.get("gan_weight", 0.2)
        self.gru_weight = 1 - self.quantum_weight - self.gan_weight

    def validate_and_backup_csv(self):
        """Valida o CSV e cria backup assinado com SHA-256."""
        try:
            logging.info("Validando o arquivo CSV.")
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
            logging.debug(f"SHA-256 do CSV: {self.current_hash}")
            
            # Cria pasta de backups com caminho absoluto
            script_dir = os.path.dirname(os.path.abspath(__file__))
            backup_dir = os.path.join(script_dir, self.backup_dir)
            os.makedirs(backup_dir, exist_ok=True)
            logging.info(f"Pasta de backups criada/verificada: {backup_dir}")
            
            # Cria backup com timestamp
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"backup_{timestamp}.csv")
            df.to_csv(backup_path, index=False, sep=';')
            logging.info(f"Backup salvo em: {backup_path}")
        except Exception as e:
            logging.error(f"Erro na validação/backup: {e}")
            raise

    def load_and_validate_data(self):
        """Carrega e valida os dados do CSV."""
        try:
            logging.info("Carregando e validando os dados do CSV.")
            df = pd.read_csv("base_Lotofacil.csv", sep=';')
            required_cols = ["Concurso"] + [f"Bola{i}" for i in range(1, 16)]
            if not all(col in df.columns for col in required_cols):
                raise ValueError("CSV inválido: colunas faltantes.")
            df = df.drop_duplicates(subset=["Concurso"]).sort_values(by="Concurso").reset_index(drop=True)
            logging.info("Dados carregados e validados com sucesso.")
            return df
        except Exception as e:
            logging.error(f"Erro ao carregar dados: {e}")
            raise

    def load_best_params(self):
        """Carrega parâmetros salvos."""
        if os.path.exists(self.params_file):
            logging.info("Carregando os melhores parâmetros salvos.")
            with open(self.params_file, "r") as f:
                return json.load(f)
        logging.warning("Nenhum parâmetro salvo encontrado.")
        return None

    def save_best_params(self):
        """Salva os melhores parâmetros em JSON."""
        try:
            logging.info("Salvando os melhores parâmetros em JSON.")
            with open(self.params_file, "w") as f:
                json.dump(self.best_params, f)
            logging.info("Melhores parâmetros salvos com sucesso.")
        except Exception as e:
            logging.error(f"Erro ao salvar os melhores parâmetros: {e}")
            raise

    # Outros métodos continuam...

if __name__ == "__main__":
    logging.info("Iniciando a execução do script Super400.")
    try:
        super400 = Super400()
        test_range = range(2376, 3376)  # Últimos 1000 concursos
        acertos_15, acertos_14, acertos_13 = super400.backtest(test_range)
        
        logging.info(f"=== Resultados do Backtest ===")
        logging.info(f"Acertos de 15 pontos: {acertos_15}")
        logging.info(f"Acertos de 14 pontos: {acertos_14}")
        logging.info(f"Acertos de 13 pontos: {acertos_13}")
    except Exception as e:
        logging.critical(f"Erro crítico durante a execução do script: {e}")
