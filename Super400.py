import os
import pandas as pd
import hashlib
import logging

# Configuração de log
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("execution.log", mode="w"),
        logging.StreamHandler()
    ]
)

class Super400:
    def __init__(self):
        self.backup_dir = "backups"  # Nome da pasta de backups
        self.setup_backup_directory()  # Criação da pasta
        self.validate_and_backup_csv()  # Validação e backup do CSV

    def setup_backup_directory(self):
        """Cria a pasta de backups."""
        try:
            # Caminho absoluto para evitar problemas de diretório
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.backup_dir = os.path.join(script_dir, self.backup_dir)

            # Tenta criar a pasta
            os.makedirs(self.backup_dir, exist_ok=True)
            logging.info(f"Pasta de backups criada/verificada: {self.backup_dir}")
        except Exception as e:
            logging.error(f"Erro ao criar a pasta de backups: {e}")
            raise

    def validate_and_backup_csv(self):
        """Valida o CSV e cria um backup."""
        try:
            logging.info("Validando o arquivo CSV...")
            # Verifica se o arquivo existe
            csv_path = "base_Lotofacil.csv"
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"O arquivo {csv_path} não foi encontrado.")

            # Lê o arquivo
            df = pd.read_csv(csv_path, sep=';')
            required_cols = ["Concurso"] + [f"Bola{i}" for i in range(1, 16)]
            if not all(col in df.columns for col in required_cols):
                raise ValueError("CSV inválido: colunas faltantes.")

            # Calcula hash SHA-256 do CSV
            sha256_hash = hashlib.sha256()
            with open(csv_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            current_hash = sha256_hash.hexdigest()
            logging.info(f"SHA-256 do CSV: {current_hash}")

            # Cria o backup com timestamp
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}.csv")
            df.to_csv(backup_path, index=False, sep=';')
            logging.info(f"Backup salvo com sucesso: {backup_path}")
        except Exception as e:
            logging.error(f"Erro na validação ou criação do backup: {e}")
            raise

# Execução principal
if __name__ == "__main__":
    logging.info("Iniciando a execução do script Super400...")
    try:
        super400 = Super400()
        logging.info("Execução concluída com sucesso!")
    except Exception as e:
        logging.critical(f"Falha crítica na execução: {e}")
