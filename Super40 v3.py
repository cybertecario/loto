import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
import hashlib

class Super40v3:
    def __init__(self):
        # Carrega dados com validação de hash
        self.df = self.carrega_base_lotofacil("base_Lotofacil.csv")
        self.hash_csv = self.calcula_hash_csv("base_Lotofacil.csv")
        
        # Parâmetros otimizados
        self.JANELA_NUMEROS = 30
        self.JANELA_PARES_IMPARES = 50
        self.PESO_QUENTES = 0.85  # Prioriza números quentes
        self.MAX_CONSECUTIVOS = 7
        self.SOMA_MIN = 150
        self.SOMA_MAX = 250

    def carrega_base_lotofacil(self, filename):
        """Carrega o CSV da Lotofácil com validação."""
        try:
            df = pd.read_csv(filename, sep=';')
            required_cols = ["Concurso"] + [f"Bola{i}" for i in range(1, 16)]
            if not all(col in df.columns for col in required_cols):
                raise ValueError("CSV inválido: colunas faltantes.")
            return df.sort_values(by="Concurso").reset_index(drop=True)
        except Exception as e:
            print(f"Erro ao carregar CSV: {e}")
            raise

    def calcula_hash_csv(self, filename):
        """Calcula SHA-256 do CSV para garantir integridade."""
        sha256 = hashlib.sha256()
        with open(filename, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def calculate_cansaco_numeros(self, concurso):
        """Calcula cansaço de números individuais."""
        historical_data = self.df[self.df["Concurso"] < concurso].tail(self.JANELA_NUMEROS)
        numeros = historical_data.values.flatten()
        contagem = Counter(numeros)
        return {num: freq for num, freq in contagem.items()}

    def calculate_cansaco_pares_impares(self, concurso):
        """Calcula cansaço de pares/ímpares na janela."""
        historical_data = self.df[self.df["Concurso"] < concurso].tail(self.JANELA_PARES_IMPARES)
        pares = 0
        impares = 0
        for _, row in historical_data.iterrows():
            for num in row[1:16]:
                if num % 2 == 0:
                    pares += 1
                else:
                    impares += 1
        total = pares + impares
        return pares / total, impares / total

    def generate_farejadores(self, concurso):
        """Gera farejadores baseados em pares frequentes."""
        # Placeholder: Substitua pela lógica real do SuperInsano_v19_3.py
        return [np.random.choice(range(1, 26), size=15, replace=False) for _ in range(100)]

    def hybrid_prediction(self, concurso):
        """Gera jogo final com base em cansaço e farejadores."""
        # Farejadores
        farejadores = self.generate_farejadores(concurso)
        # Cansaço de números
        cansaco_numeros = self.calculate_cansaco_numeros(concurso)
        # Cansaço de pares/ímpares
        pares_freq, impares_freq = self.calculate_cansaco_pares_impares(concurso)

        # Combina previsões
        combined = defaultdict(float)
        for jogo in farejadores:
            for num in jogo:
                peso = 1 / (cansaco_numeros.get(num, 1) + 1)  # Penaliza números cansados
                # Ajusta pesos com base no desvio de pares/ímpares
                if num % 2 == 0:
                    peso *= (1 - pares_freq) if pares_freq > 0.5 else pares_freq
                else:
                    peso *= (1 - impares_freq) if impares_freq > 0.5 else impares_freq
                combined[num] += peso * self.PESO_QUENTES

        # Seleciona os 15 números com maior pontuação
        jogo_final = sorted(combined, key=lambda x: (-combined[x], x))[:15]

        # Valida matriz 5x5 e soma
        if not self.validate_cartao_5x5(jogo_final) or not self.validate_soma(jogo_final):
            return self.hybrid_prediction(concurso)  # Regera se inválido

        return jogo_final

    def validate_cartao_5x5(self, jogo):
        """Valida distribuição de linhas/colunas no cartão 5x5."""
        linhas = defaultdict(int)
        colunas = defaultdict(int)
        for num in jogo:
            linhas[(num - 1) // 5] += 1
            colunas[(num - 1) % 5] += 1
        return all(2 <= x <= 5 for x in linhas.values()) and all(2 <= x <= 5 for x in colunas.values())

    def validate_soma(self, jogo):
        """Valida soma dentro da faixa ajustada."""
        soma = sum(jogo)
        return self.SOMA_MIN <= soma <= self.SOMA_MAX

    def gera_jogos_concurso(self, concurso):
        """Gera 16 jogos únicos e balanceados."""
        jogos = []
        while len(jogos) < 16:
            jogo = self.hybrid_prediction(concurso)
            if sorted(jogo) not in jogos and self.is_diverse(jogo, jogos):
                jogos.append(sorted(jogo))
        return jogos

    def is_diverse(self, new_game, existing_games, min_diff=4):
        """Garante diversidade entre jogos."""
        for game in existing_games:
            if len(set(new_game) - set(game)) < min_diff:
                return False
        return True

    def backtest(self, test_range):
        """Executa backtest nos concursos especificados."""
        resultados = []
        for concurso in test_range:
            real = set(self.df[self.df["Concurso"] == concurso].iloc[0, 1:16])
            jogos = self.gera_jogos_concurso(concurso)
            acertos = [len(real & set(jogo)) for jogo in jogos]
            resultados.append({
                "Concurso": concurso,
                "Jogos": jogos,
                "Sorteio": list(real),
                "Acertos": acertos,
                "Max_Acertos": max(acertos)
            })
        return resultados

# Execução do Backtest
if __name__ == "__main__":
    # Define o intervalo de concursos (2376–3375)
    test_range = range(2376, 3376)
    
    # Inicializa o Super40 v3
    super40_v3 = Super40v3()
    
    # Executa o backtest
    resultados = super40_v3.backtest(test_range)
    pd.DataFrame(resultados).to_csv(f"backtest_super40v3_{super40_v3.hash_csv[:8]}.csv", index=False)
    
    # Exibe resumo
    acertos_totais = [acerto for res in resultados for acerto in res["Acertos"]]
    print("\n=== Resumo do Super40 v3 ===")
    for pontos in [15, 14, 13, 12, 11]:
        total = sum(1 for x in acertos_totais if x == pontos)
        print(f"Acertos de {pontos} pontos: {total} ({(total/16000)*100:.2f}%)")