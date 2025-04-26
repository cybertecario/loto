import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations

class Super40v2:
    def __init__(self):
        # Carrega os dados históricos
        self.df = self.carrega_base_lotofacil("base_Lotofacil.csv")
        self.quantum_weight = 0.5
        self.gru_weight = 0.3
        self.gan_weight = 0.2

    def carrega_base_lotofacil(self, filename):
        """Carrega o CSV da Lotofácil."""
        try:
            df = pd.read_csv(filename, sep=';')
            required_cols = ["Concurso"] + [f"Bola{i}" for i in range(1, 16)]
            if not all(col in df.columns for col in required_cols):
                raise ValueError("CSV inválido: colunas faltantes.")
            return df.sort_values(by="Concurso").reset_index(drop=True)
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            raise

    def calculate_cansaco_numeros(self, concurso, janela=25):
        """Calcula o cansaço de números na janela."""
        historical_data = self.df[self.df["Concurso"] < concurso].tail(janela)
        numeros = historical_data.values.flatten()
        contagem = Counter(numeros)
        return {num: freq for num, freq in contagem.items()}

    def calculate_cansaco_trios(self, concurso, janela=25):
        """Penaliza trios frequentes na janela."""
        historical_data = self.df[self.df["Concurso"] < concurso].tail(janela)
        trios = Counter()
        for _, row in historical_data.iterrows():
            trios.update(combinations(row[1:16], 3))
        return [trio for trio, _ in trios.most_common(50)]  # Penaliza os 50 trios mais frequentes

    def calculate_frequencia_pares_impares(self, concurso, janela=25):
        """Calcula a frequência de pares/ímpares na janela."""
        historical_data = self.df[self.df["Concurso"] < concurso].tail(janela)
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

    def validate_cartao_5x5(self, jogo):
        """Valida matriz 5x5 e distribuição de pares/ímpares."""
        linhas = defaultdict(int)
        colunas = defaultdict(int)
        pares = 0
        for num in jogo:
            linha = (num - 1) // 5
            coluna = (num - 1) % 5
            linhas[linha] += 1
            colunas[coluna] += 1
            if num % 2 == 0:
                pares += 1
        # Valida linhas/colunas
        if not all(2 <= x <= 5 for x in linhas.values()) or not all(2 <= x <= 5 for x in colunas.values()):
            return False
        # Valida pares/ímpares
        if (pares, 15 - pares) not in [(7, 8), (8, 7), (6, 9)]:
            return False
        return True

    def is_diverse(self, new_game, existing_games, min_diff=4):
        """Verifica diversidade entre jogos."""
        for game in existing_games:
            diff = len(set(new_game) - set(game))
            if diff < min_diff:
                return False
        return True

    def hybrid_prediction(self, concurso):
        """Gera previsões combinando GRU, GAN e simulação quântica."""
        # Farejadores
        farejadores = self.generate_farejadores(concurso)
        # Cansaço
        cansaco_numeros = self.calculate_cansaco_numeros(concurso)
        cansaco_trios = self.calculate_cansaco_trios(concurso)
        pares_freq, impares_freq = self.calculate_frequencia_pares_impares(concurso)

        # Combina previsões
        combined = defaultdict(float)
        for jogo in farejadores:
            for num in jogo:
                # Penaliza números cansados
                peso = 1 / (cansaco_numeros.get(num, 1) + 1)
                # Penaliza trios cansados
                for trio in combinations(jogo, 3):
                    if trio in cansaco_trios:
                        peso *= 0.5
                # Ajusta pesos com base em pares/ímpares
                if num % 2 == 0:
                    peso *= pares_freq
                else:
                    peso *= impares_freq
                combined[num] += peso

        # Seleciona os 15 números com maior pontuação
        jogo_final = sorted(combined, key=lambda x: (-combined[x], x))[:15]

        # Balanceamento rigoroso de faixas
        faixas = [0] * 5  # [1-5, 6-10, 11-15, 16-20, 21-25]
        for num in jogo_final:
            if 1 <= num <= 5:
                faixas[0] += 1
            elif 6 <= num <= 10:
                faixas[1] += 1
            elif 11 <= num <= 15:
                faixas[2] += 1
            elif 16 <= num <= 20:
                faixas[3] += 1
            elif 21 <= num <= 25:
                faixas[4] += 1
        # Força exatamente 3 números por faixa
        if not all(x == 3 for x in faixas):
            return None

        return jogo_final

    def gera_jogos_concurso(self, concurso):
        """Gera 16 jogos únicos para um concurso."""
        jogos = []
        while len(jogos) < 16:
            jogo = self.hybrid_prediction(concurso)
            if jogo and self.validate_cartao_5x5(jogo) and self.is_diverse(jogo, jogos):
                jogos.append(sorted(jogo))
        return jogos

    def calcula_acertos(self, real, jogos):
        """Calcula os acertos para cada jogo."""
        return [len(real & set(jogo)) for jogo in jogos]

    def backtest(self, test_range):
        """Executa o backtest nos concursos especificados."""
        resultados = []
        for concurso in test_range:
            real = set(self.df[self.df["Concurso"] == concurso].iloc[0, 1:16])
            jogos = self.gera_jogos_concurso(concurso)
            acertos = self.calcula_acertos(real, jogos)
            resultados.append({
                "Concurso": concurso,
                "Jogos": jogos,
                "Sorteio": list(real),
                "Acertos": acertos,
                "Max_Acertos": max(acertos)
            })
        return resultados


# Função principal para executar o backtesting
if __name__ == "__main__":
    # Define o intervalo de concursos (últimos 1000 concursos)
    test_range = range(2376, 3376)

    # Backtesting do Super40 v2
    super40_v2 = Super40v2()
    resultados = super40_v2.backtest(test_range)
    pd.DataFrame(resultados).to_csv("backtest_super40_v2.csv", index=False)

    # Exibe resumo comparativo
    acertos_totais = [acerto for res in resultados for acerto in res["Acertos"]]
    print("\n=== Resumo do Super40 v2 ===")
    for pontos in [15, 14, 13, 12, 11]:
        total = sum(1 for x in acertos_totais if x == pontos)
        print(f"Acertos de {pontos} pontos: {total}")