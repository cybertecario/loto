import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import hashlib
import json
import time
from datetime import datetime

# Configurações
NUM_JOGOS = 16
CONCURSO_ATUAL = 3374
JANELA_LONGA = 500
JANELA_CURTA = 20
PESO_LONGA = 0.7
PESO_CURTA = 0.3
JANELA_NORMAIS = 10
JANELA_AZAROES = 50
JANELA_HIBRIDOS = 250
JANELA_50_70 = 300
JANELA_35_55 = 200
JANELA_70_80 = 150
CICLOS_TREINAMENTO = 10
ITERACOES_POR_CICLO = 1000000

estrategias = {
    "FarejadorHibridoCabuloso": {"janela": 1000, "taxa_15_pontos": 0.252, "mistura_quentes": 0.5, "metodo": "hibrido", "max_duplas": 5, "min_frios": 2},
    "FarejadorDeAzarao": {"janela": 500, "taxa_15_pontos": 0.105, "mistura_quentes": 0.3, "metodo": "azarao_sequencias", "max_duplas": 6, "min_frios": 5},
    "NovoFarejadorDeAzarao": {"janela": 500, "taxa_15_pontos": 0.102, "mistura_quentes": 0.3, "metodo": "azarao_padroes", "max_duplas": 6, "min_frios": 5},
    "TurboFarejador": {"janela": 200, "taxa_15_pontos": 0.127, "mistura_quentes": 0.4, "metodo": "turbo", "max_duplas": 5, "min_frios": 2},
    "FarejadorDeRedesBayesianas": {"janela": 300, "taxa_15_pontos": 0.117, "mistura_quentes": 0.3, "metodo": "bayes", "max_duplas": 5, "min_frios": 2},
    "FarejadorDePadroesGeometricos": {"janela": 300, "taxa_15_pontos": 0.168, "mistura_quentes": 0.4, "metodo": "geometrico_50_70", "max_duplas": 5, "min_frios": 3, "pares_impares": {"min_pares": 5, "max_pares": 6, "alt_pares": 9}, "soma": {"min": 140, "max": 250}},
    "FarejadorDeRedesNeurais": {"janela": 200, "taxa_15_pontos": 0.152, "mistura_quentes": 0.3, "metodo": "neural_35_55", "max_duplas": 6, "min_frios": 5, "pares_impares": {"min_pares": 4, "max_pares": 5, "alt_pares": 10}, "soma": {"min": 130, "max": 260}},
    "FarejadorUltraMegaMonstro": {"janela": 300, "taxa_15_pontos": 0.115, "mistura_quentes": 0.5, "metodo": "hibrido_borda", "max_duplas": 5, "min_frios": 4, "pares_impares": {"min_pares": 5, "max_pares": 6, "alt_pares": 9}, "soma": {"min": 150, "max": 240}},
    "FarejadorDeCiclosSazonais": {"janela": 200, "taxa_15_pontos": 0.113, "mistura_quentes": 0.5, "metodo": "hibrido_geometrico", "max_duplas": 5, "min_frios": 3, "pares_impares": {"min_pares": 5, "max_pares": 5, "alt_pares": 10}, "soma": {"min": 150, "max": 240}},
    "FarejadorDeNormaisLeves": {"janela": 200, "taxa_15_pontos": 0.125, "mistura_quentes": 0.5, "metodo": "normais_70_80", "max_duplas": 5, "min_frios": 2, "pares_impares": {"min_pares": 5, "max_pares": 6, "alt_pares": 10}, "soma": {"min": 160, "max": 230}},
    "FarejadorNeuralAvancado": {"janela": 1000, "taxa_15_pontos": 0.248, "mistura_quentes": 0.5, "metodo": "neural_avancado", "max_duplas": 5, "min_frios": 2}
}

AZAROES_FIXOS = ["FarejadorDeAzarao", "NovoFarejadorDeAzarao"]
HIBRIDOS = ["FarejadorUltraMegaMonstro", "FarejadorDeCiclosSazonais"]
FAIXA_50_70 = ["FarejadorDePadroesGeometricos"]
FAIXA_35_55 = ["FarejadorDeRedesNeurais"]
FAIXA_70_80 = ["FarejadorDeNormaisLeves"]
NEURAL = ["FarejadorNeuralAvancado"]
NORMAIS = [est for est in estrategias.keys() if est not in AZAROES_FIXOS and est not in HIBRIDOS and est not in FAIXA_50_70 and est not in FAIXA_35_55 and est not in FAIXA_70_80 and est not in NEURAL]

# Carregar o histórico (simulado)
historico = pd.read_csv("base_Lotofacil.csv")
ultimo_concurso = historico["Concurso"].max()

# Funções auxiliares
def calcular_sha256(jogo, estrategia, concurso, timestamp):
    jogo_str = f"{sorted(jogo)}-{estrategia}-{concurso}-{timestamp}"
    return hashlib.sha256(jogo_str.encode()).hexdigest()

def contar_pares_impares(jogo):
    impares = sum(1 for num in jogo if num % 2 != 0)
    pares = len(jogo) - impares
    return impares, pares

def contar_consecutivos(jogo):
    jogo = sorted(jogo)
    max_consecutivos = 1
    atual = 1
    for i in range(len(jogo) - 1):
        if jogo[i + 1] == jogo[i] + 1:
            atual += 1
            max_consecutivos = max(max_consecutivos, atual)
        else:
            atual = 1
    return max_consecutivos

def contar_sequencias_pares_impares(jogo):
    jogo = sorted(jogo)
    max_seq_pares = max_seq_impares = 1
    atual_pares = atual_impares = 1
    for i in range(len(jogo) - 1):
        if jogo[i] % 2 == 0 and jogo[i + 1] % 2 == 0 and jogo[i + 1] == jogo[i] + 2:
            atual_pares += 1
            max_seq_pares = max(max_seq_pares, atual_pares)
        elif jogo[i] % 2 != 0 and jogo[i + 1] % 2 != 0 and jogo[i + 1] == jogo[i] + 2:
            atual_impares += 1
            max_seq_impares = max(max_seq_impares, atual_impares)
        else:
            atual_pares = 1 if jogo[i + 1] % 2 == 0 else atual_pares
            atual_impares = 1 if jogo[i + 1] % 2 != 0 else atual_impares
    return max_seq_pares, max_seq_impares

def contar_primos_fibonacci(jogo):
    primos = {2, 3, 5, 7, 11, 13, 17, 19, 23}
    fibonacci = {1, 2, 3, 5, 8, 13, 21}
    num_primos = sum(1 for num in jogo if num in primos)
    num_fibonacci = sum(1 for num in jogo if num in fibonacci)
    return num_primos, num_fibonacci

def contar_moldura_miolo(jogo):
    moldura = {1, 2, 3, 4, 5, 6, 10, 11, 15, 16, 20, 21, 22, 23, 24, 25}
    num_moldura = sum(1 for num in jogo if num in moldura)
    num_miolo = len(jogo) - num_moldura
    return num_moldura, num_miolo

def calcular_soma(jogo):
    return sum(jogo)

def contar_repetidos_concurso_anterior(jogo, historico, concurso_atual):
    if concurso_atual <= 2374:
        return 0
    anterior = historico[historico["Concurso"] == concurso_atual - 1].iloc[0, 1:16]
    return len(set(jogo) & set(anterior))

def calcular_normalidade(sorteio, historico, concurso_atual, freq):
    nums = sorted(sorteio[1:16] if isinstance(sorteio, pd.Series) else sorteio)
    impares, pares = contar_pares_impares(nums)
    max_consecutivos = contar_consecutivos(nums)
    max_seq_pares, max_seq_impares = contar_sequencias_pares_impares(nums)
    num_primos, num_fibonacci = contar_primos_fibonacci(nums)
    num_moldura, num_miolo = contar_moldura_miolo(nums)
    soma = calcular_soma(nums)
    repetidos = contar_repetidos_concurso_anterior(nums, historico, concurso_atual)
    num_frios = sum(1 for num in nums if num in freq.tail(10).index)
    
    pesos_normais = {
        "pares_6_9": 0.2 if 6 <= pares <= 9 else 0,
        "impares_6_9": 0.2 if 6 <= impares <= 9 else 0,
        "consecutivos_5": 0.15 if max_consecutivos <= 5 else 0,
        "seq_pares_5": 0.1 if max_seq_pares <= 5 else 0,
        "seq_impares_5": 0.1 if max_seq_impares <= 5 else 0,
        "frios_2_4": 0.1 if 2 <= num_frios <= 4 else 0,
        "soma_171_220": 0.1 if 171 <= soma <= 220 else 0,
        "primos_3_6": 0.05 if 3 <= num_primos <= 6 else 0,
        "fibonacci_3_5": 0.05 if 3 <= num_fibonacci <= 5 else 0,
        "repetidos_7_10": 0.05 if 7 <= repetidos <= 10 else 0,
        "moldura_9_11": 0.05 if 9 <= num_moldura <= 11 else 0,
        "miolo_4_6": 0.05 if 4 <= num_miolo <= 6 else 0
    }
    pesos_azaroes = {
        "pares_extremo": 0.2 if pares <= 5 or pares >= 10 else 0,
        "consecutivos_6": 0.15 if max_consecutivos >= 6 else 0,
        "seq_pares_6": 0.1 if max_seq_pares >= 6 else 0,
        "seq_impares_6": 0.1 if max_seq_impares >= 6 else 0,
        "frios_5": 0.1 if num_frios >= 5 else 0,
        "soma_extremo": 0.1 if soma < 150 or soma > 240 else 0,
        "primos_extremo": 0.05 if num_primos >= 7 or num_primos < 2 else 0,
        "fibonacci_extremo": 0.05 if num_fibonacci >= 6 or num_fibonacci < 2 else 0,
        "repetidos_extremo": 0.05 if repetidos <= 6 or repetidos >= 11 else 0,
        "moldura_12": 0.05 if num_moldura >= 12 else 0,
        "miolo_7": 0.05 if num_miolo >= 7 else 0
    }
    score_normal = sum(pesos_normais.values())
    score_azarao = sum(pesos_azaroes.values())
    normalidade = ((score_normal - score_azarao + 1) / 2) * 100
    return normalidade

def prever_faixa_hibrido(historico, concurso_atual):
    inicio = max(2374, concurso_atual - 10)
    dados = historico[(historico["Concurso"] >= inicio) & (historico["Concurso"] < concurso_atual)]
    features = []
    
    for _, sorteio in dados.iterrows():
        nums = sorted(sorteio[1:16])
        impares, pares = contar_pares_impares(nums)
        max_consecutivos = contar_consecutivos(nums)
        max_seq_pares, max_seq_impares = contar_sequencias_pares_impares(nums)
        num_primos, num_fibonacci = contar_primos_fibonacci(nums)
        num_moldura, num_miolo = contar_moldura_miolo(nums)
        soma = calcular_soma(nums)
        repetidos = contar_repetidos_concurso_anterior(nums, historico, sorteio["Concurso"])
        freq = pd.Series([num for bola in range(1, 16) for num in dados[f"Bola{bola}"]]).value_counts()
        num_frios = sum(1 for num in nums if num in freq.tail(10).index)
        normalidade = calcular_normalidade(sorteio, historico, sorteio["Concurso"], freq)
        
        feature = [
            1 if pares == 5 or pares >= 9 else 0,
            1 if max_consecutivos >= 5 else 0,
            1 if max_seq_pares >= 5 or max_seq_impares >= 5 else 0,
            1 if num_frios >= 4 else 0,
            1 if 150 <= soma <= 170 or 220 <= soma <= 240 else 0,
            1 if num_primos >= 6 or num_fibonacci >= 5 else 0,
            1 if num_moldura >= 11 or num_miolo >= 6 else 0,
            normalidade / 100
        ]
        features.append(feature)
    
    X = np.mean(features, axis=0) if features else np.zeros(8)
    X_train = np.random.rand(900, 8)
    y_train = np.random.choice([0, 1], 900, p=[0.75, 0.25])
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    prob = model.predict_proba([X])[0][1]
    return prob

def prever_faixa_50_70(historico, concurso_atual):
    inicio = max(2374, concurso_atual - 10)
    dados = historico[(historico["Concurso"] >= inicio) & (historico["Concurso"] < concurso_atual)]
    features = []
    
    for _, sorteio in dados.iterrows():
        nums = sorted(sorteio[1:16])
        impares, pares = contar_pares_impares(nums)
        max_consecutivos = contar_consecutivos(nums)
        max_seq_pares, max_seq_impares = contar_sequencias_pares_impares(nums)
        num_primos, num_fibonacci = contar_primos_fibonacci(nums)
        num_moldura, num_miolo = contar_moldura_miolo(nums)
        soma = calcular_soma(nums)
        repetidos = contar_repetidos_concurso_anterior(nums, historico, sorteio["Concurso"])
        freq = pd.Series([num for bola in range(1, 16) for num in dados[f"Bola{bola}"]]).value_counts()
        num_frios = sum(1 for num in nums if num in freq.tail(10).index)
        normalidade = calcular_normalidade(sorteio, historico, sorteio["Concurso"], freq)
        
        feature = [
            1 if pares in [5, 6, 9, 10] else 0,
            1 if max_consecutivos in [5, 6] else 0,
            1 if max_seq_pares >= 5 or max_seq_impares >= 5 else 0,
            1 if num_frios in [3, 4] else 0,
            1 if 140 <= soma <= 150 or 240 <= soma <= 250 else 0,
            1 if num_primos in [5, 6] or num_fibonacci in [4, 5] else 0,
            1 if num_moldura in [10, 11] or num_miolo in [5, 6] else 0,
            normalidade / 100
        ]
        features.append(feature)
    
    X = np.mean(features, axis=0) if features else np.zeros(8)
    X_train = np.random.rand(900, 8)
    y_train = np.random.choice([0, 1], 900, p=[0.7, 0.3])
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    prob = model.predict_proba([X])[0][1]
    return prob

def prever_faixa_35_55(historico, concurso_atual):
    inicio = max(2374, concurso_atual - 10)
    dados = historico[(historico["Concurso"] >= inicio) & (historico["Concurso"] < concurso_atual)]
    features = []
    
    for _, sorteio in dados.iterrows():
        nums = sorted(sorteio[1:16])
        impares, pares = contar_pares_impares(nums)
        max_consecutivos = contar_consecutivos(nums)
        max_seq_pares, max_seq_impares = contar_sequencias_pares_impares(nums)
        num_primos, num_fibonacci = contar_primos_fibonacci(nums)
        num_moldura, num_miolo = contar_moldura_miolo(nums)
        soma = calcular_soma(nums)
        repetidos = contar_repetidos_concurso_anterior(nums, historico, sorteio["Concurso"])
        freq = pd.Series([num for bola in range(1, 16) for num in dados[f"Bola{bola}"]]).value_counts()
        num_frios = sum(1 for num in nums if num in freq.tail(10).index)
        normalidade = calcular_normalidade(sorteio, historico, sorteio["Concurso"], freq)
        
        feature = [
            1 if pares <= 5 or pares >= 10 else 0,
            1 if max_consecutivos >= 6 else 0,
            1 if max_seq_pares >= 6 or max_seq_impares >= 6 else 0,
            1 if num_frios >= 5 else 0,
            1 if soma < 140 or soma > 250 else 0,
            1 if num_primos >= 7 or num_primos < 2 or num_fibonacci >= 6 or num_fibonacci < 2 else 0,
            1 if num_moldura >= 12 or num_miolo >= 7 else 0,
            normalidade / 100
        ]
        features.append(feature)
    
    X = np.mean(features, axis=0) if features else np.zeros(8)
    X_train = np.random.rand(900, 8)
    y_train = np.random.choice([0, 1], 900, p=[0.8, 0.2])
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    prob = model.predict_proba([X])[0][1]
    return prob

def prever_faixa_70_80(historico, concurso_atual):
    inicio = max(2374, concurso_atual - 10)
    dados = historico[(historico["Concurso"] >= inicio) & (historico["Concurso"] < concurso_atual)]
    features = []
    
    for _, sorteio in dados.iterrows():
        nums = sorted(sorteio[1:16])
        impares, pares = contar_pares_impares(nums)
        max_consecutivos = contar_consecutivos(nums)
        max_seq_pares, max_seq_impares = contar_sequencias_pares_impares(nums)
        num_primos, num_fibonacci = contar_primos_fibonacci(nums)
        num_moldura, num_miolo = contar_moldura_miolo(nums)
        soma = calcular_soma(nums)
        repetidos = contar_repetidos_concurso_anterior(nums, historico, sorteio["Concurso"])
        freq = pd.Series([num for bola in range(1, 16) for num in dados[f"Bola{bola}"]]).value_counts()
        num_frios = sum(1 for num in nums if num in freq.tail(10).index)
        normalidade = calcular_normalidade(sorteio, historico, sorteio["Concurso"], freq)
        
        feature = [
            1 if pares in [5, 10] else 0,
            1 if max_consecutivos == 5 else 0,
            1 if max_seq_pares == 5 or max_seq_impares == 5 else 0,
            1 if num_frios == 3 else 0,
            1 if 160 <= soma <= 170 or 220 <= soma <= 230 else 0,
            1 if num_primos == 5 or num_fibonacci == 4 else 0,
            1 if num_moldura == 10 or num_miolo == 5 else 0,
            normalidade / 100
        ]
        features.append(feature)
    
    X = np.mean(features, axis=0) if features else np.zeros(8)
    X_train = np.random.rand(900, 8)
    y_train = np.random.choice([0, 1], 900, p=[0.85, 0.15])
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    prob = model.predict_proba([X])[0][1]
    return prob

def prever_faixa_azarao(historico, concurso_atual):
    inicio = max(2374, concurso_atual - 10)
    dados = historico[(historico["Concurso"] >= inicio) & (historico["Concurso"] < concurso_atual)]
    features = []
    
    for _, sorteio in dados.iterrows():
        nums = sorted(sorteio[1:16])
        max_consecutivos = contar_consecutivos(nums)
        max_seq_pares, max_seq_impares = contar_sequencias_pares_impares(nums)
        num_primos, num_fibonacci = contar_primos_fibonacci(nums)
        num_moldura, num_miolo = contar_moldura_miolo(nums)
        soma = calcular_soma(nums)
        repetidos = contar_repetidos_concurso_anterior(nums, historico, sorteio["Concurso"])
        freq = pd.Series([num for bola in range(1, 16) for num in dados[f"Bola{bola}"]]).value_counts()
        num_frios = sum(1 for num in nums if num in freq.tail(10).index)
        normalidade = calcular_normalidade(sorteio, historico, sorteio["Concurso"], freq)
        
        feature = [
            1 if num_frios >= 5 else 0,
            1 if max_consecutivos >= 6 else 0,
            1 if max_seq_pares >= 6 or max_seq_impares >= 6 else 0,
            1 if soma < 150 or soma > 240 else 0,
            1 if num_primos >= 7 or num_primos < 2 else 0,
            1 if num_moldura >= 12 or num_miolo >= 7 else 0,
            1 if repetidos <= 6 or repetidos >= 11 else 0,
            normalidade / 100
        ]
        features.append(feature)
    
    X = np.mean(features, axis=0) if features else np.zeros(8)
    X_train = np.random.rand(900, 8)
    y_train = np.random.choice([0, 1], 900, p=[0.8, 0.2])
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    prob = model.predict_proba([X])[0][1]
    return prob

def treinar_rede_neural(historico):
    X_train = []
    y_train = []
    
    for concurso in range(2374, 3274):
        inicio = max(2374, concurso - 10)
        dados = historico[(historico["Concurso"] >= inicio) & (historico["Concurso"] < concurso)]
        features = []
        for _, sorteio in dados.iterrows():
            nums = sorted(sorteio[1:16])
            impares, pares = contar_pares_impares(nums)
            max_consecutivos = contar_consecutivos(nums)
            max_seq_pares, max_seq_impares = contar_sequencias_pares_impares(nums)
            num_primos, num_fibonacci = contar_primos_fibonacci(nums)
            num_moldura, num_miolo = contar_moldura_miolo(nums)
            soma = calcular_soma(nums)
            repetidos = contar_repetidos_concurso_anterior(nums, historico, sorteio["Concurso"])
            freq = pd.Series([num for bola in range(1, 16) for num in dados[f"Bola{bola}"]]).value_counts()
            num_frios = sum(1 for num in nums if num in freq.tail(10).index)
            normalidade = calcular_normalidade(sorteio, historico, sorteio["Concurso"], freq)
            
            feature = [
                1 if pares in [5, 6, 9, 10] else 0,
                1 if max_consecutivos in [5, 6] else 0,
                1 if max_seq_pares >= 5 or max_seq_impares >= 5 else 0,
                1 if num_frios in [3, 4] else 0,
                1 if 140 <= soma <= 150 or 240 <= soma <= 250 else 0,
                1 if num_primos in [5, 6] or num_fibonacci in [4, 5] else 0,
                1 if num_moldura in [10, 11] or num_miolo in [5, 6] else 0,
                normalidade / 100
            ]
            features.append(feature)
        
        X_train.append(np.mean(features, axis=0) if features else np.zeros(8))
        sorteio = historico[historico["Concurso"] == concurso].iloc[0, 1:16]
        y = np.zeros(25)
        for num in sorteio:
            y[num-1] = 1
        y_train.append(y)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    model = Sequential([
        Dense(256, activation='relu', input_shape=(8,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(25, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)
    
    return model

neural_model = treinar_rede_neural(historico)

def aplicar_regras(jogo, freq, estrategia, jogos_gerados, historico, concurso_atual):
    impares, pares = contar_pares_impares(jogo)
    max_consecutivos = contar_consecutivos(jogo)
    max_seq_pares, max_seq_impares = contar_sequencias_pares_impares(jogo)
    num_primos, num_fibonacci = contar_primos_fibonacci(jogo)
    num_moldura, num_miolo = contar_moldura_miolo(jogo)
    soma = calcular_soma(jogo)
    repetidos = contar_repetidos_concurso_anterior(jogo, historico, concurso_atual)
    
    if max_consecutivos >= 8 or max_seq_pares >= 8 or max_seq_impares >= 8:
        return False
    if soma < 100 or soma > 270:
        return False
    if num_primos == 0 or num_fibonacci == 0:
        return False
    
    if estrategia in FAIXA_50_70:
        condicoes_50_70 = [
            pares in [5, 6, 9, 10],
            max_consecutivos in [5, 6],
            sum(1 for num in jogo if num in freq.tail(10).index) in [3, 4],
            140 <= soma <= 150 or 240 <= soma <= 250,
            num_primos in [5, 6] or num_fibonacci in [4, 5],
            repetidos in [6, 7, 10, 11],
            num_moldura in [10, 11] or num_miolo in [5, 6]
        ]
        if sum(condicoes_50_70) < 1 or sum(condicoes_50_70) > 3:
            return False
        if not (estrategias[estrategia]["pares_impares"]["min_pares"] <= pares <= estrategias[estrategia]["pares_impares"]["max_pares"] or
                pares >= estrategias[estrategia]["pares_impares"]["alt_pares"]):
            return False
        if num_moldura < 10 and num_miolo < 5:
            return False
    
    elif estrategia in FAIXA_35_55:
        condicoes_35_55 = [
            pares <= 5 or pares >= 10,
            max_consecutivos >= 6,
            sum(1 for num in jogo if num in freq.tail(10).index) >= estrategias[estrategia]["min_frios"],
            soma < 140 or soma > 250,
            num_primos >= 7 or num_primos < 2 or num_fibonacci >= 6 or num_fibonacci < 2,
            repetidos <= 6 or repetidos >= 11,
            num_moldura >= 12 or num_miolo >= 7
        ]
        if sum(condicoes_35_55) < 1 or sum(condicoes_35_55) > 2:
            return False
        if not (estrategias[estrategia]["pares_impares"]["min_pares"] <= pares <= estrategias[estrategia]["pares_impares"]["max_pares"] or
                pares >= estrategias[estrategia]["pares_impares"]["alt_pares"]):
            return False
    
    elif estrategia in FAIXA_70_80:
        condicoes_70_80 = [
            pares in [5, 10],
            max_consecutivos == 5,
            sum(1 for num in jogo if num in freq.tail(10).index) == 3,
            160 <= soma <= 170 or 220 <= soma <= 230,
            num_primos == 5 or num_fibonacci == 4,
            repetidos in [6, 11],
            num_moldura == 10 or num_miolo == 5
        ]
        if sum(condicoes_70_80) < 1 or sum(condicoes_70_80) > 2:
            return False
        if not (estrategias[estrategia]["pares_impares"]["min_pares"] <= pares <= estrategias[estrategia]["pares_impares"]["max_pares"] or
                pares >= estrategias[estrategia]["pares_impares"]["alt_pares"]):
            return False
    
    elif estrategia in HIBRIDOS:
        condicoes_hibrido = [
            pares == 5 or pares >= 9,
            max_consecutivos >= estrategias[estrategia].get("max_consecutivos", 5),
            max_seq_pares >= 5 or max_seq_impares >= 5,
            sum(1 for num in jogo if num in freq.tail(10).index) >= estrategias[estrategia]["min_frios"],
            soma >= estrategias[estrategia]["soma"]["min"] and soma <= estrategias[estrategia]["soma"]["max"],
            num_primos >= 6 or num_fibonacci >= 5 or num_primos <= 2 or num_fibonacci <= 2,
            repetidos == 6 or repetidos >= 11,
            num_moldura >= 11 or num_miolo >= 6
        ]
        if sum(condicoes_hibrido) < 1 or sum(condicoes_hibrido) > 2:
            return False
    
    elif estrategia in AZAROES_FIXOS:
        condicoes_azarao = [
            pares <= 5 or pares >= 10,
            max_consecutivos >= 6,
            max_seq_pares >= 6 or max_seq_impares >= 6,
            sum(1 for num in jogo if num in freq.tail(10).index) >= estrategias[estrategia]["min_frios"],
            soma < 150 or soma > 240,
            num_primos >= 7 or num_fibonacci >= 6 or num_primos < 2 or num_fibonacci < 2,
            repetidos <= 6 or repetidos >= 11,
            num_moldura >= 12 or num_miolo >= 7
        ]
        if sum(condicoes_azarao) < 2:
            return False
    
    return True

def gerar_jogo_estrategia(historico, freq, estrategia, jogos_gerados, concurso_atual):
    tentativa = 0
    max_tentativas = 1000
    metodo = estrategias[estrategia]["metodo"]
    
    while tentativa < max_tentativas:
        if metodo == "neural_avancado":
            inicio = max(2374, concurso_atual - 10)
            dados = historico[(historico["Concurso"] >= inicio) & (historico["Concurso"] < concurso_atual)]
            features = []
            for _, sorteio in dados.iterrows():
                nums = sorted(sorteio[1:16])
                impares, pares = contar_pares_impares(nums)
                max_consecutivos = contar_consecutivos(nums)
                max_seq_pares, max_seq_impares = contar_sequencias_pares_impares(nums)
                num_primos, num_fibonacci = contar_primos_fibonacci(nums)
                num_moldura, num_miolo = contar_moldura_miolo(nums)
                soma = calcular_soma(nums)
                repetidos = contar_repetidos_concurso_anterior(nums, historico, sorteio["Concurso"])
                freq_temp = pd.Series([num for bola in range(1, 16) for num in dados[f"Bola{bola}"]]).value_counts()
                num_frios = sum(1 for num in nums if num in freq_temp.tail(10).index)
                normalidade = calcular_normalidade(sorteio, historico, sorteio["Concurso"], freq_temp)
                
                feature = [
                    1 if pares in [5, 6, 9, 10] else 0,
                    1 if max_consecutivos in [5, 6] else 0,
                    1 if max_seq_pares >= 5 or max_seq_impares >= 5 else 0,
                    1 if num_frios in [3, 4] else 0,
                    1 if 140 <= soma <= 150 or 240 <= soma <= 250 else 0,
                    1 if num_primos in [5, 6] or num_fibonacci in [4, 5] else 0,
                    1 if num_moldura in [10, 11] or num_miolo in [5, 6] else 0,
                    normalidade / 100
                ]
                features.append(feature)
            
            X = np.mean(features, axis=0) if features else np.zeros(8)
            probs = neural_model.predict(np.array([X]), verbose=0)[0]
            jogo = sorted(np.argsort(probs)[-15:]) + 1
            jogo = [int(x) for x in jogo]
        else:
            quentes = freq.head(15).index.tolist()
            frios = freq.tail(10).index.tolist()
            mistura_quentes = estrategias[estrategia]["mistura_quentes"]
            num_quentes = int(15 * mistura_quentes)
            num_frios = max(estrategias[estrategia]["min_frios"], int(np.random.randint(2, 6)))
            num_restantes = 15 - num_quentes - num_frios
            
            candidatos_quentes = np.random.choice(quentes, num_quentes, replace=False)
            candidatos_frios = np.random.choice(frios, num_frios, replace=False)
            restantes = [x for x in range(1, 26) if x not in candidatos_quentes and x not in candidatos_frios]
            candidatos_restantes = np.random.choice(restantes, num_restantes, replace=False)
            
            jogo = sorted(list(candidatos_quentes) + list(candidatos_frios) + list(candidatos_restantes))
        
        if len(jogo) == 15 and set(jogo) not in [set(j) for j in jogos_gerados]:
            if aplicar_regras(jogo, freq, estrategia, jogos_gerados, historico, concurso_atual):
                return jogo
        tentativa += 1
    
    return None

def verificar_cobertura_numeros(jogos):
    todos_numeros = set()
    for jogo in jogos:
        todos_numeros.update(jogo)
    return todos_numeros

def ajustar_cobertura(jogo, numeros_faltantes, freq, estrategia, jogos_gerados, historico, concurso_atual):
    tentativa = 0
    max_tentativas = 100
    while tentativa < max_tentativas:
        novos_numeros = np.random.choice(list(numeros_faltantes), min(len(numeros_faltantes), 2), replace=False)
        jogo_temp = jogo.copy()
        for _ in range(len(novos_numeros)):
            idx = np.random.randint(0, 15)
            while jogo_temp[idx] in novos_numeros:
                idx = np.random.randint(0, 15)
            jogo_temp[idx] = novos_numeros[_]
        jogo_temp = sorted(jogo_temp)
        if aplicar_regras(jogo_temp, freq, estrategia, jogos_gerados, historico, concurso_atual):
            return jogo_temp
        tentativa += 1
    return jogo

def distribuir_jogos(historico, concurso_atual):
    prob_hibrido = prever_faixa_hibrido(historico, concurso_atual)
    prob_50_70 = prever_faixa_50_70(historico, concurso_atual)
    prob_35_55 = prever_faixa_35_55(historico, concurso_atual)
    prob_70_80 = prever_faixa_70_80(historico, concurso_atual)
    prob_azarao = prever_faixa_azarao(historico, concurso_atual)
    
    probs = {
        "65-85%": prob_hibrido,
        "50-70%": prob_50_70,
        "35-55%": prob_35_55,
        "70-80%": prob_70_80,
        "Azarões": prob_azarao
    }
    
    faixa_mais_provavel = max(probs, key=probs.get)
    probs.pop(faixa_mais_provavel)
    faixa_segunda_provavel = max(probs, key=probs.get)
    
    jogos_por_estrategia = {est: 0 for est in estrategias}
    jogos_por_estrategia["FarejadorNeuralAvancado"] = 2
    jogos_por_estrategia["FarejadorHibridoCabuloso"] = 2
    
    if faixa_mais_provavel == "65-85%":
        jogos_por_estrategia["FarejadorUltraMegaMonstro"] += 2
        jogos_por_estrategia["FarejadorDeCiclosSazonais"] += 1
    elif faixa_mais_provavel == "50-70%":
        jogos_por_estrategia["FarejadorDePadroesGeometricos"] += 3
    elif faixa_mais_provavel == "35-55%":
        jogos_por_estrategia["FarejadorDeRedesNeurais"] += 3
    elif faixa_mais_provavel == "70-80%":
        jogos_por_estrategia["FarejadorDeNormaisLeves"] += 3
    elif faixa_mais_provavel == "Azarões":
        jogos_por_estrategia["FarejadorDeAzarao"] += 2
        jogos_por_estrategia["NovoFarejadorDeAzarao"] += 1
    
    if faixa_segunda_provavel == "65-85%":
        jogos_por_estrategia["FarejadorUltraMegaMonstro"] += 1
        jogos_por_estrategia["FarejadorDeCiclosSazonais"] += 1
    elif faixa_segunda_provavel == "50-70%":
        jogos_por_estrategia["FarejadorDePadroesGeometricos"] += 2
    elif faixa_segunda_provavel == "35-55%":
        jogos_por_estrategia["FarejadorDeRedesNeurais"] += 2
    elif faixa_segunda_provavel == "70-80%":
        jogos_por_estrategia["FarejadorDeNormaisLeves"] += 2
    elif faixa_segunda_provavel == "Azarões":
        jogos_por_estrategia["FarejadorDeAzarao"] += 1
        jogos_por_estrategia["NovoFarejadorDeAzarao"] += 1
    
    faixas_cobertas = set()
    if jogos_por_estrategia["FarejadorUltraMegaMonstro"] > 0 or jogos_por_estrategia["FarejadorDeCiclosSazonais"] > 0:
        faixas_cobertas.add("65-85%")
    if jogos_por_estrategia["FarejadorDePadroesGeometricos"] > 0:
        faixas_cobertas.add("50-70%")
    if jogos_por_estrategia["FarejadorDeRedesNeurais"] > 0:
        faixas_cobertas.add("35-55%")
    if jogos_por_estrategia["FarejadorDeNormaisLeves"] > 0:
        faixas_cobertas.add("70-80%")
    if jogos_por_estrategia["FarejadorDeAzarao"] > 0 or jogos_por_estrategia["NovoFarejadorDeAzarao"] > 0:
        faixas_cobertas.add("Azarões")
    
    faixas_nao_cobertas = {"65-85%", "50-70%", "35-55%", "70-80%", "Azarões"} - faixas_cobertas
    for faixa in faixas_nao_cobertas:
        if faixa == "65-85%":
            jogos_por_estrategia["FarejadorUltraMegaMonstro"] += 1
        elif faixa == "50-70%":
            jogos_por_estrategia["FarejadorDePadroesGeometricos"] += 1
        elif faixa == "35-55%":
            jogos_por_estrategia["FarejadorDeRedesNeurais"] += 1
        elif faixa == "70-80%":
            jogos_por_estrategia["FarejadorDeNormaisLeves"] += 1
        elif faixa == "Azarões":
            jogos_por_estrategia["FarejadorDeAzarao"] += 1
    
    total_jogos = sum(jogos_por_estrategia.values())
    restantes = NUM_JOGOS - total_jogos
    if restantes > 0:
        estrategias_restantes = [est for est in estrategias if est not in ["FarejadorNeuralAvancado", "FarejadorHibridoCabuloso"]]
        total_taxas = sum(estrategias[est]["taxa_15_pontos"] for est in estrategias_restantes)
        for est in estrategias_restantes:
            jogos_por_estrategia[est] += round((estrategias[est]["taxa_15_pontos"] / total_taxas) * restantes) if total_taxas > 0 else 0
    
    total_jogos = sum(jogos_por_estrategia.values())
    if total_jogos < NUM_JOGOS:
        est_max = max(estrategias, key=lambda x: estrategias[x]["taxa_15_pontos"])
        jogos_por_estrategia[est_max] += NUM_JOGOS - total_jogos
    elif total_jogos > NUM_JOGOS:
        est_min = min(estrategias, key=lambda x: jogos_por_estrategia[x])
        jogos_por_estrategia[est_min] -= total_jogos - NUM_JOGOS
    
    return jogos_por_estrategia, probs

def auditar_jogo(jogo, estrategia, freq, probs, historico, concurso_atual, timestamp):
    impares, pares = contar_pares_impares(jogo)
    max_consecutivos = contar_consecutivos(jogo)
    max_seq_pares, max_seq_impares = contar_sequencias_pares_impares(jogo)
    num_primos, num_fibonacci = contar_primos_fibonacci(jogo)
    num_moldura, num_miolo = contar_moldura_miolo(jogo)
    soma = calcular_soma(jogo)
    repetidos = contar_repetidos_concurso_anterior(jogo, historico, concurso_atual)
    num_frios = sum(1 for num in jogo if num in freq.tail(10).index)
    normalidade = calcular_normalidade(jogo, historico, concurso_atual, freq)
    
    auditoria = {
        "jogo": jogo,
        "estrategia": estrategia,
        "concurso": concurso_atual,
        "timestamp": timestamp,
        "sha256": calcular_sha256(jogo, estrategia, concurso_atual, timestamp),
        "probs_faixas": probs,
        "regras": {
            "pares": pares,
            "impares": impares,
            "consecutivos": max_consecutivos,
            "seq_pares": max_seq_pares,
            "seq_impares": max_seq_impares,
            "primos": num_primos,
            "fibonacci": num_fibonacci,
            "moldura": num_moldura,
            "miolo": num_miolo,
            "soma": soma,
            "repetidos": repetidos,
            "frios": num_frios,
            "normalidade": normalidade
        },
        "valido": aplicar_regras(jogo, freq, estrategia, [], historico, concurso_atual)
    }
    return auditoria

def gerar_jogos(historico, concurso_atual):
    freq = pd.Series([num for _, row in historico.iterrows() for num in row[1:16]]).value_counts()
    jogos_por_estrategia, probs = distribuir_jogos(historico, concurso_atual)
    jogos_gerados = []
    auditorias = []
    
    for estrategia, num_jogos in jogos_por_estrategia.items():
        for _ in range(num_jogos):
            jogo = gerar_jogo_estrategia(historico, freq, estrategia, jogos_gerados, concurso_atual)
            if jogo:
                timestamp = datetime.now().isoformat()
                auditoria = auditar_jogo(jogo, estrategia, freq, probs, historico, concurso_atual, timestamp)
                if auditoria["valido"]:
                    jogos_gerados.append(jogo)
                    auditorias.append(auditoria)
    
    todos_numeros = verificar_cobertura_numeros(jogos_gerados)
    numeros_faltantes = set(range(1, 26)) - todos_numeros
    if numeros_faltantes:
        ultima_estrategia = list(jogos_por_estrategia.keys())[-1]
        jogo_ajustado = ajustar_cobertura(jogos_gerados[-1], numeros_faltantes, freq, ultima_estrategia, jogos_gerados[:-1], historico, concurso_atual)
        jogos_gerados[-1] = jogo_ajustado
        auditorias[-1] = auditar_jogo(jogo_ajustado, ultima_estrategia, freq, probs, historico, concurso_atual, datetime.now().isoformat())
    
    with open(f"auditoria_concurso_{concurso_atual}.json", "w") as f:
        json.dump(auditorias, f, indent=4)
    
    with open(f"jogos_concurso_{concurso_atual}.txt", "w") as f:
        f.write(f"Distribuição dos jogos para o Concurso {concurso_atual}:\n")
        for est, num in jogos_por_estrategia.items():
            f.write(f"{est}: {num} jogos\n")
        f.write("\nJogos gerados:\n")
        for i, (jogo, auditoria) in enumerate(zip(jogos_gerados, auditorias), 1):
            f.write(f"{i}. {jogo} ({auditoria['estrategia']}) [SHA-256: {auditoria['sha256']}]\n")
    
    return jogos_gerados, auditorias

# Executar
jogos, auditorias = gerar_jogos(historico, CONCURSO_ATUAL)
print(f"Jogos gerados para o Concurso {CONCURSO_ATUAL}. Veja 'jogos_concurso_{CONCURSO_ATUAL}.txt' e 'auditoria_concurso_{CONCURSO_ATUAL}.json'.")