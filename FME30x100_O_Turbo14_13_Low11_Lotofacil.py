import random
import pandas as pd

# Função para carregar e processar os dados
def load_data(file_path, start_concurso, end_concurso):
    df = pd.read_csv(file_path, sep=';')
    df = df[(df['Concurso'] >= start_concurso) & (df['Concurso'] <= end_concurso)]
    return df

# Função para calcular frequências e identificar números quentes/estáveis/frios
def calculate_frequencies(df, last_n=5):
    numbers = {i: 0 for i in range(1, 26)}
    for _, row in df.iterrows():
        for col in [f'Bola{i}' for i in range(2, 16)]:
            num = row[col]
            numbers[num] += 1
    
    # Identificar números que saíram menos nos últimos 5 sorteios
    last_5 = df.tail(last_n)
    recent_counts = {i: 0 for i in range(1, 26)}
    for _, row in last_5.iterrows():
        for col in [f'Bola{i}' for i in range(2, 16)]:
            num = row[col]
            recent_counts[num] += 1
    
    # Priorizar números quentes que saíram menos vezes
    sorted_numbers = sorted(numbers.items(), key=lambda x: x[1], reverse=True)
    hot = [num for num, _ in sorted_numbers[:10]]
    stable = [num for num, _ in sorted_numbers[10:20]]
    cold = [num for num, _ in sorted_numbers[20:]]
    
    # Priorizar quentes que saíram menos nos últimos 5 sorteios
    hot = sorted(hot, key=lambda x: recent_counts[x])
    stable = sorted(stable, key=lambda x: recent_counts[x])
    cold = sorted(cold, key=lambda x: recent_counts[x])
    
    return hot, stable, cold

# Função para verificar balanceamento de pares/ímpares e faixas
def is_balanced(game):
    evens = sum(1 for num in game if num % 2 == 0)
    odds = 15 - evens
    
    # Verificar faixas (1-5, 6-10, 11-15, 16-20, 21-25)
    ranges = [0] * 5  # [1-5, 6-10, 11-15, 16-20, 21-25]
    for num in game:
        if 1 <= num <= 5:
            ranges[0] += 1
        elif 6 <= num <= 10:
            ranges[1] += 1
        elif 11 <= num <= 15:
            ranges[2] += 1
        elif 16 <= num <= 20:
            ranges[3] += 1
        elif 21 <= num <= 25:
            ranges[4] += 1
    
    # Forçar exatamente 3 números por faixa
    range_check = all(r == 3 for r in ranges)
    
    return evens == 8 and odds == 7 and range_check

# Função para verificar diversidade entre jogos
def is_diverse(new_game, existing_games, min_diff=4):
    for game in existing_games:
        diff = len(set(new_game) - set(game))
        if diff < min_diff:
            return False
    return True

# Função para gerar um jogo
def generate_game(hot, stable, cold, hot_percent, stable_percent, cold_percent, existing_games):
    game = []
    hot_count = int(15 * hot_percent)
    stable_count = int(15 * stable_percent)
    cold_count = 15 - hot_count - stable_count
    
    while True:
        game = []
        game.extend(random.sample(hot, hot_count))
        game.extend(random.sample(stable, stable_count))
        game.extend(random.sample(cold, cold_count))
        
        if is_balanced(game) and is_diverse(game, existing_games):
            break
    
    return sorted(game)

# Função principal para gerar os 16 jogos
def generate_games_for_concurso(file_path, target_concurso):
    # Janela de 1500 sorteios anteriores
    start_concurso = max(1, target_concurso - 1500)
    end_concurso = target_concurso - 1
    
    # Carrega os dados
    df = load_data(file_path, start_concurso, end_concurso)
    
    # Adiciona o resultado do concurso 3374 manualmente
    concurso_3374 = pd.DataFrame({
        'Concurso': [3374],
        'Bola2': [4], 'Bola3': [5], 'Bola4': [6], 'Bola5': [7], 'Bola6': [8],
        'Bola7': [10], 'Bola8': [11], 'Bola9': [12], 'Bola10': [14], 'Bola11': [15],
        'Bola12': [17], 'Bola13': [18], 'Bola14': [20], 'Bola15': [21], 'Bola16': [24]
    })
    df = pd.concat([df, concurso_3374], ignore_index=True)
    
    # Calcula frequências
    hot, stable, cold = calculate_frequencies(df)
    
    # Gera os 16 jogos
    games = []
    existing_games = []
    # 12 jogos principais (FME30x100-O-Turbo14_13_Low11)
    for _ in range(12):
        game = generate_game(hot, stable, cold, 0.75, 0.15, 0.10, existing_games)
        games.append(('FME30x100-O-Turbo14_13_Low11', game))
        existing_games.append(game)
    
    # 3 jogos Turbo (FME30x100-O-Turbo14_13_Low11-Turbo)
    for _ in range(3):
        game = generate_game(hot, stable, cold, 0.75, 0.15, 0.10, existing_games)
        games.append(('FME30x100-O-Turbo14_13_Low11-Turbo', game))
        existing_games.append(game)
    
    # 1 jogo Azarão (FME30x100-O-Turbo14_13_Low11-Azarão)
    for _ in range(1):
        game = generate_game(hot, stable, cold, 0.70, 0.20, 0.10, existing_games)
        games.append(('FME30x100-O-Turbo14_13_Low11-Azarão', game))
    
    return games

# Exemplo de uso
if __name__ == "__main__":
    file_path = "base_Lotofacil_3373.csv"
    target_concurso = 3375
    games = generate_games_for_concurso(file_path, target_concurso)
    
    for i, (strategy, game) in enumerate(games, 1):
        print(f"Jogo {i} ({strategy}): {game}")