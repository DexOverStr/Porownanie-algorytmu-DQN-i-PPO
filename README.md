# Labirynt - DQN/PPO

Projekt zawiera aplikację do generowania losowych labiryntów oraz trenowania agentów uczenia wzmacniającego w zadaniu nawigacji od punktu startowego do celu. W projekcie porównywane są dwa algorytmy: Deep Q-Network (DQN) oraz Proximal Policy Optimization (PPO).

## Zawartość projektu

- app_ui.py - interfejs graficzny aplikacji.
- main.py - konsolowy punkt startowy projektu.
- train_compare.py - trening, ewaluacja, porównanie DQN i PPO oraz uruchamianie wielu seedów.
- DQN.py - implementacja agenta DQN.
- ppo.py - implementacja agenta PPO.
- lab_env.py - środowisko labiryntu z metodami reset() i step().
- Generator_lab.py - generator labiryntów.
- eval_helpers.py - funkcje ewaluacji agentów.
- analyze_results.py - agregacja wyników zapisanych w logach.
- requirements.txt - lista wymaganych bibliotek.

## Instrukcja uruchomienia

### 1. Wymagania

Projekt wymaga środowiska Python 3.11 oraz bibliotek wymienionych w pliku requirements.txt.

### 2. Instalacja zależności

Po rozpakowaniu archiwum należy przejść do katalogu projektu i zainstalować zależności:

    python -m pip install -r requirements.txt

### 3. Uruchomienie aplikacji z interfejsem graficznym

    python app_ui.py

### 4. Uruchomienie wersji konsolowej

    python main.py

### 5. Wyniki

Wyniki treningu i ewaluacji są zapisywane w katalogu logs_out. Pliki CSV zawierają metryki z kolejnych punktów pomiarowych.

### 6. Gotowa aplikacja

Gotowa wersja aplikacji dla systemu Windows jest dostępna w repozytorium GitHub w sekcji Releases.
