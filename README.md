# Trabalho de Grafos - Cisco Secure Workload G21

Este repositório contém os scripts e dados tratados utilizados na análise do grafo G21 do dataset Cisco Secure Workload.

## Estrutura

```txt
├── Cisco_22_networks/
│   └── dados originais do grafo G21
├── data/
│   └── processed/
│       └── dados tratados
├── results/
│   ├── tabelas CSV
│   └── gráficos gerados
├── 01_ler_g21.py
├── 02_metricas_parte1.py
├── 03_algoritmos_parte2.py
├── 04_analise_estrutural_parte3.py
├── README.md
├── .gitignore
└── requirements.txt
```

- `Cisco_22_networks/`: pasta contendo os dados originais do grafo G21.
- `data/processed/`: conjunto de dados tratado utilizado no trabalho.
- `results/`: pasta onde estão armazenados os resultados, incluindo tabelas CSV e gráficos gerados.
- `01_ler_g21.py`: script para leitura e tratamento do grafo G21.
- `02_metricas_parte1.py`: script para cálculo de métricas do grafo G21.
- `03_algoritmos_parte2.py`: script para execução de algoritmos no grafo G21.
- `04_analise_estrutural_parte3.py`: script para análise estrutural do grafo G21.

## Criando o ambiente virtual e instalando dependências

1. Crie um ambiente virtual:

```sh
python3 -m venv venv
```

2. Ative o ambiente virtual:

- No Windows:
```sh
venv\Scripts\activate
```

- No macOS/Linux:
```sh
source venv/bin/activate
```

3. Instale as dependências:
```sh
pip3 install -r requirements.txt
```

## Execução dos scripts

1. Para ler e tratar o grafo G21:
```sh
python3 01_ler_g21.py
```

2. Para calcular as métricas do grafo G21:
```sh
python3 02_metricas_parte1.py
```

3. Para executar os algoritmos no grafo G21:
```sh
python3 03_algoritmos_parte2.py
```

4. Para realizar a análise estrutural do grafo G21:
```sh
python3 04_analise_estrutural_parte3.py
```

## Resultados

Os resultados dos scripts estão disponíveis na pasta `results/`, onde você pode encontrar as tabelas CSV e os gráficos gerados a partir da análise do grafo G21.
