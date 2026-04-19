# Q-Learning Based Trading Project

This project implements a **Q-Learning based trading agent** for **Bitcoin (BTC-USD)** using historical market data.  
The main goal is to let an agent learn a trading policy (long / short / hold) from price data and **compare its performance** to a simple **buy-and-hold** strategy.

---

## Project Overview

- **Asset:** BTC-USD (Bitcoin / US Dollar)  
- **Data Source:** Yahoo Finance  
- **Data Range:** 2015–2024 (daily data)  
- **Algorithm:** Tabular **Q-Learning**  
- **Environment:** Custom trading environment (`TradingEnv`)  
- **Benchmark:** Buy & Hold on BTC

The agent interacts with a simplified trading environment, makes decisions at each time step, receives rewards based on price movements, and gradually improves its policy through Q-Learning.

---

## Environment: `TradingEnv`

The environment is a custom trading simulator built around BTC-USD historical data.

### Actions

At each time step, the agent can choose one of three discrete actions:

1. `0` – **Short** (bet price will go down)  
2. `1` – **Hold** (no position / stay in cash)  
3. `2` – **Long** (bet price will go up)

### State Representation

The state is based on **discretized features** derived from the market data. Typical features include:

- **Close** – Closing price  
- **Return** – Daily return (e.g., percentage price change)  
- **RSI** – Relative Strength Index (momentum indicator)  
- **Volume** – Trading volume  

These continuous features are **discretized** into bins so they can be used as keys in a tabular Q-table.

### Reward Function

The reward at each step is based on:

- **Price movement** in the direction of the current position (profit / loss)  
- **Transaction cost** (a small penalty for changing positions)

So the agent is encouraged to:

- Take positions that correctly anticipate price direction  
- Avoid over-trading due to transaction costs  

---

## Q-Learning Agent: `QAgent`

The Q-Learning trading agent is implemented in the class **`QAgent`**.

Key components:

- **Q-Table**  
  A table (e.g., a dictionary) mapping `(state, action)` pairs to Q-values.

- **Policy – ε-greedy**  
  - With probability **ε** → choose a random action (**exploration**)  
  - With probability **1 − ε** → choose the best action based on current Q-values (**exploitation**)

- **Update Rule (Q-Learning)**  

  For each transition \((s, a, r, s')\), Q-values are updated as:

  ```math
  Q(s, a) \leftarrow Q(s, a) + lpha ig( r + \gamma \max_{a'} Q(s', a') - Q(s, a) ig)
  ```

  where:
  - \(lpha\): learning rate  
  - \(\gamma\): discount factor  
  - \(r\): immediate reward  
  - \(s'\): next state  

---

## Code Structure

The core logic is inside the notebook:

- **`AI_Project.ipynb`**
  - Loads `data.csv` (BTC-USD historical data)
  - Computes features: `Close`, `Return`, `RSI`, `Volume`
  - Discretizes features into state bins
  - Splits data into **train** and **test**
  - Defines:
    - `TradingEnv` (trading environment)
    - `QAgent` (Q-Learning trading agent)
  - Trains the agent over multiple episodes (e.g., 2000 episodes)
  - Evaluates the learned policy on the test period
  - Compares performance against a **Buy & Hold** benchmark
  - Plots equity curves and prints performance metrics

Other files:

- **`data.csv`** – Historical BTC-USD data with columns:  
  `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

- **`README.txt`** – Short textual description of the project goals and data source.

- **`ai-project.pdf`** – A detailed report/description (e.g., course or project report) summarizing the method and results.

---

## Training & Evaluation Pipeline

The notebook typically follows these steps:

1. **Load Data**
   - Read `data.csv`
   - Sort by date and handle missing values if needed

2. **Feature Engineering**
   - Compute daily **returns**
   - Compute **RSI**
   - Use `Close`, `Return`, `RSI`, `Volume` as features
   - Discretize continuous features into bins (for tabular Q-Learning)

3. **Train/Test Split**
   - Early period → **train** set  
   - Later period → **test** set (out-of-sample evaluation)

4. **Initialize Environment & Agent**
   - Create `TradingEnv` for the train set
   - Initialize `QAgent` with:
     - learning rate (α)  
     - discount factor (γ)  
     - exploration rate (ε) and decay schedule  

5. **Training**
   - Run ~**2000 episodes** over the training period
   - In each episode:
     - Reset environment  
     - For each time step:
       - Choose action via ε-greedy  
       - Observe reward and next state  
       - Update Q-table with Q-Learning rule  

6. **Testing**
   - Freeze the Q-table (no more learning)
   - Run the agent greedily (always choose best action) on the **test** period
   - Track **Net Asset Value (NAV)** over time  

7. **Performance Metrics**

   The notebook reports metrics such as:

   - **Total Return**  
     - Example: `Total Return: 152.51%`
   - **Sharpe Ratio**  
   - **Max Drawdown**  
     - Example: `Max Drawdown: 29.34%`
   - **Win Rate** (percentage of profitable trades)  
     - Example: `Win Rate: 14.43%`

   It also plots:

   - NAV curve of the **Q-Learning Agent**  
   - NAV curve of **Buy & Hold**  

---

## Requirements

You will need **Python 3.8+** and a few common libraries.

Typical dependencies (based on the notebook):

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `ta` (or an equivalent implementation for computing RSI, if used)
- `seaborn` (optional, for plots)
- `jupyter` (to run the notebook)

Example `requirements.txt`:

```text
numpy
pandas
matplotlib
scikit-learn
ta
seaborn
jupyter
```

Install dependencies via:

```bash
pip install -r requirements.txt
```

or manually:

```bash
pip install numpy pandas matplotlib scikit-learn ta seaborn jupyter
```

---

## How to Run

1. **Clone the Repository**

```bash
git clone https://github.com/Mhmd-Alami/AI_Q_Learning.git
cd AI_Q_Learning
```

2. **(Optional) Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scriptsctivate         # Windows
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**

```bash
jupyter notebook
```

5. **Open and Run the Notebook**

- Open `AI_Project.ipynb`  
- Run all cells from top to bottom:
  - Data loading & feature engineering  
  - Environment and agent definitions  
  - Training loop  
  - Evaluation and plotting  

---

## What You Will Learn

By using this project, you will:

- See how to **formulate a trading problem** as a reinforcement learning task  
- Understand how to:
  - Design **states**, **actions**, and **rewards** for financial time series  
  - Apply **tabular Q-Learning** to historical market data  
  - Use **ε-greedy** exploration and Q-table updates  
- Compare RL-based trading vs. a simple **buy-and-hold** strategy  
- Interpret typical trading performance metrics:
  - Total Return  
  - Sharpe Ratio  
  - Max Drawdown  
  - Win Rate  

---

## Limitations

- The environment is a **simplified** model of real markets:
  - No slippage or liquidity constraints  
  - Simple, fixed transaction cost  
  - Usually fixed position size  

- Tabular Q-Learning requires **discretization** of states:
  - Does not scale well to many features or high-frequency data  

- Backtest results on historical data do **not** guarantee future performance.

---

## Possible Extensions

Some ideas to extend this project:

- Replace tabular Q-Learning with **Deep Q-Network (DQN)**  
- Use function approximation to handle **continuous state spaces**  
- Add more technical indicators (MACD, Bollinger Bands, etc.)  
- Add **position sizing**, leverage, and risk management rules  
- Try multiple assets (crypto pairs, stocks, ETFs)  
- Use different time resolutions (e.g., 4h, hourly, weekly)  
- Perform **walk-forward analysis** or rolling retraining  

---

## Contributing

Contributions are welcome.

You can:

- Open issues for bugs or suggestions  
- Submit pull requests for:
  - Code improvements  
  - New features or indicators  
  - Better evaluation and visualization  
  - Documentation updates  

Please keep the code clean and add comments/docstrings where useful.

---

## License

This project is for educational and personal use. Feel free to fork and improve it!

---

## Acknowledgements

- Bitcoin (BTC-USD) historical price data from **Yahoo Finance**  
- Q-Learning algorithm based on standard Reinforcement Learning literature  
- This project can serve as a starting point for:
  - Academic assignments  
  - Personal experiments in algorithmic trading  
  - Learning and teaching basic RL concepts in finance.
