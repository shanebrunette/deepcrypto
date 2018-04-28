# deepcrytpo
Exploration of deep reinforcement learning algorithms for predicting price data.

#### src/gym_crypto/
OpenAI inspired gym environment for preprocessing data sourced from [coinmarketcap](https://coinmarketcap.com).

#### src/agent/
Deulling deep double q learning network with options for lstm or feed forward configurations. Compatible with OpenAI gym and gym_crypto.

[Duelling DDQN Tensorflow Graph](./src/agent/duelling_ddqn_graph.jpg?raw=true "Duelling DDQN Tensor Graph")

## Get Started

### Requirements
* Python >= 3.6
* Tensorflow >= 1.0

### Usage

From agent run main.py. Environment must be declared with one of:
* --coin
* --gym
* --coinlist

Other parameters are optional. Example:
```
python main.py --coin bitcoin --verbose --dropout 0.2 --seed 2 --episodes 10000 --epsilon_strength 500 --target_epsilon 0.05 --gamma 0.95
```

## Roadmap
* Analysis of results
* Tutorial
* A3C agent
* Exploration of different time series
* API

## Acknowledgement and further reading
* [Crypto price data](https://coinmarketcap.com)
* [Introduction to deep reinforcement agents](https://github.com/awjuliani/DeepRL-Agents)
* [Dueling Network](https://arxiv.org/pdf/1511.06581.pdf)
* [Advantage function](https://arxiv.org/pdf/1511.06581.pdf)
* [LSTM and Deep Q Learning](https://arxiv.org/pdf/1507.06527.pdf)
* [Dropout and RNN](https://arxiv.org/pdf/1409.2329.pdf)


