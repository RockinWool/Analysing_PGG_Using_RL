# Analysing Public Goods Games Using Reinforcement Learning, Effect of Increasing Group Size on Cooperation

This repository is used to publish information about papers with the above titles.

## Table of Contents

- [About](#about)
- [Installation&Usage](#installation&Usage)
- [Contributing](#contributing)
- [License](#license)

## About

The key results of this project can be found in [./data/cooperation_rate.csv](./data/cooperation_rate.csv). \
You can get similar results by executing [main.sh](./main.sh). The output file is `./data/sample.csv` as default.\
But executing [main.sh](./main.sh) needs a lot of time (It takes almost 1 day). \ 
So, if you just checking the result, you should check [cooperation_rate.csv](./data/cooperation_rate.csv) directly or change trial number from 50 -> 1 in [parameters.yaml](./data/parameters.yaml).  

## Installation & Usage

Clone this repository and then run main.sh.\
NOTE: as I said, main.sh takes a lot of time. You should change some parameters or main.sh's code.
NOTE2: This code requires tensorflow and some other python packages.(tensorflow, pyyaml, scipy, numpy, pandas, networkx..)
```bash
bash main.sh
```

# Contributing
You can be free to contribute this project.  If you want to do it, just send PR to this project. 

# License

This project is licensed under the GPLv3. You can find the details in the LICENSE file. \
But additionaly, you should read my paper "Cooperation is encouraged simply by increasing the number of participants" first.
