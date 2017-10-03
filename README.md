# Starcraft 2 Reinforcement Learning

### Dependencies:
  - [PySC2][pysc2]
  - [Tensorflow][tf]
  - [Keras][keras]
  - [Python 3.5][python]

### Installation

  - Install Dependencies
  - copy Maps to Starcraft/Maps/rl_scenarios Folder
  - run train.py

### Run Options

for full list of options run with

```python train.py --help --map None```

Standard Train run:

```python train.py --map Vulture_Firebats --threads <num_cpu> --algorithm a3c.a3c.A3c -run_time <num_seconds>```

for validation add flag ```--validate```

   [pysc2]: <https://github.com/deepmind/pysc2>
   [tf]: <https://www.tensorflow.org>
   [keras]: <https://keras.io>
   [python]: <https://www.python.org>