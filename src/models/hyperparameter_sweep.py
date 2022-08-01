import argparse
import wandb

sweep_config = {
  "name": "my-sweep",
  "method": "random",
  "parameters": {
    "epochs": {
      "values": [10, 20, 50]
    },
    "learning_rate": {
      "min": 0.0001,
      "max": 0.1
    }
  }
}

sweep_id = wandb.sweep(sweep_config)

def train():
    with wandb.init() as run:
        pass