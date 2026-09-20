"""Compatibility entry point: train the shared LSTM32 experiment."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.lstm.run_all import cli

if __name__ == '__main__':
    cli(fixed_history=32)
