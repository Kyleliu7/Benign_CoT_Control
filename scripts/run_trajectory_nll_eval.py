"""
Trajectory NLL & Prefix Surprisal Evaluator.
Computes token-level surprisal under base foundation model priors across position segments and prefix conditioning.
"""
import os
import sys
import torch
import pandas as pd
from pathlib import Path

SEGMENTS = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 250), (250, None)]
PREFIX_K_LIST = [5, 10, 20, 50]

if __name__ == "__main__":
    print("Trajectory Surprisal and Prefix Continuation Evaluator")
