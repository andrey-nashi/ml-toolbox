from random import random, seed

class Randomizer:

    def __init__(self, seed_value: int):
        seed(seed_value)