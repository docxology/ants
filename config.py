import numpy as np

# Ant parameters
ADD_ANT_EVERY = 50
INIT_X = 20
INIT_Y = 30
NEST_FACTOR = 0.1

# Grid parameters
GRID_WIDTH = 40
GRID_HEIGHT = 40
GRID = (GRID_WIDTH, GRID_HEIGHT)
GRID_SIZE = np.prod(GRID)

# Focal area parameters
FOCAL_WIDTH = 3
FOCAL_HEIGHT = 3
FOCAL_AREA = (FOCAL_WIDTH, FOCAL_HEIGHT)
FOCAL_SIZE = np.prod(FOCAL_AREA)

# Action parameters
ACTION_MAP = [
    (-1, -1), (0, -1), (1, -1),  # Up-left, Up, Up-right
    (-1, 0), (0, 0), (1, 0),     # Left, Stay, Right
    (-1, 1), (0, 1), (1, 1)      # Down-left, Down, Down-right
]
NUM_ACTIONS = len(ACTION_MAP)
OPPOSITE_ACTIONS = list(reversed(range(NUM_ACTIONS)))

# Food source parameters
FOOD_X = GRID_WIDTH
FOOD_Y = 5
FOOD_LOCATION = (FOOD_X, FOOD_Y)
FOOD_WIDTH = 10
FOOD_HEIGHT = 10
FOOD_SIZE = (FOOD_WIDTH, FOOD_HEIGHT)

# Wall parameters
WALL_LEFT = 15
WALL_RIGHT = 25
WALL_TOP = 10

# Pheromone parameters
NUM_PHEROMONE_LEVELS = 10
DECAY_FACTOR = 0.01

# MDP parameters
NUM_OBSERVATIONS = NUM_PHEROMONE_LEVELS
NUM_STATES = FOCAL_SIZE

# Simulation parameters
MAX_STEPS = 500
