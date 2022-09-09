from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.COARSE_RAY_SAMPLING = 64
_C.MODEL.FINE_RAY_SAMPLING = 80
_C.MODEL.INPORTANCE_RAY_SAMPLE = 0
_C.MODEL.SAMPLE_METHOD = "NEAR_FAR"
_C.MODEL.BOARDER_WEIGHT = 1e10
_C.MODEL.SAME_SPACENET = False
_C.MODEL.DEPTH_FIELD = 1e-1
_C.MODEL.DEPTH_RATIO = 1e-1
_C.MODEL.SAMPLE_INF = False
_C.MODEL.USE_MOTION = False

_C.MODEL.UNET_DIR = False
_C.MODEL.USE_SH = True
_C.MODEL.SH_DIM = 9
_C.MODEL.SH_FEAT_DIM = 3
_C.MODEL.BONE_FEAT_DIM = 0
_C.MODEL.UNET_FEAT = False
_C.MODEL.USE_RENDER_NET = True
_C.MODEL.USE_LIGHT_RENDERER = False
_C.MODEL.USE_BONE_NET = False
_C.MODEL.USE_WARP_NET = False

_C.MODEL.SAMPLE_NET = False
_C.MODEL.OPACITY_NET = False
_C.MODEL.NOISE_STD = 0.0

_C.MODEL.TKERNEL_INC_RAW = True
_C.MODEL.ENCODE_POS_DIM = 10
_C.MODEL.ENCODE_DIR_DIM = 4
_C.MODEL.GAUSSIAN_SIGMA = 10.
_C.MODEL.KERNEL_TYPE = "POS"
_C.MODEL.PAIR_TRAINING = False
_C.MODEL.TREE_DEPTH = 8
_C.MODEL.RANDOM_INI = True

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [400, 250]
# Size of the image during test
_C.INPUT.SIZE_TEST = [400, 250]
# Minimum scale for the image during training
_C.INPUT.MIN_SCALE_TRAIN = 0.5
# Maximum scale for the image during test
_C.INPUT.MAX_SCALE_TRAIN = 1.2
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.1307, ]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.3081, ]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ""
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ""
_C.DATASETS.SHIFT = 0.0
_C.DATASETS.MAXRATION = 0.0
_C.DATASETS.ROTATION = 0.0
_C.DATASETS.USE_MASK = False
_C.DATASETS.USE_DEPTH = False
_C.DATASETS.USE_ALPHA = False
_C.DATASETS.USE_BG = False
_C.DATASETS.NUM_FRAME = 1
_C.DATASETS.NUM_CAMERA = 1000
_C.DATASETS.TYPE = "NR"
_C.DATASETS.SYNTHESIS = True
_C.DATASETS.NO_BOUNDARY = False
_C.DATASETS.BOUNDARY_WIDTH = 3
_C.DATASETS.PATCH_SIZE = 16
_C.DATASETS.KEEP_BG = False
_C.DATASETS.PAIR_SAMPLE = False

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"

_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.LOSS_FN = "L1"

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 100
_C.SOLVER.BUNCH = 4096
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.START_ITERS = 50
_C.SOLVER.END_ITERS = 200
_C.SOLVER.LR_SCALE = 0.1
_C.SOLVER.COARSE_STAGE = 10

# used for update 3d geometry
_C.SOLVER.START_EPOCHS = 10
_C.SOLVER.EPOCH_STEP = 1
_C.SOLVER.CUBE_RESOLUTION = 512
_C.SOLVER.CURVE_BUNCH = 10000
_C.SOLVER.THRESHOLD = 5.
_C.SOLVER.CURVE_SAMPLE_NUM = 64
_C.SOLVER.CURVE_SAMPLE_SCALE = 1
_C.SOLVER.UPDATE_GEOMETRY = False
_C.SOLVER.UPDATE_RANGE = False
_C.SOLVER.USE_BOARDER_COLOR = False
_C.SOLVER.USE_AMP = False
_C.SOLVER.SEED = 2021


# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 8
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""
