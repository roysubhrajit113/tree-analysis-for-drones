#====================================
#------------- README ---------------
#====================================

# Below listed are the paths of the directories that will be used for
# the images to be used in the model, and the output directories that
# will be created.

# You must change these paths according to your directory, otherwise the
# model wouldn't function properly.

#====================================
#------------------------------------
#====================================

# Base Directory of the Project

BASE_DIR = "/home/user/tree_plantation/"

# Model and Test Paths

YOLO_MODEL_PATH = f"{BASE_DIR}/models/best.pt"
LLAMA_MODEL_PATH = f"{BASE_DIR}/models/your-model-name.gguf"
INPUT_IMAGE_PATH = f"{BASE_DIR}/test_images"
OUTPUT_IMAGE_PATH = f"{BASE_DIR}/test_results"
OUTPUT_JSON_PATH = f"{BASE_DIR}/test_inference"


