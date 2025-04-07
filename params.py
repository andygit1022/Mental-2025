########### params.py ##############
# general
MODEL_PATH = "model.keras"
DATASET_PATH = "data/dataset_mci_nc_prd.csv"
CLASSES = ["NC", "PRD","MCI"]
NUM_CLASSES = len(CLASSES)

FULL_FEATURES = {
    "Patient_ID": 'int16',
    "Label": 'str',
    "Gender": 'str',
    "Age": "str",
    "Education": "str",
    #"Age": "int16",
    #"Education": "int16",
    "Literacy and Numeracy": "str",
    "Medical History": "str",
    "Medications": "str",
    "Surgeries": "str",
    "Stroke": "str",
    "Other History": "str",
    "Vision": "str",
    "Hearing": "str",
    "Diet": "str",
    "Sleep": "str",
    "Alcohol": "str",
    "Smoking": "str",
    "Family History": "str",
    "Main Complaints": "str",
    "Memory": "str",
    "Language": "str",
    "Orientation": "str",
    "Judgment and Problem Solving": "str",
    "Social Activities": "str",
    "Home and Hobbies": "str",
    "Daily Living": "str",
    "Personality and Behavior": "str",
    # 'MMSE': 'int32',
    # 'CIST': 'int32',
    # 'GDS': 'int32'
}
FEATURES = [
    "Patient_ID",   # for loggig
    "Age",
    "Main Complaints",
    "Gender",
    "Memory",
    "Language",
    "Orientation",
    "Judgment and Problem Solving",
    #"Social Activities",
    "Home and Hobbies",
    #"Daily Living",
    "Personality and Behavior"
]

# FEATURES = [
#     "Patient_ID",   # for loggig
#     "Age",
#     "Main Complaints",
#     "Memory",
#     "Language",
#     "Orientation",
#     "Judgment and Problem Solving",
#     #"Social Activities",
#     "Home and Hobbies",
#     #"Daily Living",
#     "Personality and Behavior"
# ]




# training parameters
LEARNING_RATE = 2e-5
EPOCHS = 5
BATCH_SIZE = 1
EPOCHS_PER_CYCLE = 4


#POOLING_MODE="cls" # Mean Pooling
POOLING_MODE="mean" # Mean Pooling
#POOLING_MODE="max" # Max Pooling
#POOLING_MODE="attention" # Attention-weighted Pooling(기존 코드와 동일)
#POOLING_MODE="learnable" # MLP로 각 토큰 점수 산출 → 가중합
#POOLING_MODE="proj" # Mean Pooling 후 (4096→1024→256)
#POOLING_MODE="mlp_after_pooling : max" # Max Pool 후 (4096→1024→256→64→1)
#POOLING_MODE="query_attn" # Learnable Query Self-Attention
#POOLING_MODE="mlp_multi_layers"

# bert
MAX_LEN = 64

# llama
LLAMA_MODEL_PATH = "./Llama3.1-8B-Instruct"