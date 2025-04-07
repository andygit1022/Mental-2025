# general
MODEL_PATH = "my_model.keras"      # "model.keras"
DATASET_PATH = "data/250109/dataset.csv"
CLASSES = ["NC", "MCI", "AD"]
NUM_CLASSES = 3 #len(CLASSES)

FULL_FEATURES = {
    "Patient_ID": 'int32',
    "Label": 'str',
    "Gender": 'int32',
    "Age": "int32",
    "Edu": "int32",
    "Education": "str",
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
    "Gender",
    "Age",
    "Edu",
    "Main Complaints",
    "Memory",
    "Language",
    "Orientation",
    "Judgment and Problem Solving",
    "Social Activities",
    "Home and Hobbies",
    "Daily Living",
    "Personality and Behavior",
]


# sentence feature
SENTENCE_FEATURE = 0
STAT_DIM = 64 # 16

# training parameters
LEARNING_RATE = 1e-4
EPOCHS = 75
BATCH_SIZE = 3
EPOCHS_PER_CYCLE = 4
SS = True

# bert
MAX_LEN = 128
NUM_HEADS = 4
ATTENTION_DIM = 128
HIDDEN_SIZE = 768
PROJ_DIM = HIDDEN_SIZE
NUM_SENTENCES = 10
NUM_STR_FEATURES = 9
REPRESENTATION_DIM = 255

# llama
LLAMA_MODEL_PATH = "./Llama3.2-3B-Instruct"