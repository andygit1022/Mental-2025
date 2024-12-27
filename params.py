########### params.py ##############
# general
MODEL_PATH = "model.keras"
DATASET_PATH = "data/dataset.csv"
CLASSES = ["NC", "PRD"]
NUM_CLASSES = len(CLASSES)

FULL_FEATURES = {
    "Patient_ID": 'int16',
    "Label": 'str',
    "Gender": 'str',
    "Age": "int16",
    "Education": "int16",
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
    "Memory",
    "Language",
    "Orientation",
    "Judgment and Problem Solving",
    # "Social Activities",
    # "Home and Hobbies",
    # "Daily Living",
    "Personality and Behavior",
]

# training parameters
LEARNING_RATE = 1e-4
EPOCHS = 10000
BATCH_SIZE = 1
EPOCHS_PER_CYCLE = 4

# bert
MAX_LEN = 64

# llama
LLAMA_MODEL_PATH = "./Llama3.1-8B-Instruct"