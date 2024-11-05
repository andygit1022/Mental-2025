# general
MODEL_PATH = "model.keras"
DATASET_PATH = "data/dataset.csv"
CLASSES = ["NC", "PRD"]
NUM_CLASSES = len(CLASSES)

FULL_FEATURES = {
    "Patient_ID": 'int32',
    "Type": 'str',
    "Gender": 'str',
    "Age": "int32",
    "Education": "int32",
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
    "Age", "Main Complaints", "Memory", "Language", "Orientation",
    "Judgment and Problem Solving", "Social Activities", "Home and Hobbies",
    "Daily Living", "Personality and Behavior"
]

# training parameters
LEARNING_RATE = 5e-6
EPOCHS = 200
BATCH_SIZE = 4

# bert
MAX_LEN = 512
