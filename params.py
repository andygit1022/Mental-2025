# general
MODEL_PATH = "model.keras"
DATASET_PATH = "data/dataset.csv"
CLASSES = ["NC", "PRD"]
NUM_CLASSES = len(CLASSES)

# FEATURES = [
#     "Literacy and Numeracy", "Medical History", "Medications", "Surgeries", "Stroke",
#     "Other History", "Vision", "Hearing", "Diet", "Sleep", "Alcohol", "Smoking",
#     "Family History", "Main Complaints", "Memory", "Language", "Orientation",
#     "Judgment and Problem Solving", "Social Activities", "Home and Hobbies",
#     "Daily Living", "Personality and Behavior"
# ]
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
