# general
MODEL_PATH = "model.keras"
DATASET_PATH = "data/dataset.csv"
CLASSES = ["NC", "PRD"]
NUM_CLASSES = len(CLASSES)

# features = [
#     "Literacy and Numeracy", "Medical History", "Medications", "Surgeries", "Stroke",
#     "Other History", "Vision", "Hearing", "Diet", "Sleep", "Alcohol", "Smoking",
#     "Family History", "Main Complaints", "Memory", "Language", "Orientation",
#     "Judgment and Problem Solving", "Social Activities", "Home and Hobbies",
#     "Daily Living", "Personality and Behavior"
# ]
FEATURES = [
    "Main Complaints", "Memory", "Language", "Orientation",
    "Judgment and Problem Solving", "Social Activities", "Home and Hobbies",
    "Daily Living", "Personality and Behavior"
]

# training parameters
LEARNING_RATE = 3e-5
EPOCHS = 1
BATCH_SIZE = 4
