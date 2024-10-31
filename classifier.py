import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
from tensorflow.keras.layers import Dense, Concatenate, Input, Lambda
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam

# Load data
df = pd.read_csv("template.csv")

# Define features and label columns
# features = [
#     "Literacy and Numeracy", "Medical History", "Medications", "Surgeries", "Stroke",
#     "Other History", "Vision", "Hearing", "Diet", "Sleep", "Alcohol", "Smoking",
#     "Family History", "Main Complaints", "Memory", "Language", "Orientation",
#     "Judgment and Problem Solving", "Social Activities", "Home and Hobbies",
#     "Daily Living", "Personality and Behavior"
# ]
features = [
    "Memory", "Language", "Orientation",
    "Judgment and Problem Solving", "Social Activities", "Home and Hobbies",
    "Daily Living", "Personality and Behavior"
]
columns = ["Type"] + features
df[features] = df[features].astype(str)
df = df[columns]

# Encode labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['Type'])
num_classes = len(label_encoder.classes_)

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Load DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Define the BERT embedding layer function
def bert_embedding_layer(input_ids, attention_mask):
    return Lambda(
        lambda x: bert_model(x[0], attention_mask=x[1])[0][:, 0, :],  # Extract [CLS] token embedding
        output_shape=(768,)
    )([input_ids, attention_mask])

# Tokenize each feature separately with fixed max_length
def tokenize_feature(texts, max_length=512):
    encoding = tokenizer(
        list(texts),
        max_length=max_length,
        padding="max_length",  # Ensures fixed length of max_length
        truncation=True,
        return_tensors="tf"
    )
    # Return tensors directly
    return encoding["input_ids"], encoding["attention_mask"]

# Process train and validation encodings
train_encodings = {feature: tokenize_feature(train_df[feature]) for feature in features}
val_encodings = {feature: tokenize_feature(val_df[feature]) for feature in features}

# Convert labels to tensors
train_labels = tf.convert_to_tensor(train_df['label_encoded'], dtype=tf.int32)
val_labels = tf.convert_to_tensor(val_df['label_encoded'], dtype=tf.int32)

# Load DistilBERT model
bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Create input and embedding layers for each feature
feature_embeddings = []
model_inputs = []
for feature in features:
    input_ids = Input(shape=(512,), dtype=tf.int32, name=f"{feature.replace(' ', '_')}_input_ids")
    attention_mask = Input(shape=(512,), dtype=tf.int32, name=f"{feature.replace(' ', '_')}_attention_mask")

    # Get BERT embeddings for the feature using Lambda wrapper
    feature_embedding = bert_embedding_layer(input_ids, attention_mask)
    feature_embeddings.append(feature_embedding)

    # Add to model inputs
    model_inputs.extend([input_ids, attention_mask])

# Concatenate all feature embeddings and create classification layer
concatenated_features = Concatenate()(feature_embeddings)
# output = Dense(100, activation='relu')(concatenated_features)
# output = Dense(100, activation='relu')(output)
output = Dense(num_classes, activation='softmax')(concatenated_features)

# Define inputs for model and create final model
model = Model(inputs=model_inputs, outputs=output)

# Compile model
optimizer = Adam(learning_rate=3e-5)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary to check structure
model.summary()

# Prepare input data for training and validation with exact key names
train_inputs = {}
val_inputs = {}

for feature in features:
    # Remove spaces in feature names for compatibility with input layer names
    feature_key = feature.replace(" ", "_")
    train_inputs[f"{feature_key}_input_ids"] = train_encodings[feature][0]
    train_inputs[f"{feature_key}_attention_mask"] = train_encodings[feature][1]
    val_inputs[f"{feature_key}_input_ids"] = val_encodings[feature][0]
    val_inputs[f"{feature_key}_attention_mask"] = val_encodings[feature][1]

# Train the model
history = model.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    epochs=100,
    batch_size=4
)

# Evaluate the model
loss, accuracy = model.evaluate(val_inputs, val_labels)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")
