import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Generate some example text for training
text = "This is an example text for training a character-level LSTM model."
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Prepare the data
maxlen = 40
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i + maxlen])
    next_chars.append(text[i + maxlen])

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# Build the model
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model with GPU acceleration
with tf.device('/GPU:0'):
    model.fit(x, y, epochs=50, batch_size=128)

# Function to generate text
def generate_text(seed_text, temperature=1.0, max_generated=400):
    generated_text = seed_text

    for i in range(max_generated):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(seed_text):
            x_pred[0, t, char_indices[char]] = 1.0

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.random.choice(len(chars), p=preds)

        next_char = indices_char[next_index]

        generated_text += next_char
        seed_text = seed_text[1:] + next_char

    return generated_text

# Generate text based on a seed
seed_text = "This is an example text for training"
generated_text = generate_text(seed_text)
print(generated_text)
