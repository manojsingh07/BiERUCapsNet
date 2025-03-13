import tensorflow as tf
MAX_SEQUENCE_LENGTH = 100
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#copy code from load pre train embading
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts = ["sample text data here", "another sample text"]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)  


#copy code from pre process data
from sklearn.model_selection import train_test_split

# Load IEMOCAP data (Text and Emotion labels)
def load_iemocap_data(data_dir):
    texts, labels = [], []
    for session in os.listdir(data_dir):
        transcripts_dir = os.path.join(data_dir, session, 'transcriptions')
        emotion_dir = os.path.join(data_dir, session, 'EmoEvaluation')
        
        if not os.path.isdir(transcripts_dir) or not os.path.isdir(emotion_dir):
            continue
        
        for transcript_file in os.listdir(transcripts_dir):
            with open(os.path.join(transcripts_dir, transcript_file), 'r') as f:
                texts.append(f.read().strip())
            #emotion_file = transcript_file.replace('transcript', 'emotion')
            emotion_file = transcript_file.replace('path', 'emotion')
            with open(os.path.join(emotion_dir, emotion_file), 'r') as f:
                labels.append(f.read().strip())
    
    return texts, labels

# Specify data directory and load data
#Changes done by manoj
#data_dir = '/path/to/IEMOCAP'
#data_dir = '/content/drive/MyDrive/GitHub/BiERUCapChatGPT/data/iemocap_full_dataset.csv'
data_dir = '/content/drive/MyDrive/GitHub/BiERUCapChatGPT/data/Session2'
#code added by manoj
#df = pd.read_csv(data_dir)
#texts = df['path'].tolist()
#labels = df['emotion'].tolist()

texts, labels = load_iemocap_data(data_dir)

# Preprocess text: Tokenization and Padding
MAX_SEQUENCE_LENGTH = 100
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Encode labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
y_data = pd.get_dummies(encoded_labels).values  # One-hot encoding

# Split data
#code added by manoj
print(f"Data shape: {data.shape}")
print(f"Labels shape: {y_data.shape}")

# Ensure y_data is one-hot encoded
if len(y_data.shape) == 1:  # If shape is (samples,) instead of (samples, num_classes)
    y_data = to_categorical(y_data, num_classes=num_classes)

x_train, x_test, y_train, y_test = train_test_split(data, y_data, test_size=0.2, random_state=42)

# Load GloVe embeddings
def load_glove_embeddings(glove_dir, word_index, embedding_dim=300):
    embeddings_index = {}
    with open(os.path.join(glove_dir, 'glove.6B.300d.txt'), 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

# Define embedding layer
glove_dir = '/content/drive/MyDrive/GitHub/BiERUCapChatGPT/data'
embedding_matrix = load_glove_embeddings(glove_dir, tokenizer.word_index)
embedding_layer = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1,
                                            output_dim=300,
                                            weights=[embedding_matrix],
                                            input_length=MAX_SEQUENCE_LENGTH,
                                            trainable=False)
#end copied code




# BiERU Layer
class BiERU(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BiERU, self).__init__()
        self.rnn = layers.Bidirectional(layers.GRU(units, return_sequences=True))

    def call(self, inputs):
        return self.rnn(inputs)

''' old code
# Capsule Layer with Dynamic Routing
class CapsuleLayer(tf.keras.layers.Layer):
    def __init__(self, num_capsules, capsule_dim, num_routing=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.num_routing = num_routing

    def build(self, input_shape):
        self.W = self.add_weight(shape=[self.num_capsules, input_shape[1], self.capsule_dim, input_shape[2]],
                                 initializer='glorot_uniform', trainable=True)

    def call(self, inputs):
        inputs_expand = tf.expand_dims(inputs, 1)
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsules, 1, 1])
        inputs_hat = tf.einsum('bij,cijk->bcjk', inputs_tiled, self.W)
        b = tf.zeros(shape=[tf.shape(inputs)[0], self.num_capsules, inputs.shape[1]])

        for i in range(self.num_routing):
            c = tf.nn.softmax(b, axis=1)
            s = tf.reduce_sum(c[..., None] * inputs_hat, axis=2)
            v = self.squash(s)
            if i < self.num_routing - 1:
                b += tf.einsum('bcjk,bcjk->bcj', inputs_hat, v[..., None, :])
        return v

    def squash(self, s):
        s_squared_norm = tf.reduce_sum(tf.square(s), axis=-1, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm)
        return scale * s / tf.sqrt(s_squared_norm + K.epsilon())

'''
#new code
class CapsuleLayer(Layer):
    def __init__(self, num_capsules, capsule_dim, num_routing=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules  # Number of output capsules
        self.capsule_dim = capsule_dim    # Dimensionality of each capsule
        self.num_routing = num_routing    # Routing iterations

    def build(self, input_shape):
        input_dim = input_shape[-1]  # Get input feature dimension (e.g., 128)
        self.W = self.add_weight(shape=[input_dim, self.num_capsules * self.capsule_dim],
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        inputs = tf.reshape(inputs, [batch_size, -1, tf.shape(inputs)[-1]])  # Reshape input

        # Compute capsule output using `einsum`
        inputs_hat = tf.einsum('bij,jk->bik', inputs, self.W)  # Now shape is (batch, seq_len, num_capsules * capsule_dim)
        inputs_hat = tf.reshape(inputs_hat, [batch_size, -1, self.num_capsules, self.capsule_dim])  # Reshape properly
        
        # Squash Activation Function for Capsules
        def squash(vector):
            squared_norm = tf.reduce_sum(tf.square(vector), axis=-1, keepdims=True)
            scale = squared_norm / (1 + squared_norm) / tf.sqrt(squared_norm + tf.keras.backend.epsilon())
            return scale * vector

        outputs = squash(inputs_hat)  # Apply squash function
        return outputs


# Assemble the Model
def build_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = embedding_layer(inputs)
    x = BiERU(units=64)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)  # Removes extra time step dimension
    primary_capsules = CapsuleLayer(num_capsules=10, capsule_dim=16)(x)
    capsule_output = CapsuleLayer(num_capsules=num_classes, capsule_dim=16)(primary_capsules)
    capsule_output_length = layers.Lambda(lambda z: tf.sqrt(tf.reduce_sum(tf.square(z), axis=-1)))(capsule_output)
    #output = layers.Dense(num_classes, activation='softmax')(capsule_output_length)
    #output = layers.Dense(num_classes, activation='softmax')(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  # Ensure 28 classes
   
    model = models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create model
input_shape = (MAX_SEQUENCE_LENGTH,)
#code added manoj
# Load dataset (Replace with actual dataset loading method)
df = pd.read_csv('/content/drive/MyDrive/GitHub/BiERUCapChatGPT/data/iemocap_full_dataset.csv')

# Ensure 'emotion' column exists
if 'emotion' not in df.columns:
    print("Error: 'emotion' column not found in dataset!")
    exit()

# Encode labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(df['emotion'])  # Convert emotions to integers

# Convert to one-hot encoding
y_data = pd.get_dummies(encoded_labels).values  # Shape: (num_samples, num_classes)
# Convert labels to categorical if needed
if len(y_data.shape) == 1:  
    y_data = to_categorical(y_data, num_classes=num_classes)
num_classes = y_data.shape[1]
#model = build_model(input_shape, num_classes)
model = build_model(input_shape, 30)

#code copy from train evaluation
# Train the Model
history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))

# Evaluate the Model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
