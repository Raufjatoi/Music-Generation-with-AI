import numpy as np
import os
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from tensorflow.keras.utils import to_categorical

def get_notes():
    """Extract notes and chords from MIDI files in the dataset."""
    notes = []
    for root, _, files in os.walk('midi_songs'):
        for file in files:
            if file.endswith(".mid"):
                file_path = os.path.join(root, file)
                try:
                    midi = converter.parse(file_path)
                    parts = instrument.partitionByInstrument(midi)
                    if parts:  # file has instrument parts
                        notes_to_parse = parts.parts[0].recurse()
                    else:
                        notes_to_parse = midi.flat.notes

                    for element in notes_to_parse:
                        if isinstance(element, note.Note):
                            notes.append(str(element.pitch))
                        elif isinstance(element, chord.Chord):
                            notes.append('.'.join(str(n) for n in element.normalOrder))
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")
    return notes

# Extract notes
notes = get_notes()
print(f"Total notes extracted: {len(notes)}")

# Prepare the sequences used by the Neural Network
sequence_length = 100
n_vocab = len(set(notes))
print(f"Vocabulary size: {n_vocab}")

pitchnames = sorted(set(item for item in notes))
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])

print(f"Total patterns: {len(network_input)}")

if len(network_output) == 0:
    print("Error: No sequences were created. Check the sequence length and input notes.")
    exit()

n_patterns = len(network_input)
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
network_input = network_input / float(n_vocab)
network_output = to_categorical(network_output)

# Build the LSTM network
model = Sequential()
model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# Train the model
model.fit(network_input, network_output, epochs=3, batch_size=64)

# Save the model
model.save('music_generator_model.h5')
