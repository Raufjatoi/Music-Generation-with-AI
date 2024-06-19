import numpy as np
from keras.models import load_model
from music21 import converter, instrument, note, chord, stream
import os

# Function to recursively get all MIDI files
def get_all_midi_files(directory):
    """Recursively get all MIDI files in a directory."""
    midi_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".mid"):
                midi_files.append(os.path.join(root, file))
    return midi_files

# Function to load the notes
def get_notes(selected_files):
    """Extract notes and chords from the selected MIDI files."""
    notes = []
    for file_path in selected_files:
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
            print(f"Error processing {file_path}: {e}")
    
    return notes

# Function to generate notes
def generate_notes(model, network_input, pitchnames, n_vocab, total_notes=500):
    """ Generate notes from the trained model """
    if len(network_input) == 0:
        print("No input sequences available. Ensure your dataset has enough notes.")
        return []

    start = np.random.randint(0, len(network_input) - 1)
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    for note_index in range(total_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        
        prediction = model.predict(prediction_input, verbose=0)
        
        index = np.argmax(prediction)
        result = int_to_note.get(index)
        if result:
            prediction_output.append(result)
        
        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

# Function to create MIDI file
def create_midi(prediction_output, file_path='output.mid'):
    """ Convert the output from the prediction to notes and create a midi file from the notes """
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():  # chord
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:  # note
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    try:
        midi_stream.write('midi', fp=file_path)
        print(f"MIDI file saved to: {file_path}")  # Debugging output
    except Exception as e:
        print(f"Error writing MIDI file: {e}")

# Load model and data
midi_files = get_all_midi_files('midi_songs')[:10]  # Use only the first ten MIDI files
print(f"Using {len(midi_files)} MIDI files for generation.")

notes = get_notes(midi_files)
print(f"Number of notes extracted: {len(notes)}")

sequence_length = 100
if len(notes) <= sequence_length:
    print("Not enough notes to generate sequences. Please provide more MIDI files.")
else:
    n_vocab = len(set(notes))
    
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    
    network_input = []
    for i in range(0, len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
    
    print(f"Number of sequences created: {len(network_input)}")
    
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    
    try:
        model = load_model('music_generator_model.h5')  # Replace with your model path
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    total_notes_to_generate = 500  # Adjust as needed
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab, total_notes=total_notes_to_generate)
    if prediction_output:
        create_midi(prediction_output)
