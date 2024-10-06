from midiutil import MIDIFile
import numpy as np
from PIL import Image
import colorsys
import random
import cv2
import sys
from scipy.stats import entropy
import scipy.ndimage as ndimage
from itertools import permutations

print("Initiating enhanced cosmic soundscape generation with advanced musical structures...")

def get_pitch_class_set(scale, key):
    pitch_class_set = [note % 12 for note in scale]
    return sorted(list(set(pitch_class_set)))

def add_safe_note(midi, track, channel, pitch, time, duration, volume):
    try:
        midi.addNote(track, channel, pitch, time, duration, volume)
    except ValueError as e:
        print(f"Warning: Could not add note. {e}")

def generate_tone_row(image_data):
    row = list(range(12))
    random.shuffle(row)
    return row

def get_chord_progression(scale, key):
    print(f"Entering get_chord_progression with scale: {scale}, key: {key}")
    
    if isinstance(scale, str):
        scale = list(scale)
    
    if not isinstance(scale, list):
        raise ValueError(f"Scale must be a list or string, got {type(scale)}")
    
    if len(scale) < 5:
        raise ValueError(f"Scale must have at least 5 elements, got {len(scale)}")
    
    if len(scale) == 5: 
        progression = [
            [scale[0], scale[2], scale[4]],  
            [scale[1], scale[3], scale[0]],
            [scale[2], scale[4], scale[1]], 
            [scale[3], scale[0], scale[2]],  
            [scale[4], scale[1], scale[3]]   
        ]
    elif len(scale) == 6: 
        progression = [
            [scale[0], scale[2], scale[5]],  
            [scale[3], scale[5], scale[1]], 
            [scale[4], scale[0], scale[2]], 
            [scale[1], scale[3], scale[5]],  
            [scale[2], scale[4], scale[0]],  
            [scale[5], scale[1], scale[3]]   
        ]
    elif len(scale) == 7: 
        progression = [
            [scale[0], scale[2], scale[4]], 
            [scale[1], scale[3], scale[5]],  
            [scale[2], scale[4], scale[6]],  
            [scale[3], scale[5], scale[0]], 
            [scale[4], scale[6], scale[1]],  
            [scale[5], scale[0], scale[2]],  
            [scale[6], scale[1], scale[3]]  
        ]
    else: 
        progression = [
            [scale[0], scale[2], scale[4]],  
            [scale[1], scale[3], scale[5]],  
            [scale[2], scale[4], scale[6]],  
            [scale[3], scale[5], scale[7]], 
            [scale[4], scale[6], scale[0]], 
            [scale[5], scale[7], scale[1]], 
            [scale[6], scale[0], scale[2]], 
            [scale[7], scale[1], scale[3]]  
        ]
    
    result = random.sample(progression, 4) 
    print(f"Exiting get_chord_progression with result: {result}")
    return result

def serialize_melody(tone_row, length):
    serialized_melody = []
    for i in range(length):
        note = tone_row[i % 12] + (i // 12) * 12
        duration = random.choice([0.25, 0.5, 1])
        serialized_melody.append((note, duration))
    return serialized_melody

def generate_polytonal_progression(scale1, key1, scale2, key2):
    chord_prog1 = get_chord_progression(scale1, key1)
    chord_prog2 = get_chord_progression(scale2, key2)
    return list(zip(chord_prog1, chord_prog2))

def create_polytonal_melody(scale1, key1, scale2, key2, length):
    melody1 = generate_cosmic_melody(scale1, key1, get_chord_progression(scale1, key1), length//2)
    melody2 = generate_cosmic_melody(scale2, key2, get_chord_progression(scale2, key2), length//2)
    return melody1 + melody2

def generate_aleatoric_sequence(length, pitch_range):
    return [(random.randint(pitch_range[0], pitch_range[1]), random.choice([0.25, 0.5, 1])) for _ in range(length)]

def modulate_key(current_scale, current_key):
    print(f"Entering modulate_key with current_scale: {current_scale}, current_key: {current_key}")
    
    
    scales = {
        'major': [0, 2, 4, 5, 7, 9, 11],
        'minor': [0, 2, 3, 5, 7, 8, 10],
        'dorian': [0, 2, 3, 5, 7, 9, 10],
        'phrygian': [0, 1, 3, 5, 7, 8, 10],
        'lydian': [0, 2, 4, 6, 7, 9, 11],
        'mixolydian': [0, 2, 4, 5, 7, 9, 10],
        'locrian': [0, 1, 3, 5, 6, 8, 10],
        'whole_tone': [0, 2, 4, 6, 8, 10],
        'pentatonic': [0, 2, 4, 7, 9],
        'blues': [0, 3, 5, 6, 7, 10]
    }
    
    new_key = random.choice(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
    new_scale_name = random.choice(list(scales.keys()))
    new_scale = scales[new_scale_name]
    
    key_to_midi = {'C': 60, 'C#': 61, 'D': 62, 'D#': 63, 'E': 64, 'F': 65, 'F#': 66, 'G': 67, 'G#': 68, 'A': 69, 'A#': 70, 'B': 71}
    base_note = key_to_midi[new_key]
    
    new_scale_midi = [(base_note + interval) % 12 + 60 for interval in new_scale]
    
    print(f"Exiting modulate_key with new_scale: {new_scale_midi}, new_key: {new_key}_{new_scale_name}")
    return new_scale_midi, f"{new_key}_{new_scale_name}"

def create_modulating_progression(original_scale, original_key, length):
    print(f"Entering create_modulating_progression with original_scale: {original_scale}, original_key: {original_key}, length: {length}")
    
    if isinstance(original_scale, str):
        original_scale = list(original_scale)
    
    if len(original_scale) not in [5, 6, 7, 8]:
        raise ValueError(f"Unexpected scale length. Expected 5, 6, 7, or 8, got {len(original_scale)}")
    
    progression = []
    current_scale, current_key = original_scale, original_key
    for i in range(length):
        print(f"Iteration {i}: current_scale = {current_scale}, current_key = {current_key}")
        if random.random() < 0.2:
            current_scale, current_key = modulate_key(current_scale, current_key)
            print(f"Modulated to: current_scale = {current_scale}, current_key = {current_key}")
        progression.extend(get_chord_progression(current_scale, current_key))
    
    print(f"Exiting create_modulating_progression with progression: {progression}")
    return progression

def generate_ostinato(pitch_class_set, length):
    return [random.choice(pitch_class_set) for _ in range(length)]

def create_layered_ostinatos(scale, key, num_layers):
    pitch_class_set = get_pitch_class_set(scale, key)
    ostinatos = [generate_ostinato(pitch_class_set, random.randint(3, 8)) for _ in range(num_layers)]
    return ostinatos

def generate_motif(pitch_class_set, length=4):
    return [random.choice(pitch_class_set) for _ in range(length)]

def develop_motif(motif, technique):
    if technique == "retrograde":
        return list(reversed(motif))
    elif technique == "inversion":
        return [(12 - note) % 12 for note in motif]
    elif technique == "transposition":
        shift = random.randint(1, 11)
        return [(note + shift) % 12 for note in motif]
    else:
        return motif

def get_advanced_chord_progression(scale, key):
    if len(scale) < 5:
        raise ValueError("Scale must have at least 5 notes")
    
    if len(scale) == 5: 
        progression = [
            [scale[0], scale[2], scale[4]], 
            [scale[1], scale[3], scale[0]], 
            [scale[2], scale[4], scale[1]], 
            [scale[3], scale[0], scale[2]],
            [scale[4], scale[1], scale[3]]   
        ]
    else: 
        progression = [
            [scale[0], scale[2], scale[4]],
            [scale[1], scale[3], scale[5]],  
            [scale[2], scale[4], scale[6]], 
            [scale[3], scale[5], scale[0]],
            [scale[4], scale[6], scale[1]], 
            [scale[5], scale[0], scale[2]],
            [scale[6], scale[1], scale[3]]  
        ]
    
    return progression

def get_image_data(image_path):
    try:
        im = Image.open(image_path)
        im = im.convert('RGB') 
        rgb_array = np.array(im, dtype=np.float32) / 255.0
        hsv_array = np.array([colorsys.rgb_to_hsv(*pixel) for pixel in rgb_array.reshape(-1, 3)]).reshape(rgb_array.shape)
        luminance = np.mean(rgb_array, axis=2)
        return rgb_array, hsv_array, luminance, im
    except Exception as e:
        print(f"Error in get_image_data: {e}")
        sys.exit(1)

def measure_image_complexity(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    complexity = np.mean(edges) / 255.0
    return complexity

def get_tempo(image):

    image_array = np.array(image)
    

    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    

    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    

    hue_mean = np.mean(hsv[:,:,0])
    saturation_mean = np.mean(hsv[:,:,1])
    value_mean = np.mean(hsv[:,:,2])
    

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    complexity = np.sum(edges) / (image_array.shape[0] * image_array.shape[1])
    

    brightness = np.mean(image_array)

    image_array = (image_array * 255).astype(np.uint8)
    

    tempo_base = (
        60 +  
        hue_mean * 0.5 +  
        saturation_mean * 0.2 + 
        value_mean * 0.1 +  
        complexity * 50 +  
        brightness * 0.1  
    )
    

    tempo_variation = random.randint(-30, 70)
    

    final_tempo = max(40, min(240, int(tempo_base + tempo_variation)))
    
    return final_tempo

def get_stellar_time_signature(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None and len(lines) > 10:
        return random.choice([(3, 4), (4, 4), (5, 4), (6, 8), (7, 8)])
    return 4, 4


    print(f"Added effects - Reverb: {reverb}, Chorus: {chorus}, Modulation: {modulation}")
def get_galactic_key_and_scale(hsv_array):
    if hsv_array.ndim == 3:
        hue_mean = np.mean(hsv_array[:,:,0])
    elif hsv_array.ndim == 2:
        hue_mean = np.mean(hsv_array)
    else:
        raise ValueError("Unexpected array dimensions")
    
    scale_type = random.choices([
        "major", "minor", "dorian", "phrygian", "lydian", "mixolydian", 
        "locrian", "whole_tone", "diminished", "augmented",
        "pentatonic_major", "pentatonic_minor", "blues"
    ], weights=[8, 5, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 3])[0]
    
    base_note = int(hue_mean * 12) + 60
    
    scales = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
        "dorian": [0, 2, 3, 5, 7, 9, 10],
        "phrygian": [0, 1, 3, 5, 7, 8, 10],
        "lydian": [0, 2, 4, 6, 7, 9, 11],
        "mixolydian": [0, 2, 4, 5, 7, 9, 10],
        "locrian": [0, 1, 3, 5, 6, 8, 10],
        "whole_tone": [0, 2, 4, 6, 8, 10],
        "diminished": [0, 2, 3, 5, 6, 8, 9, 11],
        "augmented": [0, 3, 4, 7, 8, 11],
        "pentatonic_major": [0, 2, 4, 7, 9],
        "pentatonic_minor": [0, 3, 5, 7, 10],
        "blues": [0, 3, 5, 6, 7, 10]
    }
    
    scale = [base_note + interval for interval in scales[scale_type]]
    
    return [note % 128 for note in scale], scale_type

def get_tempo(image):

    image_array = np.array(image).astype(np.uint8)
    

    if image_array.shape[-1] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    

    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    

    hue_mean = np.mean(hsv[:,:,0])
    saturation_mean = np.mean(hsv[:,:,1])
    value_mean = np.mean(hsv[:,:,2])

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    complexity = np.sum(edges) / (image_array.shape[0] * image_array.shape[1])

    brightness = np.mean(image_array)
    

    tempo_base = (
        60 + 
        hue_mean * 0.5 +  
        saturation_mean * 0.2 + 
        value_mean * 0.1 +  
        complexity * 50 +  
        brightness * 0.1  
    )
    
    tempo_variation = random.randint(-30, 70)

    final_tempo = max(40, min(240, int(tempo_base + tempo_variation)))
    
    return final_tempo

def create_advanced_chord_progression(scale, key, length):
    base_progression = get_advanced_chord_progression(scale, key)
    
    progression = []
    for _ in range(length // 4):
        if random.random() < 0.6:  
            progression.extend(base_progression[:4])
        else:

            chord_choices = random.sample(base_progression, 4)
            if random.random() < 0.3: 
                secondary_dominant = random.choice(base_progression[1:])
                chord_choices.insert(random.randint(0, 3), secondary_dominant)
            progression.extend(chord_choices[:4])
    
    return progression[:length]

def generate_cosmic_melody(scale, key, chord_progression, complexity, length=64):
    melody = []
    for i in range(length):
        if random.random() < 0.7: 
            if key == "whole_tone":
                note = random.choice(scale)
            elif key in ["pentatonic_major", "pentatonic_minor", "blues"]:
                note = random.choice(scale + [scale[0]+12])  
            elif key == "diminished":
                note = random.choice(scale + [scale[random.randint(0,3)]+12])  
            elif key == "augmented":
                note = random.choice([scale[0], scale[1], scale[2], scale[0]+12, scale[1]+12, scale[2]+12])
            else:
                note = random.choice(scale)
            
            duration = random.choice([0.25, 0.5, 1, 1.5])
            melody.append((note, duration))
        else:
            melody.append((0, 0.25))  
    return melody

def create_nebula_chords(chord_progression, length=32):
    return [chord_progression[i % len(chord_progression)] for i in range(length)]

def select_cosmic_instrument(image_features):
    instruments = {
        "piano": [0, 1, 2, 3, 4, 5, 6, 7], 
        "strings": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49], 
        "pad": [88, 89, 90, 91, 92, 93, 94, 95], 
        "bass": [32, 33, 34, 35, 36, 37, 38, 39], 
        "lead": [80, 81, 82, 83, 84, 85, 86, 87], 
        "percussion": [112, 113, 114, 115, 116, 117, 118, 119], 
        "ethnic": [104, 105, 106, 107, 108, 109, 110, 111],  
        "ensemble": [48, 49, 50, 51, 52, 53, 54, 55] 
    }
    

    complexity = image_features['complexity']
    brightness = image_features['brightness']
    
    selected = {}
    

    if brightness > 0.7:
        selected["melody"] = random.choice(instruments["piano"] + instruments["lead"])
    elif brightness < 0.3:
        selected["melody"] = random.choice(instruments["pad"] + instruments["strings"])
    else:
        selected["melody"] = random.choice(instruments["piano"] + instruments["strings"] + instruments["lead"])
    

    selected["bass"] = random.choice(instruments["bass"])
    

    if complexity > 0.7:
        selected["harmony"] = random.choice(instruments["pad"] + instruments["ensemble"])
    else:
        selected["harmony"] = random.choice(instruments["strings"] + instruments["pad"])
    

    if complexity > 0.5:
        selected["counter_melody"] = random.choice(instruments["lead"] + instruments["ethnic"])
        selected["rhythm"] = random.choice(instruments["percussion"])
        selected["arpeggio"] = random.choice(instruments["piano"] + instruments["lead"])
        selected["ambient"] = random.choice(instruments["pad"] + instruments["strings"])
    
    return selected

def get_cosmic_volume(luminance):
    return int(48 + luminance.mean() * 47)

def generate_texture_based_piano(scale, key, chord_progression, length=64):
    base_note = scale[0]
    texture_melody = []
    for i in range(length):
        chord = chord_progression[i % len(chord_progression)]
        note = random.choice(chord)
        note = max(0, min(127, note))
        duration = 0.5
        velocity = random.randint(40, 80)
        texture_melody.append((note, duration, velocity))
    return texture_melody

def generate_jitter_strings(scale, key, length=64):
    base_note = scale[0]
    scale_intervals = [0, 2, 4, 5, 7, 9, 11] if key == "major" else [0, 2, 3, 5, 7, 8, 10]
    strings = []
    for _ in range(length):
        note = base_note + random.choice(scale_intervals)
        note = max(0, min(127, note))
        duration = 0.5
        velocity = random.randint(2, 10)
        strings.append((note, duration, velocity))
    return strings

def generate_metro_bass(chord_progression, length=64):
    return [(max(0, min(127, chord_progression[i % len(chord_progression)][0] - 12)), 1.0, 70) for i in range(length)]

def generate_dream_voice(chord_progression, length=16):
    dream = []
    for i in range(length):
        chord = chord_progression[i % len(chord_progression)]
        for note in chord:
            note = max(0, min(127, note))
            dream.append((note, 4.0, 10))
    return dream

def generate_pulsating_waves(scale, key, chord_progression, length=64):
    base_note = scale[0]
    scale_intervals = [0, 2, 4, 5, 7, 9, 11] if key == "major" else [0, 2, 3, 5, 7, 8, 10]
    waves = []
    for i in range(length):
        chord = chord_progression[i % len(chord_progression)]
        if i % 8 == 0:
            note = random.choice(chord) + 12
            note = max(0, min(127, note))
            velocity = random.randint(5, 15)
        else:
            note = 0
            velocity = 0
        waves.append((note, 0.25, velocity))
    return waves

def generate_sonata_form(scale, key, chord_progression, complexity, image_features, length=256):
    theme1 = generate_cosmic_melody(scale, key, chord_progression[:8], complexity, length=32)
    theme1 = [(adjust_octave(note, image_features), duration) for note, duration in theme1]
    

    if len(scale) == 5:
        theme2 = generate_cosmic_melody(scale, key, chord_progression[8:16], complexity, length=32)
    else:

        dominant_scale = [note + 7 for note in scale]
        theme2 = generate_cosmic_melody(dominant_scale, key, chord_progression[8:16], complexity, length=32)
    theme2 = [(adjust_octave(note, image_features), duration) for note, duration in theme2]
    

    development = generate_cosmic_melody(scale, key, chord_progression[16:32], complexity, length=64)
    development = [(adjust_octave(note, image_features), duration) for note, duration in development]
    

    recap_theme1 = generate_cosmic_melody(scale, key, chord_progression[:8], complexity, length=32)
    recap_theme1 = [(adjust_octave(note, image_features), duration) for note, duration in recap_theme1]
    recap_theme2 = generate_cosmic_melody(scale, key, chord_progression[8:16], complexity, length=32)
    recap_theme2 = [(adjust_octave(note, image_features), duration) for note, duration in recap_theme2]
    

    coda = generate_cosmic_melody(scale, key, chord_progression[-8:], complexity, length=64)
    coda = [(adjust_octave(note, image_features), duration) for note, duration in coda]
    
    return theme1 + theme2 + development + recap_theme1 + recap_theme2 + coda

def generate_fractal_melody(image, scale, length):
    edges = cv2.Canny(image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    melody = []
    for contour in contours[:length]:
        for point in contour:
            x, y = point[0]
            note = scale[x % len(scale)]
            duration = (y % 4) + 1
            melody.append((note, duration))
    
    return melody[:length]

def generate_bass_line(chord_progression, length):
    bass_line = []
    for chord in chord_progression:
        bass_note = chord[0] - 12 
        duration = 1  
        bass_line.append((bass_note, duration))
    

    if len(bass_line) < length:
        bass_line = bass_line * (length // len(bass_line)) + bass_line[:length % len(bass_line)]
    elif len(bass_line) > length:
        bass_line = bass_line[:length]
    
    return bass_line

def generate_harmony(chord_progression, length):
    harmony = []
    for chord in chord_progression:

        harmony.append((chord, 1))  
    

    if len(harmony) < length:
        harmony = harmony * (length // len(harmony)) + harmony[:length % len(harmony)]
    elif len(harmony) > length:
        harmony = harmony[:length]
    
    return harmony

def add_image_based_effects(midi, hsv_array):

    avg_hue = np.mean(hsv_array[:,:,0])
    avg_saturation = np.mean(hsv_array[:,:,1])
    avg_value = np.mean(hsv_array[:,:,2])


    reverb = int(avg_hue * 127 / 360)
    

    chorus = int(avg_saturation * 127)

    modulation = int(avg_value * 127)


    midi.addControllerEvent(0, 0, 0, 91, reverb)  
    midi.addControllerEvent(0, 0, 0, 93, chorus) 
    midi.addControllerEvent(0, 0, 0, 1, modulation) 

def adjust_octave(note, image_features):
    brightness = image_features['brightness']
    complexity = image_features['complexity']
    

    if brightness < 0.3:
        octave_shift = -2 
    elif brightness < 0.6:
        octave_shift = -1
    elif brightness > 0.8:
        octave_shift = 1  
    else:
        octave_shift = 0
    
    if complexity > 0.7:
        octave_shift += random.choice([-1, 0, 1]) 
    
    new_note = note + (octave_shift * 12)

    return max(0, min(127, new_note))

def create_cosmic_midi(image_path, output_file):
    try:
        rgb_array, hsv_array, luminance, im = get_image_data(image_path)
        complexity = measure_image_complexity(im)
        brightness = np.mean(luminance)
    
        image_features = {
            'complexity': complexity,
            'brightness': brightness
        }
        
        time_sig_num, time_sig_den = get_stellar_time_signature(im)
        scale, key = get_galactic_key_and_scale(hsv_array)
        print(f"Scale from get_galactic_key_and_scale: {scale}, Key: {key}")
        tempo = get_tempo(im)
        print(f"Tempo: {tempo} BPM")

        chord_progression = create_modulating_progression(scale, key, 16)

        num_tracks = min(int(complexity * 8) + 3, 8) 
        midi = MIDIFile(num_tracks)
        midi.addTempo(0, 0, tempo)
        midi.addTimeSignature(0, 0, time_sig_num, time_sig_den, 24)
        add_image_based_effects(midi, hsv_array)

        base_volume = get_cosmic_volume(luminance)
        instruments = select_cosmic_instrument(image_features)

        # Main melody
        midi.addProgramChange(0, 0, 0, instruments["melody"])
        sonata_melody = generate_sonata_form(scale, key, chord_progression, complexity, image_features, length=256)
        time = 0
        for note, duration in sonata_melody:
            for note, duration in sonata_melody:
                if note != 0:
                    add_safe_note(midi, 0, 0, note, time, duration * 4, int(base_volume * 0.9))
                time += duration * 4

        # Bass line
        midi.addProgramChange(1, 0, 0, instruments["bass"])
        bass_line = generate_bass_line(chord_progression, len(sonata_melody))
        bass_line = [(adjust_octave(max(note - 24, 0), image_features), duration) for note, duration in bass_line]
        time = 0
        for note, duration in bass_line:
            if note != 0:
                add_safe_note(midi, 1, 0, note, time, duration * 4, int(base_volume * 0.8))
            time += duration * 4

        # Harmony
        midi.addProgramChange(2, 0, 0, instruments["harmony"])
        harmony = generate_harmony(chord_progression, len(sonata_melody))
        time = 0
        for chord, duration in harmony:
            for note in chord:
                midi.addNote(2, 0, note, time, duration * 4, int(base_volume * 0.6))
            time += duration * 4

        # Counter melody
        if num_tracks > 3 and "counter_melody" in instruments:
            midi.addProgramChange(3, 0, 0, instruments["counter_melody"])
            counter_melody = generate_counter_melody(scale, key, sonata_melody)
            time = 0
            for note, duration in counter_melody:
                if note != 0:
                    midi.addNote(3, 0, note, time, duration * 4, int(base_volume * 0.7))
                time += duration * 4

        # Rhythm
        if num_tracks > 4 and "rhythm" in instruments:
            midi.addProgramChange(4, 0, 0, instruments["rhythm"])
            rhythm = generate_rhythm_pattern(time_sig_num, time_sig_den, len(sonata_melody))
            time = 0
            for note, duration in rhythm:
                if note != 0:
                    midi.addNote(4, 0, note, time, duration * 4, int(base_volume * 0.8))
                time += duration * 4

        # Arpeggios
        if num_tracks > 5 and "arpeggio" in instruments:
            midi.addProgramChange(5, 0, 0, instruments["arpeggio"])
            arpeggios = generate_arpeggios(chord_progression, len(sonata_melody))
            time = 0
            for note, duration in arpeggios:
                if note != 0:
                    midi.addNote(5, 0, note, time, duration * 4, int(base_volume * 0.6))
                time += duration * 4

        # Ambient pad
        if num_tracks > 6 and "ambient" in instruments:
            midi.addProgramChange(6, 0, 0, instruments["ambient"])
            ambient_pad = generate_ambient_pad(chord_progression, len(sonata_melody))
            time = 0
            for chord, duration in ambient_pad:
                for note in chord:
                    midi.addNote(6, 0, note, time, duration * 4, int(base_volume * 0.5))
                time += duration * 4

        # tempo changes
        for i in range(3):
            change_point = random.randint(64, 192)
            new_tempo = int(tempo * random.uniform(0.95, 1.05))  
            midi.addTempo(0, change_point, new_tempo)

        with open(output_file, "wb") as output_midi:
            midi.writeFile(output_midi)
        print(f"Harmonious cosmic MIDI soundscape created: {output_file}")

    except Exception as e:
        print(f"Error in create_cosmic_midi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Initiating enhanced cosmic soundscape generation...")
    try:
        create_cosmic_midi("data/images/8.webp", "data/midi/cosmic_output_harmonious3.mid")
        print("Cosmic soundscape generation complete. Output saved to 'cosmic_output_harmonious3.mid'.")
    except Exception as e:
        print(f"An anomaly occurred during cosmic soundscape generation: {e}")
