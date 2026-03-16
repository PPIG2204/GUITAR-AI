import jams

def extract_guitarset_data(file_path):
    # Load the JAMS file
    jam = jams.load(file_path)

    print(f"--- Extracting data from: {file_path} ---")

    # 1. Extract Note Annotations (Pitch + Duration)
    # GuitarSet typically has 6 note_midi annotations (one per string)
    note_annons = jam.search(namespace='note_midi')
    
    print(f"\nFound {len(note_annons)} string tracks.")
    for i, string_track in enumerate(note_annons):
        print(f"\nString {i+1} Data (First 5 notes):")
        for obs in string_track.data[:5]:
            print(f"  Time: {obs.time:.2f}s | Duration: {obs.duration:.2f}s | MIDI Pitch: {obs.value:.2f}")

    # 2. Extract Tempo
    tempo_annons = jam.search(namespace='tempo')
    if tempo_annons:
        tempo = tempo_annons[0].data[0].value
        print(f"\nDetected Tempo: {tempo} BPM")

    # 3. Extract Metadata
    metadata = jam.file_metadata
    print(f"\nMetadata: Duration: {metadata.duration:.2f}s")

# Example usage:
# extract_guitarset_data('00_BN1-129-Eb_comp.jams')
extract_guitarset_data(r"E:\Personal\Sound to Guitar Tabs\GUITAR-AI\data\annotation\00_BN1-129-Eb_comp.jams")