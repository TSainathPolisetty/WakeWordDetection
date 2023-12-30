import os
from pydub import AudioSegment

# Directory containing the .ogg files
directory = "/home/tulasi/eml/P6/P6_submission/Python/"

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".ogg"):
        # Full path of the .ogg file
        ogg_file = os.path.join(directory, filename)

        # Load the .ogg file
        sound = AudioSegment.from_ogg(ogg_file)

        # Define the output filename (change extension to .wav)
        wav_file = os.path.join(directory, os.path.splitext(filename)[0] + ".wav")

        # Export as a .wav file
        sound.export(wav_file, format="wav")
        print(f"Converted {filename} to WAV format.")

print("Conversion completed.")
