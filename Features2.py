import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

input_folder = "W:\\Data\\LA\\LA\\ASVspoof2019_LA_train\\flac"
output_folder_fake = "W:\\workdir2\\CQT\\train\\fake"
output_folder_real = "W:\\workdir2\\CQT\\train\\real"
def pad(x, max_len=48000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x
if not os.path.exists(output_folder_fake):
    os.makedirs(output_folder_fake)
if not os.path.exists(output_folder_real):
    os.makedirs(output_folder_real)
# Parameters for mel spectrogram calculation
#hop_length = 512
factor = 1
sr = 16000
# Function to save mel spectrogram as PNG image
def save_cqt_spectrogram(input_file, label):
    print(f"Processing file: {input_file}")

    signal, sr = librosa.load(input_file)
    signal = pad(signal)

    audiolength = librosa.get_duration(y=signal, sr=sr)
    cqt = librosa.core.cqt(y=signal, sr=sr)
    cqt_spectrogram = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    plt.figure(figsize=(factor * audiolength, 1))
    plt.axis('off')
    plt.imshow(cqt_spectrogram, cmap='magma', aspect='auto', extent=[0, 1, 0, 1])

    output_folder = output_folder_fake if label == 'spoof' else output_folder_real
    output_filename = f"{os.path.splitext(os.path.basename(input_file))[0]}.png"
    output_path = os.path.join(output_folder, output_filename)

    print(f"Saving as {label}: {output_path}")
    plt.savefig(output_path, dpi=224, bbox_inches="tight", pad_inches=0)
    plt.close()

# Process each audio file in the input folder
label_file = "W:\\Data\\LA\\LA\\ASVspoof2019_LA_cm_protocols\\ASVspoof2019.LA.cm.train.trn.txt"
with open(label_file, 'r') as labels:
    label_data = labels.readlines()
# Process each audio file in the input folder
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".flac"):
            input_file = os.path.join(root, file)
            filename = os.path.splitext(os.path.basename(input_file))[0]
            for line in label_data:
                parts = line.strip().split()
                file_name = parts[1] + ".flac"  # Extracting the audio file name
                label = parts[4]  # Extracting the label from the 5th part
                if file_name == filename:
                    print(f"Match found: {file_name}, {label}")
                    save_cqt_spectrogram(input_file, label)
                    break  # Stop searching for labels once found