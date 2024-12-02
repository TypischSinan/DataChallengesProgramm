import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
import librosa

txt_folder_path = "C:/Users/Sinan/Desktop/DataChallengesProgramm/data/ICBHI_final_database_txt"
wav_folder_path = "C:/Users/Sinan/Desktop/DataChallengesProgramm/data/ICBHI_final_database_wav"
demographic_file = "C:/Users/Sinan/Desktop/DataChallengesProgramm/data/ICBHI_Challenge_demographic_information.txt"
diagnosis_file = "C:/Users/Sinan/Desktop/DataChallengesProgramm/data/ICBHI_Challenge_diagnosis.txt"
output_csv = "C:/Users/Sinan/Desktop/DataChallengesProgramm/data/output.csv"

demographics_df = pd.read_csv(demographic_file, sep='\t', header=None)
demographics_df.columns = ["Participant ID", "Age",
                           "Sex", "Adult BMI", "Child Weight", "Child Height"]
print("Demographic Data Sample:")
print(demographics_df.head())

diagnosis_df = pd.read_csv(diagnosis_file, sep='\t', header=None)
diagnosis_df.columns = ["Participant ID", "Diagnosis"]
diagnosis_df['Participant ID'] = diagnosis_df['Participant ID'].astype(str)
print("Diagnosis Data Sample:")
print(diagnosis_df.head())


def extract_filename_metadata(filename):
    parts = filename.split('_')
    if len(parts) == 5:
        metadata = {
            "Patient Number": parts[0],
            "Recording Index": parts[1],
            "Chest Location": parts[2],
            "Acquisition Mode": parts[3],
            "Recording Equipment": parts[4].replace(".wav", "")
        }
        return metadata
    else:
        print(f"Error: Invalid filename format for {filename}")
        return {}


def extract_audio_features(file_path, start_time, end_time):
    try:
        sampling_rate, data = wavfile.read(file_path)
        start_sample = int(start_time * sampling_rate)
        end_sample = int(end_time * sampling_rate)
        data = data[start_sample:end_sample]
        data = data / np.max(np.abs(data))
        if len(data.shape) == 2:
            data = np.mean(data, axis=1)
        duration = len(data) / sampling_rate
        rms = np.sqrt(np.mean(data**2))
        zcr = librosa.feature.zero_crossing_rate(data)[0].mean()
        spectral_centroid = librosa.feature.spectral_centroid(
            y=data, sr=sampling_rate).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=data, sr=sampling_rate).mean()
        mfcc = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=13)
        mfcc_means = np.mean(mfcc, axis=1)
        return {
            "Sampling Rate": sampling_rate,
            "Duration": duration,
            "RMS": rms,
            "Zero Crossing Rate": zcr,
            "Spectral Centroid": spectral_centroid,
            "Spectral Bandwidth": spectral_bandwidth,
            **{f"MFCC_{i+1}": mfcc_means[i] for i in range(len(mfcc_means))}
        }
    except Exception as e:
        return {"Error": str(e)}


results = []
wav_files = [f for f in os.listdir(wav_folder_path) if f.endswith('.wav')]

for wav_file in wav_files:
    print(f"Processing file: {wav_file}")
    file_path = os.path.join(wav_folder_path, wav_file)
    metadata = extract_filename_metadata(wav_file)
    if "Patient Number" not in metadata:
        print(f"Skipping file due to missing metadata: {wav_file}")
        continue
    corresponding_txt_file = wav_file.replace(".wav", ".txt")
    txt_file_path = os.path.join(txt_folder_path, corresponding_txt_file)
    if not os.path.exists(txt_file_path):
        print(
            f"Warning: No corresponding TXT file found for {wav_file}. Skipping...")
        continue
    with open(txt_file_path, 'r') as txt_file:
        for line in txt_file:
            columns = line.strip().split('\t')
            if len(columns) == 4:
                beginning_time = float(columns[0])
                end_time = float(columns[1])
                crackles = columns[2]
                wheezes = columns[3]
                audio_features = extract_audio_features(
                    file_path, beginning_time, end_time)
                results.append({
                    "Filename": wav_file,
                    **metadata,
                    "Beginning Time": beginning_time,
                    "End Time": end_time,
                    "Crackles": crackles,
                    "Wheezes": wheezes,
                    **audio_features
                })
print(results)
wav_df = pd.DataFrame(results)
print(f"Columns in wav_df: {wav_df.columns}")

if 'Patient Number' in wav_df.columns:
    wav_df['Patient Number'] = wav_df['Patient Number'].astype(str)
else:
    print("Error: 'Patient Number' column not found in wav_df. Exiting.")
    exit(1)

demographics_df['Participant ID'] = demographics_df['Participant ID'].astype(
    str)
merged_df = pd.merge(wav_df, demographics_df, left_on="Patient Number",
                     right_on="Participant ID", how="outer")
merged_df = pd.merge(merged_df, diagnosis_df, left_on="Patient Number",
                     right_on="Participant ID", how="outer")

if "Participant ID" in merged_df.columns:
    merged_df = merged_df.drop(columns=["Participant ID"])
else:
    print("Warning: 'Participant ID' column not found in merged_df.")

merged_df.to_csv(output_csv, index=False)
print(f"Data successfully saved to {output_csv}")
