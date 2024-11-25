import os
import pandas as pd

# Define the paths
folder_path = "C:/Users/Sinan/Desktop/DataChallengesProgramm/data/ICBHI_final_database"
demographic_file = "data/ICBHI_Challenge_demographic_information.txt"
output_csv = "data/output.csv"  # Output CSV file
diagnosis_file = "data/ICBHI_Challenge_diagnosis.txt"  # Diagnosis file

# Load demographic data
demographics_df = pd.read_csv(demographic_file, sep='\t', header=None)
demographics_df.columns = ["Participant ID", "Age",
                           "Sex", "Adult BMI", "Child Weight", "Child Height"]
print("Demographic Data Sample:")
print(demographics_df.head())

# Load diagnosis data
diagnosis_df = pd.read_csv(diagnosis_file, sep='\t', header=None)
diagnosis_df.columns = ["Participant ID", "Diagnosis"]
print("Diagnosis Data Sample:")
print(diagnosis_df.head())

# Initialize a list to hold all data
data = []

# Loop through each txt file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        # Split the filename by underscores to extract metadata
        parts = filename.split('_')
        if len(parts) == 5:
            patient_number = parts[0]
            recording_index = parts[1]
            chest_location = parts[2]
            acquisition_mode = parts[3]
            recording_equipment = parts[4].replace(
                ".txt", "")  # Remove the .txt extension

            # Get demographic info for the current patient
            demographic_info = demographics_df[demographics_df["Participant ID"] == int(
                patient_number)]
            if demographic_info.empty:
                # If no demographic info found, set all to "NA"
                age = sex = adult_bmi = child_weight = child_height = "NA"
            else:
                # Otherwise, get the demographic info, replacing missing data with "NA"
                demographic_info = demographic_info.iloc[0]
                age = demographic_info["Age"] if pd.notna(
                    demographic_info["Age"]) else "NA"
                sex = demographic_info["Sex"] if pd.notna(
                    demographic_info["Sex"]) else "NA"
                adult_bmi = demographic_info["Adult BMI"] if pd.notna(
                    demographic_info["Adult BMI"]) else "NA"
                child_weight = demographic_info["Child Weight"] if pd.notna(
                    demographic_info["Child Weight"]) else "NA"
                child_height = demographic_info["Child Height"] if pd.notna(
                    demographic_info["Child Height"]) else "NA"

            # Get diagnosis info for the current patient
            diagnosis_info = diagnosis_df[diagnosis_df["Participant ID"] == int(
                patient_number)]
            diagnosis = diagnosis_info["Diagnosis"].iloc[0] if not diagnosis_info.empty else "NA"

            # Open the txt file and read each line
            with open(os.path.join(folder_path, filename), 'r') as file:
                for line in file:
                    # Split the line into columns
                    columns = line.strip().split('\t')
                    if len(columns) == 4:
                        # Add metadata, demographic info, and diagnosis to the line data
                        row = columns + [
                            patient_number,
                            recording_index,
                            chest_location,
                            acquisition_mode,
                            recording_equipment,
                            age,
                            sex,
                            adult_bmi,
                            child_weight,
                            child_height,
                            diagnosis
                        ]
                        # Append the row to the data list
                        data.append(row)

# Define column headers, including the diagnosis column
columns = [
    "Beginning of respiratory cycle(s)",
    "End of respiratory cycle(s)",
    "Presence/absence of crackles",
    "Presence/absence of wheezes",
    "Patient number",
    "Recording index",
    "Chest location",
    "Acquisition mode",
    "Recording equipment",
    "Age",
    "Sex",
    "Adult BMI",
    "Child Weight",
    "Child Height",
    "Diagnosis"
]

# Convert data to a DataFrame
df = pd.DataFrame(data, columns=columns)

# Save DataFrame to CSV
df.to_csv(output_csv, index=False)

print(f"Data successfully saved to {output_csv}")
