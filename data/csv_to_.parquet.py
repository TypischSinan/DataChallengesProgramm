import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Überprüfe das aktuelle Arbeitsverzeichnis
current_dir = os.path.dirname(os.path.abspath(
    __file__))  # Verzeichnis der .py Datei
csv_path = os.path.join(current_dir, "output.csv")

# Lade die CSV-Datei
df = pd.read_csv(csv_path)

# Erstelle die Parquet-Datei
table = pa.Table.from_pandas(df)
parquet_path = os.path.join(current_dir, "dataset.parquet")
pq.write_table(table, parquet_path)
