import pandas
import seaborn as sn
import matplotlib.pyplot as plt

df = pandas.read_csv("test_files.csv")

df["processing_speed_pct"] = df["processing_duration_seconds"] / df["input_duration_seconds"]
df = df.drop(columns=["processing_duration_seconds", "input_duration_seconds"])

df["output_filesize_pct"] = df["output_filesize_MB"] / df["input_filesize_MB"]
df = df.drop(columns=["output_filesize_MB", "input_filesize_MB"])

# See 

corr_matrix = df.corr()
corr_matrix = corr_matrix.dropna(axis="index", how="all")
corr_matrix = corr_matrix.dropna(axis="columns")
corr_matrix = corr_matrix.drop(index=["downscale_factor", "dilate_kernel_size", "persist_factor", "movement_threshold"])
corr_matrix = corr_matrix.drop(columns=["processing_speed_pct", "output_filesize_pct"])

corr_matrix = corr_matrix.round(decimals=2)
print(corr_matrix)

sn.heatmap(corr_matrix, annot=True)
plt.tight_layout()
plt.show()