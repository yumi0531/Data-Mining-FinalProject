import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("../playground-series-s5e6/train.csv")

df_sample = df.copy()

crop_counts = df_sample["Crop Type"].value_counts()
rare_crops = crop_counts[crop_counts < 500].index
df_sample["Crop Type Simplified"] = df_sample["Crop Type"].replace(rare_crops, "Other")

soil_counts = df_sample["Soil Type"].value_counts()
rare_soils = soil_counts[soil_counts < 500].index
df_sample["Soil Type Simplified"] = df_sample["Soil Type"].replace(rare_soils, "Other")

df_sample["Crop_Soil"] = df_sample["Crop Type"] + "_" + df_sample["Soil Type"]
crop_soil_counts = df_sample["Crop_Soil"].value_counts()
rare_combinations = crop_soil_counts[crop_soil_counts < 50].index
df_sample["Crop_Soil"] = df_sample["Crop_Soil"].replace(rare_combinations, "Other")

df_sample.to_csv("enhanced_dataset.csv", index=False)
print("已儲存檔案 enhanced_dataset.csv")


