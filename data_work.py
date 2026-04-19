import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial import cKDTree

chem_df = pd.read_csv("CALCOFI_DIC_20250122.csv")
# chem_df = chem_df[(chem_df['Year_UTC'] <= 2006) & (chem_df['Year_UTC'] >= 1987)] # keep the year range the same
chem_df = chem_df.iloc[1:].reset_index(drop=True)
chem_df = chem_df.replace(-999, np.nan)
chem_df['datetime'] = pd.to_datetime(
    chem_df['Year_UTC'].astype(int).astype(str) + '-' +
    chem_df['Month_UTC'].astype(int).astype(str) + '-' +
    chem_df['Day_UTC'].astype(int).astype(str) + ' ' +
    chem_df['Time_UTC'].astype(str)
)
chem_df = chem_df.rename(columns={
    'Year_UTC': 'year',
    'Month_UTC': 'month'
})
chem_df = chem_df.drop(columns=['EXPOCODE', 'Ship_Name', 'Station_ID'])
chem_cols = ['DIC', 'TA', 'CTDTEMP_ITS90', 'Salinity_PSS78', 'Latitude', 'Longitude']
for col in chem_cols:
    chem_df[col] = pd.to_numeric(chem_df[col], errors='coerce')

chem_df = chem_df.dropna(subset=chem_cols)

seamap_p_df = pd.read_csv("obis_seamap_dataset_507_points.csv")
seamap_p_df = seamap_p_df.drop(columns=['dataset_id', 'row_id', 'series_id', 'itis_tsn', 'lprecision', 'tprecision', 'notes', 'last_mod', 'timezone', 'provider', 'platform', 'oceano'])
seamap_p_df['date_time'] = pd.to_datetime(seamap_p_df['date_time'], errors='coerce')
seamap_p_df = seamap_p_df.dropna(subset=['date_time'])
seamap_p_df['year'] = seamap_p_df['date_time'].dt.year
seamap_p_df['month'] = seamap_p_df['date_time'].dt.month
# seamap_p_df = seamap_p_df[(seamap_p_df['year'] >= 1987) & (seamap_p_df['year'] <= 2006)]
seamap_p_df = seamap_p_df.dropna(subset=['latitude', 'longitude'])

# model 1
plt.figure(figsize=(8,6))

plt.hexbin(
    seamap_p_df['longitude'],
    seamap_p_df['latitude'],
    gridsize=50,
    cmap='Reds',
    mincnt=1
)

plt.colorbar(label="Observation density")
plt.title("OBIS Sampling Intensity Heatmap")
plt.show()

# model 2
plt.figure(figsize=(8,6))

plt.hexbin(
    chem_df['Longitude'],
    chem_df['Latitude'],
    C=chem_df['DIC'],
    reduce_C_function=np.mean,
    gridsize=50,
    cmap='viridis'
)

plt.colorbar(label="Mean DIC")
plt.title("Ocean Chemistry Regimes (DIC)")
plt.show()

# model 3
plt.figure(figsize=(8,6))

plt.hexbin(
    chem_df['Longitude'],
    chem_df['Latitude'],
    C=chem_df['DIC'],
    reduce_C_function=np.mean,
    gridsize=50,
    cmap='viridis',
    alpha=0.6
)

plt.scatter(
    seamap_p_df['longitude'],
    seamap_p_df['latitude'],
    color='red',
    s=3,
    alpha=0.3
)

plt.colorbar(label="DIC")
plt.title("Marine Observations vs Ocean Chemistry Regimes")
plt.show()

# model 4 (maybe keep)
chem_lat = chem_df.groupby(pd.cut(chem_df['Latitude'], 20))['DIC'].mean()

plt.figure(figsize=(8,5))
plt.plot(chem_lat.values)
plt.title("Latitudinal Gradient of DIC")
plt.xlabel("Latitude bins (south → north)")
plt.ylabel("Mean DIC")
plt.show()

chem_df['DIC_bin'] = pd.qcut(chem_df['DIC'], 5)

bio_density = pd.cut(
    seamap_p_df['longitude'], bins=20
).value_counts().sort_index()