import warnings
import os
from zipfile import ZipFile
import numpy as np
import pandas as pd
import logging
from dbfread import DBF
from shapely.geometry import Polygon
import geopandas as gpd
from shapely.geometry import box

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


#### Take out unique Chlorophyll coordinates

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:

    # Importing all csv files and concating to one
    logging.info("Importing all csv files and concating to one...")
    chlMonth = pd.read_csv()
    
    # Importing all csv files and concating to one
    logging.info("Select columns and keep unique rows...")
    chlUnique = chlMonth[['xCoor', 'yCoor', 'year', 'month', 'chl']]

    logging.info("ALL PROCESS COMPLETED")

except Exception as e:
    logging.error(f"An error occurred: {e}")

#### Constructing chlorophyll and 5km grid geodata

try:
    # Path configurations
    data_dir = ""
    zip_file_path = os.path.join(data_dir, "zip")
    dbf_file_name = 'dbf'
    temp_directory = os.path.join(data_dir, "temp_directory")
    
    # Ensure temporary directory exists
    os.makedirs(temp_directory, exist_ok=True)
    
    # Extract the DBF file from the ZIP
    logging.info("Extracting DBF file from ZIP...")
    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extract(dbf_file_name, temp_directory)

    # Read the DBF file using dbfread with specified encoding
    logging.info("Reading DBF file...")
    dbf_path = os.path.join(temp_directory, dbf_file_name)
    table = DBF(dbf_path, encoding='cp949')
    ChlGrid5km = pd.DataFrame(iter(table))
    
    # Select and rename columns
    logging.info("Data constructing before exporting...")
    ChlGeo = ChlGrid5km[['id','year','month','chl']]
    ChlGeo = ChlGeo.groupby(['id','year','month'])['chl'].mean().reset_index()
    ChlGeo = ChlGeo.drop_duplicates(subset='id')
    ChlGeo = ChlGeo.sort_values(by=['id','year','month'])
    
    logging.info("ALL PROCESS COMPLETED")

except Exception as e:
    logging.error(f"An error occurred: {e}")


#### Making mean altitude by region

try:
    # Path configurations
    data_dir = ""
    zip_file_path = os.path.join(data_dir, "zip")
    dbf_file_name = 'dbf'
    temp_directory = os.path.join(data_dir, "temp_directory")
    
    # Ensure temporary directory exists
    os.makedirs(temp_directory, exist_ok=True)
    
    # Extract the DBF file from the ZIP
    logging.info("Extracting DBF file from ZIP...")
    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extract(dbf_file_name, temp_directory)

    # Read the DBF file using dbfread with specified encoding
    logging.info("Reading DBF file...")
    dbf_path = os.path.join(temp_directory, dbf_file_name)
    table = DBF(dbf_path, encoding='cp949')
    RegionGrid5kmDEM = pd.DataFrame(iter(table))
    
    # Select and rename columns
    logging.info("Calculating mean altitude by each grids...")
    RegionDEM = RegionGrid5kmDEM.copy()
    RegionDEM['mean_altitude'] = RegionGrid5kmDEM.groupby('id')['altitude'].transform('mean')
    logging.info("Constructing dataframe...")
    RegionDEM = RegionDEM.drop(columns=['altitude','layer','path'])
    RegionDEM = RegionDEM.drop_duplicates(subset='id')
    
    # Generate date ranges correctly
    logging.info("Calculating geometry...")
    RegionDEM['geometry'] = RegionDEM.apply(lambda row: box(row['left'], row['bottom'], row['right'], row['top']), axis=1)
    # Create a GeoDataFrame
    RegionDEM = gpd.GeoDataFrame(RegionDEM, geometry='geometry')
    
    # Define a function to calculate geographic centers
    logging.info("Calculating geographic centers...")
    def calculate_center(row):
        points = [(row['left'], row['top']), (row['right'], row['top']), (row['right'], row['bottom']),
                  (row['left'], row['bottom'])]
        return Polygon(points).centroid.coords[0]

    # Apply function and expand the results into two separate columns
    RegionDEM[['center_x', 'center_y']] = RegionDEM.apply(lambda row: calculate_center(row), axis=1, result_type='expand')
    
    # selecting columnms
    logging.info("Selecting columns...")
    RegionDEM = RegionDEM[['id', 'mean_altitude',
                            'CTPRVN_CD','SIG_CD','EMD_CD',
                            'CTP_ENG_NM','SIG_ENG_NM','EMD_ENG_NM',
                            'CTP_KOR_NM','SIG_KOR_NM','EMD_KOR_NM',
                            'center_x','center_y','geometry'
                        ]]

    logging.info("ALL PROCESS COMPLETED")

except Exception as e:
    logging.error(f"An error occurred: {e}")

try:
    
    # Generate date ranges correctly
    logging.info("Creating DataFrame of years and specific months...")
    dates = pd.date_range(start='2006-01-01', end='2024-01-01', freq='MS')  # Monthly start frequency
    base = pd.DataFrame({'date': dates})
    
    # Cross join with df
    logging.info("Merging with imported DataFrame...")
    region_base = pd.merge(base.assign(key=1), RegionDEM.assign(key=1), on='key').drop(columns='key')
    
    # Constructing dataframe
    logging.info("Data constructing before exporting...")
    region_base['year'] = region_base['date'].dt.year
    region_base['month'] = region_base['date'].dt.month
    region_base = region_base[['id', 'mean_altitude',
                            'CTPRVN_CD','SIG_CD','EMD_CD',
                            'CTP_ENG_NM','SIG_ENG_NM','EMD_ENG_NM',
                            'CTP_KOR_NM','SIG_KOR_NM','EMD_KOR_NM',
                            'center_x','center_y','geometry','date','year','month'
                            ]]

    logging.info("ALL PROCESS COMPLETED")

except Exception as e:
    logging.error(f"An error occurred: {e}")


#### Combining chl & region dataframe

try:
    
    # Export to csv file
    logging.info("Merging chlGeo & region_base dataframe...")
    finalDf = pd.merge(region_base, ChlGeo, how='left', on=['id','year','month'])    
    logging.info("ALL PROCESS COMPLETED")

except Exception as e:
    logging.error(f"An error occurred: {e}")
