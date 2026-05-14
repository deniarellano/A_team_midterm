!pip install census

import census
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Proj
import matplotlib.pyplot as plt
from google.colab import userdata
!git clone https://github.com/deniarellano/A_team_midterm.git


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.2f}'.format # avoid scientific notation

home = str(Path.home())
input_path = home+'/deniarellano/A_team_midterm/data/raw/'
output_path = home+'/deniarellano/A_team_midterm/notebooks/outputs'

# ==========================================================================
# Set API Key
# ==========================================================================

key = '67e7ebf6219bef72eb8e45f877a1abf9f8c91115' #insert your API key here!
c = census.Census(key)

# Bay Area Geography
# ==========================================================================

city_name = 'Bay Area'
state     = '06'                          # California
FIPS      = ['001',                       # Alameda
             '075',                       # San Francisco
             '081',                       # San Mateo
             '085']                       # Santa Clara

sql_query = 'state:{} county:*'.format(state)

def filter_FIPS(df):
    """Keep only the four target counties."""
    return df[df['county'].isin(FIPS)]

# ==========================================================================
# Download ACS 2023 5-Year Estimates  (replaces 2018 pull)
# ==========================================================================

df_vars_23 = [
    'B03002_001E',   # total population
    'B03002_003E',   # non-Hispanic white alone
    'B19001_001E',   # households (income table universe)
    'B19013_001E',   # median household income
    'B25077_001E',   # median home value
    'B25077_001M',   # median home value margin of error
    'B25064_001E',   # median gross rent
    'B25064_001M',   # median gross rent margin of error
    'B15003_001E',   # population 25+ (educational attainment universe)
    'B15003_022E',   # bachelor's degree
    'B15003_023E',   # master's degree
    'B15003_024E',   # professional school degree
    'B15003_025E',   # doctorate degree
    'B25034_001E',   # total housing units by year built
    'B25034_010E',   # built 1940–1949
    'B25034_011E',   # built 1939 or earlier
    'B25003_002E',   # owner-occupied housing units
    'B25003_003E',   # renter-occupied housing units
    'B25105_001E',   # median monthly housing costs
    'B06011_001E',   # median income in the past 12 months (place of birth)
]

# Household income brackets (B19001_002E–B19001_017E)
var_str = 'B19001'
for i in range(1, 18):
    df_vars_23.append(var_str + '_' + str(i).zfill(3) + 'E')

# Migration by income (B07010) — same sub-variable structure as original
var_str = 'B07010'
for i in (list(range(25, 34)) +
          list(range(36, 45)) +
          list(range(47, 56)) +
          list(range(58, 67))):
    df_vars_23.append(var_str + '_' + str(i).zfill(3) + 'E')

# Run API query — ACS 2023 5-year
var_dict_acs5 = c.acs5.get(
    df_vars_23,
    geo={'for': 'tract:*', 'in': sql_query},
    year=2023
)

# Convert to DataFrame, build FIPS, filter counties
df_vars_23 = pd.DataFrame.from_dict(var_dict_acs5)
df_vars_23['FIPS'] = df_vars_23['state'] + df_vars_23['county'] + df_vars_23['tract']
df_vars_23 = filter_FIPS(df_vars_23)

# Rename — suffix _23 throughout
df_vars_23 = df_vars_23.rename(columns={
    'B03002_001E': 'pop_23',
    'B03002_003E': 'white_23',
    'B19001_001E': 'hh_23',
    'B19013_001E': 'hinc_23',
    'B25077_001E': 'mhval_23',
    'B25077_001M': 'mhval_23_se',
    'B25064_001E': 'mrent_23',
    'B25064_001M': 'mrent_23_se',
    'B25003_002E': 'ohu_23',
    'B25003_003E': 'rhu_23',
    'B25105_001E': 'mmhcosts_23',
    'B15003_001E': 'total_25_23',
    'B15003_022E': 'total_25_col_bd_23',
    'B15003_023E': 'total_25_col_md_23',
    'B15003_024E': 'total_25_col_pd_23',
    'B15003_025E': 'total_25_col_phd_23',
    'B25034_001E': 'tot_units_built_23',
    'B25034_010E': 'units_40_49_built_23',
    'B25034_011E': 'units_39_early_built_23',
    # Migration — within same county
    'B07010_025E': 'mov_wc_w_income_23',
    'B07010_026E': 'mov_wc_9000_23',
    'B07010_027E': 'mov_wc_15000_23',
    'B07010_028E': 'mov_wc_25000_23',
    'B07010_029E': 'mov_wc_35000_23',
    'B07010_030E': 'mov_wc_50000_23',
    'B07010_031E': 'mov_wc_65000_23',
    'B07010_032E': 'mov_wc_75000_23',
    'B07010_033E': 'mov_wc_76000_more_23',
    # Migration — from other county same state
    'B07010_036E': 'mov_oc_w_income_23',
    'B07010_037E': 'mov_oc_9000_23',
    'B07010_038E': 'mov_oc_15000_23',
    'B07010_039E': 'mov_oc_25000_23',
    'B07010_040E': 'mov_oc_35000_23',
    'B07010_041E': 'mov_oc_50000_23',
    'B07010_042E': 'mov_oc_65000_23',
    'B07010_043E': 'mov_oc_75000_23',
    'B07010_044E': 'mov_oc_76000_more_23',
    # Migration — from other state
    'B07010_047E': 'mov_os_w_income_23',
    'B07010_048E': 'mov_os_9000_23',
    'B07010_049E': 'mov_os_15000_23',
    'B07010_050E': 'mov_os_25000_23',
    'B07010_051E': 'mov_os_35000_23',
    'B07010_052E': 'mov_os_50000_23',
    'B07010_053E': 'mov_os_65000_23',
    'B07010_054E': 'mov_os_75000_23',
    'B07010_055E': 'mov_os_76000_more_23',
    # Migration — from abroad
    'B07010_058E': 'mov_fa_w_income_23',
    'B07010_059E': 'mov_fa_9000_23',
    'B07010_060E': 'mov_fa_15000_23',
    'B07010_061E': 'mov_fa_25000_23',
    'B07010_062E': 'mov_fa_35000_23',
    'B07010_063E': 'mov_fa_50000_23',
    'B07010_064E': 'mov_fa_65000_23',
    'B07010_065E': 'mov_fa_75000_23',
    'B07010_066E': 'mov_fa_76000_more_23',
    # Other
    'B06011_001E': 'iinc_23',
    # Income brackets
    'B19001_002E': 'I_10000_23',
    'B19001_003E': 'I_15000_23',
    'B19001_004E': 'I_20000_23',
    'B19001_005E': 'I_25000_23',
    'B19001_006E': 'I_30000_23',
    'B19001_007E': 'I_35000_23',
    'B19001_008E': 'I_40000_23',
    'B19001_009E': 'I_45000_23',
    'B19001_010E': 'I_50000_23',
    'B19001_011E': 'I_60000_23',
    'B19001_012E': 'I_75000_23',
    'B19001_013E': 'I_100000_23',
    'B19001_014E': 'I_125000_23',
    'B19001_015E': 'I_150000_23',
    'B19001_016E': 'I_200000_23',
    'B19001_017E': 'I_201000_23',
})

# ==========================================================================
# Download ACS 2019 5-Year Estimates  (replaces 2012 lag pull)
# ==========================================================================
# Note: ACS 2019 5-year (2015–2019) uses pre-2020 tract boundaries, which
# match 2023 ACS 5-year boundaries. No crosswalk is needed for ACS-to-ACS
# comparisons within this vintage. If you later add 2010 decennial data,
# you WILL need an NHGIS crosswalk to reconcile 2010 → 2020 tract boundaries.

df_vars_19 = [
    'B25077_001E',
    'B25077_001M',
    'B25064_001E',
    'B25064_001M',
    'B07010_025E', 'B07010_026E', 'B07010_027E', 'B07010_028E',
    'B07010_029E', 'B07010_030E', 'B07010_031E', 'B07010_032E',
    'B07010_033E',
    'B07010_036E', 'B07010_037E', 'B07010_038E', 'B07010_039E',
    'B07010_040E', 'B07010_041E', 'B07010_042E', 'B07010_043E',
    'B07010_044E',
    'B07010_047E', 'B07010_048E', 'B07010_049E', 'B07010_050E',
    'B07010_051E', 'B07010_052E', 'B07010_053E', 'B07010_054E',
    'B07010_055E',
    'B07010_058E', 'B07010_059E', 'B07010_060E', 'B07010_061E',
    'B07010_062E', 'B07010_063E', 'B07010_064E', 'B07010_065E',
    'B07010_066E',
    'B06011_001E',
]

# Run API query — ACS 2019 5-year
var_dict_acs5 = c.acs5.get(
    df_vars_19,
    geo={'for': 'tract:*', 'in': sql_query},
    year=2019
)

# Convert to DataFrame, build FIPS, filter counties
df_vars_19 = pd.DataFrame.from_dict(var_dict_acs5)
df_vars_19['FIPS'] = df_vars_19['state'] + df_vars_19['county'] + df_vars_19['tract']
df_vars_19 = filter_FIPS(df_vars_19)

# Rename — suffix _19 throughout
df_vars_19 = df_vars_19.rename(columns={
    'B25077_001E': 'mhval_19',
    'B25077_001M': 'mhval_19_se',
    'B25064_001E': 'mrent_19',
    'B25064_001M': 'mrent_19_se',
    # Migration — within same county
    'B07010_025E': 'mov_wc_w_income_19',
    'B07010_026E': 'mov_wc_9000_19',
    'B07010_027E': 'mov_wc_15000_19',
    'B07010_028E': 'mov_wc_25000_19',
    'B07010_029E': 'mov_wc_35000_19',
    'B07010_030E': 'mov_wc_50000_19',
    'B07010_031E': 'mov_wc_65000_19',
    'B07010_032E': 'mov_wc_75000_19',
    'B07010_033E': 'mov_wc_76000_more_19',
    # Migration — from other county same state
    'B07010_036E': 'mov_oc_w_income_19',
    'B07010_037E': 'mov_oc_9000_19',
    'B07010_038E': 'mov_oc_15000_19',
    'B07010_039E': 'mov_oc_25000_19',
    'B07010_040E': 'mov_oc_35000_19',
    'B07010_041E': 'mov_oc_50000_19',
    'B07010_042E': 'mov_oc_65000_19',
    'B07010_043E': 'mov_oc_75000_19',
    'B07010_044E': 'mov_oc_76000_more_19',
    # Migration — from other state
    'B07010_047E': 'mov_os_w_income_19',
    'B07010_048E': 'mov_os_9000_19',
    'B07010_049E': 'mov_os_15000_19',
    'B07010_050E': 'mov_os_25000_19',
    'B07010_051E': 'mov_os_35000_19',
    'B07010_052E': 'mov_os_50000_19',
    'B07010_053E': 'mov_os_65000_19',
    'B07010_054E': 'mov_os_75000_19',
    'B07010_055E': 'mov_os_76000_more_19',
    # Migration — from abroad
    'B07010_058E': 'mov_fa_w_income_19',
    'B07010_059E': 'mov_fa_9000_19',
    'B07010_060E': 'mov_fa_15000_19',
    'B07010_061E': 'mov_fa_25000_19',
    'B07010_062E': 'mov_fa_35000_19',
    'B07010_063E': 'mov_fa_50000_19',
    'B07010_064E': 'mov_fa_65000_19',
    'B07010_065E': 'mov_fa_75000_19',
    'B07010_066E': 'mov_fa_76000_more_19',
    'B06011_001E': 'iinc_19',
})

# ==========================================================================
# Export Files
# ==========================================================================

Path(output_path + 'downloads/').mkdir(parents=True, exist_ok=True)

# Merge 2019 & 2023 files — both use 2020 tract boundaries, so merge is clean
df_vars_summ = df_vars_23.merge(df_vars_19, on='FIPS')

df_vars_summ.to_csv(
    output_path + 'downloads/' + city_name.replace(' ', '') + '_census_summ_2023.csv',
    index=False
)

print(f"Done. {len(df_vars_summ)} tracts written to:")
print(f"  {output_path}downloads/{city_name.replace(' ', '')}_census_summ_2023.csv")
print(f"\nCounty tract counts:")
print(df_vars_summ['county'].value_counts().rename({
    '001': 'Alameda',
    '075': 'San Francisco',
    '081': 'San Mateo',
    '085': 'Santa Clara'
}))
