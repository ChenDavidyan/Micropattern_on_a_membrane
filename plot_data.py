import seaborn as sns
import pandas as pd

df = pd.read_csv('add csv name')

df['dencity'] = df['total'] / 500
df['total_cells_in_hole_diameter'] = df['hole_diameter'] * df['dencity']
df['positive_fraction'] = df['positive'] / df['total_cells_in_hole_diameter']

sns.set_style("whitegrid") 
  
sns.boxplot(x = 'hole_diameter', y = 'total_cells_in_hole_diameter', data = df)

