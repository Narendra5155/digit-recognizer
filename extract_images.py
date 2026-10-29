import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm


os.makedirs('./dataset',exist_ok=True)


def save_to_disk(df:pd.DataFrame,title):
  base_dir = Path("./dataset") / title
  os.makedirs(base_dir,exist_ok=True)
  
  print(f"Saving {title} dataframe to {base_dir}------")
  for index,row in tqdm(df.iterrows()):
    file_name = f"{index:05d}_{row['label']}.png"
    file_path = base_dir / file_name
    with open(file_path,'wb') as f:
      f.write(bytes(row['image']['bytes']))

test_df = pd.read_parquet('test-00000-of-00001.parquet')
train_df = pd.read_parquet('train-00000-of-00001.parquet')


save_to_disk(test_df,'test')
save_to_disk(train_df,'train')

