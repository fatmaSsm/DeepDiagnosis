"""Fully synthetic data for installation demos; not genomic or competition data."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def generate(directory, count=180):
    directory.mkdir(parents=True,exist_ok=True)
    for idx,group in enumerate(['MASTER','KANSER','CFTR','PAH']):
        rng=np.random.default_rng(42+idx)
        a=rng.normal(size=count)
        b=rng.normal(size=count)
        cat=rng.choice(['A','C','G','T'],size=count)
        score=1.2*a-0.7*b+0.4*(cat=='G')+rng.normal(size=count)
        labels=(score>np.median(score)).astype(int)
        df=pd.DataFrame({'Variant_ID':[f'SYN_{group}_{i:04d}' for i in range(count)],'AL_score':a,'EK_score':b,'CAT_base':cat,'Label':labels})
        df.to_csv(directory/f'YARISMA_TRAIN_{group}.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=Path('demo_data'));p.add_argument('--rows',type=int,default=180)
    a=p.parse_args();generate(a.output_dir,a.rows);print(f'Synthetic demo files: {a.output_dir}')
