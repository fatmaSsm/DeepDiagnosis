"""Leakage-safe baseline: all learned preprocessing fits inside CV folds."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.metrics import recall_score, precision_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix

GROUPS = ('MASTER','KANSER','CFTR','PAH')
SEED = 42

def model_pipeline(frame):
    categories = [c for c in frame if c.startswith(('CAT_','AA_')) or frame[c].dtype == 'object']
    numbers = [c for c in frame if c not in categories]
    preprocess = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                          ('scale', StandardScaler())]), numbers),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='constant',fill_value='MISSING')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categories),
    ], remainder='drop', sparse_threshold=0)
    # Compact, transparent baseline. The original competition ensemble is retained in legacy/.
    model = LogisticRegression(max_iter=2000,class_weight='balanced',random_state=SEED)
    return Pipeline([('preprocess',preprocess),('model',model)])

def optimize_threshold(y, prob, minimum_recall=0.9):
    candidates=np.unique(np.r_[0.0, np.linspace(0.01, 0.99, 99), 1.0])
    choices=[]
    for t in candidates:
        prediction=(prob>=t).astype(int)
        r=recall_score(y,prediction,zero_division=0)
        if r>=minimum_recall:
            choices.append((f1_score(y,prediction,zero_division=0),float(t)))
    return max(choices)[1] if choices else 0.5

def train_group(group, data_dir, output_dir):
    file=data_dir/f'YARISMA_TRAIN_{group}.csv'
    if not file.exists():
        raise FileNotFoundError(f'Missing {file}; see data/README.md')
    df=pd.read_csv(file)
    if 'Label' not in df: raise ValueError(f'{file}: Label missing')
    if df['Label'].isna().any() or not set(df['Label'].unique()).issubset({0,1}):
        raise ValueError('Label must contain only 0 and 1')
    X=df.drop(columns=['Label','Variant_ID'],errors='ignore')
    y=df['Label'].astype(int)
    if len(X)<20 or y.value_counts().min()<10: raise ValueError('Requires at least 20 rows and 10 per class')
    train_x, test_x, train_y, test_y=train_test_split(X,y, test_size=0.2,random_state=SEED,stratify=y)
    folds=min(5,int(train_y.value_counts().min()))
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=SEED)
    pipe=model_pipeline(train_x)
    # Threshold selected exclusively using out-of-fold training probabilities.
    oof=cross_val_predict(pipe,train_x,train_y,cv=cv,method='predict_proba')[:,1]
    threshold=optimize_threshold(train_y.to_numpy(),oof)
    pipe.fit(train_x,train_y)
    prob=pipe.predict_proba(test_x)[:,1]
    pred=(prob>=threshold).astype(int)
    metrics={'group':group,'samples':int(len(df)),'test_samples':int(len(test_y)),
             'threshold':threshold,'recall':float(recall_score(test_y,pred,zero_division=0)),
             'precision':float(precision_score(test_y,pred,zero_division=0)),
             'f1':float(f1_score(test_y,pred,zero_division=0)),
             'balanced_accuracy':float(balanced_accuracy_score(test_y,pred)),
             'roc_auc':float(roc_auc_score(test_y,prob)),
             'confusion_matrix':confusion_matrix(test_y,pred).tolist()}
    (output_dir/'models').mkdir(parents=True,exist_ok=True)
    joblib.dump({'pipeline':pipe,'threshold':threshold,'features':list(X.columns),'group':group},output_dir/'models'/f'{group.lower()}.joblib')
    return metrics

def predict_group(group, source, output_dir, target):
    if group not in GROUPS: raise ValueError('Unknown group')
    # Only load model files created by you, never untrusted joblib/pickle files.
    bundle=joblib.load(output_dir/'models'/f'{group.lower()}.joblib')
    df=pd.read_csv(source)
    features=bundle['features']
    absent=[c for c in features if c not in df]
    if absent: raise ValueError(f'Missing feature columns: {absent[:10]}')
    prob=bundle['pipeline'].predict_proba(df[features])[:,1]
    result=pd.DataFrame({'Variant_ID':df['Variant_ID'] if 'Variant_ID' in df else range(len(df)),
                         'Label_Pred':(prob>=bundle['threshold']).astype(int),
                         'Prob_Patojenik':prob})
    target.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(target,index=False)
    return target

def main():
    parser=argparse.ArgumentParser(description='DeepDiagnosis leakage-safe baseline')
    sub=parser.add_subparsers(dest='command',required=True)
    tr=sub.add_parser('train'); tr.add_argument('--data-dir',type=Path,default=Path('data')); tr.add_argument('--output-dir',type=Path,default=Path('outputs')); tr.add_argument('--group',choices=[*GROUPS,'ALL'],default='ALL')
    pr=sub.add_parser('predict'); pr.add_argument('--group',choices=GROUPS,required=True); pr.add_argument('--input',type=Path,required=True); pr.add_argument('--output',type=Path,default=Path('outputs/predictions.csv')); pr.add_argument('--model-dir',type=Path,default=Path('outputs'))
    args=parser.parse_args()
    if args.command=='train':
        groups=GROUPS if args.group=='ALL' else [args.group]
        output={g:train_group(g,args.data_dir,args.output_dir) for g in groups}
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/'baseline_metrics.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
        print(json.dumps(output,indent=2))
    else: print(predict_group(args.group,args.input,args.model_dir,args.output))

if __name__=='__main__':main()
