import unittest
import numpy as np
from src.pipeline import optimize_threshold, model_pipeline
import pandas as pd

class BaselineTests(unittest.TestCase):
    def test_threshold(self):
        t=optimize_threshold(np.array([1,1,0,0]),np.array([.9,.8,.1,.2]))
        self.assertTrue(.0<=t<=1.0)
    def test_fit_predict_with_unseen_category(self):
        x=pd.DataFrame({'AL_score':[0.,1.,2.,3.,4.,5.], 'CAT_base':['A','B','A','B','A','B']})
        y=[0,0,0,1,1,1]
        m=model_pipeline(x).fit(x,y)
        p=m.predict_proba(pd.DataFrame({'AL_score':[1.2],'CAT_base':['UNSEEN']}))
        self.assertEqual(p.shape,(1,2))

if __name__=='__main__':unittest.main()
