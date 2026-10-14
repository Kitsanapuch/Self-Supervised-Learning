import unittest
import numpy as np
import torch
from src.experiment import prepare, subset, mask_input, Encoder, Classifier, score, loader
class PipelineTests(unittest.TestCase):
    def test_split_and_scaling(self):
        rng=np.random.default_rng(1)
        raw={'train':(rng.normal(size=(60,6,128)).astype('float32'),np.arange(60)%6,np.repeat(np.arange(10),6)), 'test':(rng.normal(size=(12,6,128)).astype('float32'),np.arange(12)%6,np.repeat([10,11],6))}
        data,meta=prepare(raw)
        self.assertFalse(set(meta['train_subjects']) & set(meta['val_subjects']))
        self.assertTrue(np.allclose(data['train'][0].mean((0,2)),0,atol=1e-5))
    def test_budget_is_nested(self):
        y=np.arange(600)%6
        a,b=subset(y,.01,3),subset(y,.1,3)
        self.assertTrue(set(a)<=set(b)); self.assertEqual(len(np.unique(y[a])),6)
        np.testing.assert_array_equal(a,subset(y,.01,3))
    def test_mask_and_gradient(self):
        x=torch.randn(4,6,128)
        for mode in ['time','channel','both']:
            corrupted,mask=mask_input(x,mode)
            self.assertTrue(mask.any()); self.assertTrue((corrupted[mask]==0).all())
            self.assertTrue(torch.equal(corrupted[~mask],x[~mask]))
        model=Classifier(Encoder()); model(x).sum().backward()
        self.assertIsNotNone(model.encoder.net[0].weight.grad)
    def test_macro_f1_absent_prediction(self):
        class Constant(torch.nn.Module):
            def forward(self,x):
                out=torch.zeros(len(x),6); out[:,0]=1; return out
        result=score(Constant(),loader(np.zeros((6,6,128),dtype='float32'),np.arange(6),6),'cpu')
        self.assertAlmostEqual(result['macro_f1'],(2/7)/6)
if __name__=='__main__': unittest.main()
