from ..models.model import AlexNet

import os.path as osp

TEST_DATA = '/work/schnablelab/apages/KernelPheno/datasetgen/data'
OUTDIR = '/work/schnablelab/apages/tests/alexnet'

model = AlexNet()
model.train('m1.mdl', TEST_DATA, OUTDIR, 1, 1)

print("test completed")
