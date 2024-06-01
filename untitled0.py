# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:54:13 2021

@author: wudongsheng
"""

import torch
import torch.nn as nn

model = nn.Sequential(nn.Conv2d(3,8,3),
        nn.ReLU(),
                      nn.Conv2d(8,8,3),
        nn.ReLU(),
        nn.Conv2d(8,8,3),
        nn.ReLU())
image = torch.randn(10,3,45,45)
print(model(image))