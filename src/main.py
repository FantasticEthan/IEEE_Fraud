# -*- coding: utf-8 -*-
# author：ethan
from tools.utils import *
from DataLoader import DataLoader
from GainFeature import Features
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import lightgbm as lgb

# test code
Data = DataLoader(
                 trainfile_fraud_transaction_info='train_transaction.csv',
                 trainfile_fraud_identity_info='train_identity.csv',
                 testfile_fraud_transaction_info='test_transaction.csv',
                 testfile_fraud_identity_info='test_identity.csv',
                 submisson_info='sample_submission.csv')

# train data
TrainFeatures = Features(
						DataLoader=Data,
						MakeLabel = True
					)