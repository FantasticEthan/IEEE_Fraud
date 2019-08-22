# anthor : ethan
import pandas as pd
import copy
from sklearn.preprocessing import LabelEncoder
from tools.utils import *

# warning settings
import warnings
warnings.filterwarnings("ignore")

#logging settings
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

#constant value
TEMP_FOLDER='/tmp'

class Features(object):
    def __init__(self,
                 DataLoader,
                 MakeLabel=True
                 ):
        self.DataLoader = DataLoader
        self.MakeLabel = MakeLabel

        # label columns
        self.LabelColumns = ['isFraud']
        self.IDColumns = ['TransactionID']

        # merge feature table

        self.TrainColumns = [col for col in self.DataLoader.train.columns if
                             col not in self.IDColumns + self.LabelColumns]

        self.MakeFeature_Statistics()

        cat_cols = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22',
                    'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
                    'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType',
                    'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4', 'P_emaildomain',
                    'R_emaildomain', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6',
                    'M7', 'M8', 'M9',
                    'P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3', 'R_emaildomain_1', 'R_emaildomain_2',
                    'R_emaildomain_3']

        for col in cat_cols:
            if col in self.DataLoader.train.columns:
                le = LabelEncoder()
                le.fit(list(self.DataLoader.train[col].astype(str).values) + list(self.DataLoader.train[col].astype(str).values))
                self.DataLoader.train[col] = le.transform(list(self.DataLoader.train[col].astype(str).values))
                self.DataLoader.test[col] = le.transform(list(self.DataLoader.test[col].astype(str).values))

    def MakeFeature_Statistics(self,
                                ):
        # Create statistics features
        '''
        '''
        train_temp = copy.deepcopy(self.DataLoader.train)
        test_temp = copy.deepcopy(self.DataLoader.test)

        features_temp_ = train_temp['TransactionAmt'] / train_temp.groupby(['card1'])['TransactionAmt'].transform('mean').\
            rename('TransactionAmt_to_mean_card1')
        self.DataLoader.train = self.DataLoader.train.merge(features_temp_, on=['user_id'], how='left')

        features_temp_ = test_temp['TransactionAmt'] / test_temp.groupby(['card1'])['TransactionAmt'].transform('mean').\
            rename('TransactionAmt_to_mean_card1')
        self.DataLoader.test = self.DataLoader.test.merge(features_temp_, on=['user_id'], how='left')
        '''
        继续添加
        '''

