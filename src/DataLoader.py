# anthor : ethan 
import pandas as pd
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

class DataLoader(object):
    def __init__(self,
                 trainfile_fraud_transaction_info,
                 trainfile_fraud_identity_info,
                 testfile_fraud_transaction_info,
                 testfile_fraud_identity_info,
                 submisson_info,
                 FOLDER_PATH = '../data/'
                 ):
        self.trainfile_fraud_transaction_info = trainfile_fraud_transaction_info
        self.trainfile_fraud_identity_info = trainfile_fraud_identity_info
        self.testfile_fraud_transaction_info = testfile_fraud_transaction_info
        self.testfile_fraud_identity_info = testfile_fraud_identity_info
        self.submisson_info = submisson_info

        self.df_train_transaction = pd.read_csv(FOLDER_PATH+self.trainfile_fraud_transaction_info)
        self.df_train_identity = pd.read_csv(FOLDER_PATH+self.trainfile_fraud_identity_info)
        self.df_test_transaction = pd.read_csv(FOLDER_PATH+self.testfile_fraud_transaction_info)
        self.df_test_identity = pd.read_csv(FOLDER_PATH+self.testfile_fraud_identity_info)

        self.sub = pd.read_csv(FOLDER_PATH+self.submisson_info)

        self.train = pd.merge(self.df_train_transaction, self.df_train_identity, on='TransactionID', how='left')
        self.test = pd.merge(self.df_test_transaction, self.df_test_identity, on='TransactionID', how='left')

        del self.df_train_transaction,self.df_train_identity,self.df_test_transaction,self.df_test_identity

        logger.info(f'Train dataset has {self.train.shape[0]} rows and {self.train.shape[1]} columns.')
        logger.info(f'Test dataset has {self.test.shape[0]} rows and {self.test.shape[1]} columns.')

        self.one_value_cols = [col for col in self.train.columns if self.train[col].nunique() <= 1]
        self.one_value_cols_test = [col for col in self.test.columns if self.test[col].nunique() <= 1]
        logger.info(f'train one value colunms equal to test: {self.one_value_cols == self.one_value_cols_test}')


