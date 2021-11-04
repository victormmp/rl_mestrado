import datetime
import os
from random import shuffle

import pandas as pd

class Stock():

    def __init__(self, p_symbol: str, p_dataFrame: pd.DataFrame, p_source: str, p_exchange: str, p_name: str = ''):
        
        self.symbol = p_symbol        # asset symbol = ticker or eikon RIC, etc.
        self.source = p_source        # vendor = 'eikon' or 'yahoo' or 'investing', etc.
        self.exchange = p_exchange     # exchange = folder where .csv files are kept
        self.name = p_name              # name = asset name
        self.dataFrame = p_dataFrame
        
        if self.name == '':
            self.name = self.symbol

        
    @property
    def id(self):
        return self.symbol # todo: update to SYMBOL.EXCHANGE

    @property
    def getInfo(self):
        return self.source + ' - ' + self.exchange + ' - ' + self.symbol
    
    @property
    def getDf(self):
        return self.dataFrame
    
    @staticmethod
    def _initFromEikon(p_filePath):
        v_symbol = p_filePath.split('/')[-1].split('.csv')[0]
        v_source = 'eikon'
        v_exchange = p_filePath.split('data/eikon/')[1].split('/')[1]
        
        v_dataFrame = pd.read_csv(p_filePath)

        v_dataFrame.index = pd.to_datetime(v_dataFrame['Date'].values)
        v_dataFrame.sort_index(inplace=True)

        v_dataFrame.rename(columns={'HIGH': 'high', 'CLOSE': 'close', 'LOW': 'low', 'OPEN': 'open', 'COUNT': 'volume_ticks', 'VOLUME': 'volume_shares'}, inplace=True)
        # todo: check if VOLUME matches shares or dollar
        v_dataFrame.drop(['Date'], axis=1, inplace=True)

        v_dataFrame['volume_dollar'] = None
        v_dataFrame['avg'] = (v_dataFrame['open'] + v_dataFrame['close']) / 2.0


        return Stock(p_symbol=v_symbol, p_source=v_source, p_dataFrame=v_dataFrame,
                     p_exchange=v_exchange)


    @staticmethod
    def getStocksFromFiles(p_database='eikon', p_number=None, p_exchange='US_ETFs', p_shuffle=True, data_path: str = 'data'):

        v_stockList = []
        if (p_database.lower() == 'eikon'):
            v_folderPath = os.path.join(data_path, 'eikon', p_exchange)
            v_fileList = os.listdir(v_folderPath)
        else:
            raise Exception('Wrong parameter value for database')
 
        v_fileList = [f for f in v_fileList if ('.~lock' not in f) and ('.DS_Store' not in f) and ('.directory' not in f)]
        
        if (p_shuffle):
            shuffle(v_fileList)

        if (p_number is not None):
            v_fileList = v_fileList[:p_number]

        for i_file in v_fileList:
            p_filePath = os.path.join(v_folderPath, i_file)
            v_stockList.append(Stock._initFromEikon(p_filePath))
            

        return v_stockList
    
    
    @staticmethod
    def getAssetsAggregatedDf(p_assetsList: list, p_column: str = 'close') -> pd.DataFrame:
        v_assetsValues = []
        for i_asset in p_assetsList:
            i_assetValues = i_asset.dataFrame[p_column]
            i_assetValues.name = i_asset.id
            v_assetsValues.append(i_assetValues)
        v_assetsValues = pd.concat(v_assetsValues, axis=1)

        return v_assetsValues


    @staticmethod
    def getAssetsReturnsDf(p_assetsList: list, p_fillNa: bool = True) -> pd.DataFrame:
        v_closePrices = Stock.getAssetsAggregatedDf(p_assetsList=p_assetsList, p_column='close')
        v_assetsReturns = v_closePrices.pct_change(axis=0)
        if p_fillNa:
            v_assetsReturns.fillna(value=0, inplace=True)

        return v_assetsReturns
         

    