#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataset/store.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 12th 2025 03:17:59 am                                                #
# Modified   : Sunday October 12th 2025 06:50:29 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from pandas import DataFrame
from valuation.dataset.base import Dataset

# ------------------------------------------------------------------------------------------------ #
GROUPBY='store'
class StoreDataset(Dataset):
    def __init__(self, sales: DataFrame) -> None:
        super().__init__(sales, groupby=GROUPBY)
        self._store_analysis = None
        
    def store_analysis(self) -> DataFrame:
        if self._store_analysis is None:
            self._compute_store_analysis()
        return self._store_analysis if self._store_analysis is not None else DataFrame()    
    
    def _compute_store_analysis(self) -> None:
        full_df = self.get_full_years_data()
        if full_df.empty:
            self._store_analysis = DataFrame()
            return  
        
        df = full_df.groupby(self._groupby).agg(
            year=('year')
            total_sales=('revenue', 'sum'),
            total_transactions=('transactions', 'sum'),
            total_units_sold=('movement', 'sum'),
            average_price=('price', 'mean'),
            average_basket_size=('basket_size', 'mean'),
            number_of_years=('year', 'nunique')
        ).reset_index()
        
        df['average_annual_sales'] = df['total_sales'] / df['number_of_years']
        df['average_annual_transactions'] = df['total_transactions'] / df['number_of_years']
        df['average_annual_units_sold'] = df['total_units_sold'] / df['number_of_years']
        
        self._store_analysis = df
        
    
        