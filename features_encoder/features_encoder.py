import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from typing import Union, List

class FeaturesEncoder(BaseEstimator, TransformerMixin):
    """_summary_

    Attributes:
        numeric_features (Union[List,None], optional): _description_. Defaults to None.
        impute_numeric (bool, optional): _description_. Defaults to True.
        numeric_imputer (Union[TransformerMixin, None], optional): _description_. Defaults to None.
        scale_numeric (bool, optional): _description_. Defaults to False.
        numeric_scaler (Union[TransformerMixin, None], optional): _description_. Defaults to None.
        categorical_features (Union[List,None], optional): _description_. Defaults to None.
        max_one_hot (int, optional): _description_. Defaults to 10.
        impute_categorical (bool, optional): _description_. Defaults to True.
        categorical_imputer (Union[TransformerMixin, None], optional): _description_. Defaults to None.
        encode_target (bool, optional): _description_. Defaults to True.

        cardinalitites_: _description_.
        oh_categorical_features_: _description_.
        ord_categorical_features_: _description_.
        pipeline_:
    
    """
    def __init__(
        self, 

        numeric_features:Union[List,None] = None,
        impute_numeric:bool = True, 
        numeric_imputer:Union[TransformerMixin, None] = None,
        scale_numeric:bool = False,
        numeric_scaler:Union[TransformerMixin, None] = None,

        categorical_features:Union[List,None]  = None,
        max_one_hot:int = 10, 
        impute_categorical:bool = True,
        categorical_imputer:Union[TransformerMixin, None] = None,

        encode_target:bool = True,
        ):

        self.numeric_features = numeric_features
        self.impute_numeric = impute_numeric

        if self.impute_numeric:
            self.numeric_imputer = (
                numeric_imputer 
                if numeric_imputer 
                else SimpleImputer(strategy='median')
                )
        else:
            self.numeric_imputer = numeric_imputer

        self.scale_numeric = scale_numeric
        
        if self.scale_numeric:
            self.numeric_scaler = (
                numeric_scaler 
                if numeric_scaler 
                else StandardScaler()
                ) 
        else:
            self.numeric_scaler = numeric_scaler

        self.categorical_features = categorical_features
        self.max_one_hot = max_one_hot
        self.impute_categorical = impute_categorical
        
        if self.impute_categorical:
            self.categorical_imputer = (
                categorical_imputer 
                if categorical_imputer 
                else SimpleImputer(strategy='most_frequent')
                ) 
        else:
            self.categorical_imputer = categorical_imputer

        
        self.encode_target = encode_target


    def fit(self, X, y):

        # validate inputs. Must be dataframes
        if not isinstance(X, pd.DataFrame):
            raise Exception("X must be a pandas DataFrame object")

        if not isinstance(y, (pd.Series, pd.DataFrame)):
            raise Exception("y must be a pandas DataFrame or Series object")

        if not self.numeric_features:
            self.numeric_features = X.select_dtypes(include='number').columns.to_list()

        if not self.categorical_features:
            self.categorical_features = X.select_dtypes(exclude='number').columns.to_list()

        self.cardinalitites_ = [
            X[f].nunique()
            for f in self.categorical_features
        ]


        self.oh_categorical_features_ = [
            self.categorical_features[i] for i,c in enumerate(self.cardinalitites_) 
            if c <= self.max_one_hot
            ]

        self.ord_categorical_features_ = [
            self.categorical_features[i] for i,c in enumerate(self.cardinalitites_) 
            if c > self.max_one_hot
            ]
        
        
        if self.impute_categorical:
            oh_cat_pipeline = make_pipeline(
                self.categorical_imputer, 
                OneHotEncoder(sparse=False, handle_unknown='ignore')
            )
            ord_cat_pipeline = make_pipeline(
                self.categorical_imputer, 
                OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            )
        else:
            oh_cat_pipeline = make_pipeline(
                OneHotEncoder(sparse=False, handle_unknown='ignore')
            ) 
            ord_cat_pipeline = make_pipeline(
                OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            ) 

        if self.impute_numeric and self.scale_numeric:
            num_pipeline =  make_pipeline(
                SimpleImputer(strategy='median'), 
                StandardScaler()
            )
        elif self.impute_numeric and not self.scale_numeric:
             num_pipeline =  make_pipeline(
                SimpleImputer(strategy='median'), 
            )           
        elif not self.impute_numeric and self.scale_numeric:
            num_pipeline =  make_pipeline(
                StandardScaler()
            )
        else:
            num_pipeline="passthrough"
                

        self.pipeline_ = ColumnTransformer([
            ('oh_cat_features',
                oh_cat_pipeline,
                self.oh_categorical_features_
            ),
            ('ord_cat_features',
                ord_cat_pipeline, 
                self.ord_categorical_features_
            ),
            ('num_features',
                num_pipeline, 
                self.numeric_features
            ),
        ])

        if self.encode_target:
            self.label_encoder_ = LabelEncoder()
            y = self.label_encoder_.fit_transform(y)

        self.pipeline_.fit(X, y)

        return self

    def transform(self, X):
        X_out = self.pipeline_.transform(X)

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                X_out,
                columns = self.get_feature_names_out()
            )
        else:
            return X_out

    def get_feature_names_out(self):
        return self.pipeline_.get_feature_names_out()