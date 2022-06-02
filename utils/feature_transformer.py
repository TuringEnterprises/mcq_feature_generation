"""Classes for performing feature transformations."""
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import pandas as pd


class FeatureTransformer(ABC):
    """Container class for feature transforming methods."""

    def sigmoid_transform(
        self, x: pd.Series, minus: float, div: float = 1, sign: int = 1
    ) -> pd.Series:
        """Do a kind of sigmoid transformation."""
        return 1 / (1 + np.exp(sign * ((minus - x) / div)))

    def get_avg_max_acceptable_rate(
        self, data_frame: pd.DataFrame = None, from_data: bool = False
    ) -> float:
        """Get average maximum acceptable rate."""
        if from_data:
            data_frame_job = data_frame.loc[
                data_frame["max_acceptable_rate"] > 0, ["job_id", "max_acceptable_rate"]
            ].drop_duplicates(subset=["job_id"])
            return data_frame_job["max_acceptable_rate"].mean()
        return 35

    def one_hot_encoding(
        self,
        data_frame: pd.DataFrame,
        cat_feas: List[str],
        redundant_cats: List[str] = None,
    ):
        """One hot encode categorical features."""
        data_frame_tmp = pd.get_dummies(data_frame[cat_feas], drop_first=False)
        redundant_cats = np.intersect1d(data_frame_tmp.columns, redundant_cats).tolist()
        data_frame = pd.concat([data_frame, data_frame_tmp], axis=1).drop(
            cat_feas + redundant_cats, axis=1
        )
        return data_frame

    @abstractmethod
    def further_feature_transformation(
        self, data_frame: pd.DataFrame, cols_to_replace_zero: List[str] = None
    ) -> pd.DataFrame:
        """Perform extra feature transformation for example
        on noisy jobs or redundant cathegories."""
        data_frame = data_frame.loc[data_frame["job_id"] != 3280].reset_index(
            drop=True
        )  # this job is known to be noisy
        for col in cols_to_replace_zero:
            data_frame.loc[data_frame[col] == 0, col] = np.nan

        cat_feas = ["client_category", "region"]
        redundant_cats = []
        for col in ["acc_fire_count", "acc_opp_count", "acc_rc_count"]:
            data_frame[col] = np.log1p(data_frame[col])
        data_frame = self.one_hot_encoding(data_frame, cat_feas, redundant_cats)

        return data_frame

    
class FeatureTransformerGBDT(FeatureTransformer):

    def add_sample_weights(self, df):
        weights = 1 / df.groupby('job_id')['is_start'].transform(
            lambda x: (x == 0).sum()
        )
        weights.loc[df['is_start'] == 1] = 1
        return weights

    def further_feature_transformation(
            self, 
            df: pd.DataFrame, 
            cols_to_replace_zero: list[str] = [],
            exclude_recent_jobs: bool = False
    ) -> pd.DataFrame:
        
        if exclude_recent_jobs:
            df['mjm_create_ts'] = df['mjm_create_ts'].fillna(df['created_date'])
            df = df.loc[
                df.groupby('job_id')['mjm_create_ts'].transform(max)<"2021-05-01"
            ].reset_index(drop=True)
            
        for col in cols_to_replace_zero:
            df.loc[df[col] == 0, col] = np.nan

        cat_feas = ['client_category', 'region']
        redundant_cats = []
        for col in ['acc_fire_count', 'acc_opp_count', 'acc_rc_count']:
            df[col] = np.log1p(df[col])
        df = self.one_hot_encoding(df, cat_feas, redundant_cats)

        # job features
        for col in [
            'region_1.Bay Area', 'region_2.Rest of US', 'region_4.Others',
            'client_category_1.Platinum', 'client_category_2.Gold', 'client_category_4.Bronze'
        ]:
            if col not in df.columns:
                df[col] = 0
                
        df['leadsource'] = (df['leadsource'] == 'existing').astype(np.int8)
        df.loc[df['max_acceptable_rate'] == 0, 'max_acceptable_rate'] = np.nan
        df['max_acceptable_rate'] = df['max_acceptable_rate']
        df.loc[df['max_acceptable_rate']>500, 'max_acceptable_rate'] /= 176
        avg_max_acceptable_rate = self.get_avg_max_acceptable_rate()
        df['max_accept_rate_ratio_all'] = df['max_acceptable_rate'] / avg_max_acceptable_rate
        df['max_accept_rate_ratio_all_sq'] = df['max_accept_rate_ratio_all'] ** 2

        # dev features
        df['max_relevant_mcq_pct'] = self.sigmoid_transform(df['max_relevant_mcq_pct'], minus=50, div=10)
        df['mean_mcq_pct'] = self.sigmoid_transform(df['mean_mcq_pct'], minus=50, div=10)
        df['starts_in_weeks'] = self.sigmoid_transform(df['starts_in_weeks'], minus=4)
        df["seniority_avg"] = self.sigmoid_transform(df["seniority_avg"], minus=3.5, div=0.25)
        df["len_resume_raw"] = self.sigmoid_transform(df["len_resume_raw"], minus=150, div=50)
        df['interview_score'] = df['interview_score']
        df['max_skill_yoe'] = np.log1p(df['max_skill_yoe'])
        df['num_relevant_projs'] = np.log1p(df['num_relevant_projs'])
        df['sum_relevant_proj_words'] = self.sigmoid_transform(df['sum_relevant_proj_words'], minus=100, div=25)
        
        df['dev_not_si_from_packet_count'] = self.sigmoid_transform(
            df['dev_not_si_from_packet_count'], minus=6
        )
        
        df.loc[df['hourly_rate']>500, 'hourly_rate'] /= 176
        df['hourly_rate_notna'] = (df['hourly_rate']>0).astype(np.int8)
        df['hourly_rate_ratio'] = df['hourly_rate']/df['max_acceptable_rate']
        df['hourly_rate_ratio_sq'] = df['hourly_rate_ratio']**2
        df.loc[df['hourly_rate']==0, 'hourly_rate'] = np.nan
        
        # sample weight
        df['weight'] = self.add_sample_weights(df)
        
        df.loc[df['hourly_rate']==0, 'hourly_rate'] = np.nan
        return df
    
    

class FeatureTransformerV1(FeatureTransformer):
    """Container class for feature transforming methods specific
    to ML logistic Ranker V1."""

    def add_sample_weights(self, data_frame):
        """Reutrns sample weights"""
        weights = 1 / data_frame.groupby("job_id")["is_start"].transform(
            lambda x: (x == 0).sum()
        )
        weights.loc[data_frame["is_start"] == 1] = 1
        return weights

    def further_feature_transformation(
        self, data_frame: pd.DataFrame, cols_to_replace_zero: List[str] = None
    ) -> pd.DataFrame:
        """Perform extra feature transformation for example
        on noisy jobs or redundant cathegories."""
        data_frame = data_frame.loc[data_frame["job_id"] != 3280].reset_index(
            drop=True
        )  # this job is known to be noisy
        for col in cols_to_replace_zero:
            data_frame.loc[data_frame[col] == 0, col] = np.nan

        cat_feas = ["client_category", "region"]
        redundant_cats = []
        for col in ["acc_fire_count", "acc_opp_count", "acc_rc_count"]:
            data_frame[col] = np.log1p(data_frame[col])
        data_frame = self.one_hot_encoding(data_frame, cat_feas, redundant_cats)

        # job features
        if "region_4.Others" not in data_frame.columns:
            data_frame["region_4.Others"] = 0
        data_frame["leadsource"] = (data_frame["leadsource"] == "existing").astype(
            np.int8
        )
        data_frame.loc[
            data_frame["max_acceptable_rate"] == 0, "max_acceptable_rate"
        ] = np.nan
        data_frame["max_acceptable_rate"] = data_frame["max_acceptable_rate"].fillna(35)
        avg_max_acceptable_rate = self.get_avg_max_acceptable_rate()
        data_frame["max_accept_rate_ratio_all"] = (
            data_frame["max_acceptable_rate"] / avg_max_acceptable_rate
        )
        data_frame["max_accept_rate_ratio_all_sq"] = (
            data_frame["max_accept_rate_ratio_all"] ** 2
        )

        # dev features
        data_frame["max_relevant_mcq_pct"] = self.sigmoid_transform(
            data_frame["max_relevant_mcq_pct"].fillna(0), minus=50, div=10
        )
        data_frame["mean_mcq_pct"] = self.sigmoid_transform(
            data_frame["mean_mcq_pct"].fillna(0), minus=50, div=10
        )
        data_frame["starts_in_weeks"] = self.sigmoid_transform(
            data_frame["starts_in_weeks"].fillna(4), minus=4
        )
        data_frame["seniority_avg"] = self.sigmoid_transform(
            data_frame["seniority_avg"].fillna(3.85), minus=3.5, div=0.25
        )
        data_frame["len_resume_raw"] = self.sigmoid_transform(
            data_frame["len_resume_raw"].fillna(500), minus=100, div=50
        )
        data_frame["sum_relevant_proj_words"] = self.sigmoid_transform(
            data_frame["sum_relevant_proj_words"], minus=100, div=25
        )
        data_frame["interview_score"] = data_frame["interview_score"].fillna(7)
        data_frame["max_skill_yoe"] = np.log1p(data_frame["max_skill_yoe"])
        data_frame["num_relevant_projs"] = np.log1p(data_frame["num_relevant_projs"])

        # sample weight
        data_frame["weight"] = self.add_sample_weights(data_frame)
        return data_frame

    
