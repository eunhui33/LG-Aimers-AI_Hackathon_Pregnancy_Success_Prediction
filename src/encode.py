"""
encode.py
----------
Preprocessing utilities for infertility treatment dataset (HFEA, UK).
This script provides encoding functions for categorical features such as
treatment type, age groups, embryo purpose, and procedure counts.

Usage Example:
--------------
import pandas as pd
from src.encode import encode_data

df = pd.read_csv("data/train.csv")
encoded_df = encode_data(df)
"""

import pandas as pd


def encode_treatment_age(age_group):
    """Encode patient's age group at the time of treatment into ordinal values."""
    age_order = {
        '만18-34세': 0,
        '만35-37세': 1,
        '만38-39세': 2,
        '만40-42세': 3,
        '만43-44세': 4,
        '만45-50세': 5,
        '알 수 없음': -1
    }
    return age_order.get(age_group, -1)


def encode_donor_age(age_group):
    """Encode donor's age group into ordinal values."""
    age_order = {
        '만20세 이하': 0,
        '만21-25세': 1,
        '만26-30세': 2,
        '만31-35세': 3,
        '만36-40세': 4,
        '만41-45세': 5,
        '알 수 없음': -1
    }
    return age_order.get(age_group, -1)


def encode_treatment_type(treatment_str):
    """Multi-label binary encoding for treatment types."""
    encoded = {
        'is_IVF': 0,
        'is_ICSI': 0,
        'is_IUI': 0,
        'is_ICI': 0,
        'is_GIFT': 0,
        'is_FER': 0,
        'is_Generic_DI': 0,
        'is_IVI': 0,
        'has_BLASTOCYST': 0,
        'has_AH': 0,
        'is_Unknown': 0
    }

    if pd.isna(treatment_str) or treatment_str == 'Unknown':
        encoded['is_Unknown'] = 1
        return encoded

    all_treatments = []
    for part in treatment_str.split(':'):
        all_treatments.extend(part.split('/'))

    for treatment in all_treatments:
        treatment = treatment.strip()
        if 'IVF' in treatment:
            encoded['is_IVF'] = 1
        if 'ICSI' in treatment:
            encoded['is_ICSI'] = 1
        if 'IUI' in treatment:
            encoded['is_IUI'] = 1
        if 'ICI' in treatment:
            encoded['is_ICI'] = 1
        if 'GIFT' in treatment:
            encoded['is_GIFT'] = 1
        if 'FER' in treatment:
            encoded['is_FER'] = 1
        if 'Generic DI' in treatment:
            encoded['is_Generic_DI'] = 1
        if 'IVI' in treatment:
            encoded['is_IVI'] = 1
        if 'BLASTOCYST' in treatment:
            encoded['has_BLASTOCYST'] = 1
        if 'AH' in treatment:
            encoded['has_AH'] = 1

    return encoded


def encode_embryo_purpose(purpose_str):
    """Multi-label binary encoding for embryo purposes."""
    encoded = {
        'is_donation_purpose': 0,
        'is_egg_storage_purpose': 0,
        'is_embryo_storage_purpose': 0,
        'is_research_purpose': 0,
        'is_current_treatment_purpose': 0
    }

    if pd.isna(purpose_str):
        return encoded

    purposes = purpose_str.split(',')
    for purpose in purposes:
        purpose = purpose.strip()
        if '기증용' in purpose:
            encoded['is_donation_purpose'] = 1
        if '난자 저장용' in purpose:
            encoded['is_egg_storage_purpose'] = 1
        if '배아 저장용' in purpose:
            encoded['is_embryo_storage_purpose'] = 1
        if '연구용' in purpose:
            encoded['is_research_purpose'] = 1
        if '현재 시술용' in purpose:
            encoded['is_current_treatment_purpose'] = 1

    return encoded


def encode_count(count_str):
    """Encode treatment counts into integers."""
    if pd.isna(count_str):
        return -1
    if not isinstance(count_str, str):
        count_str = str(count_str)
    if count_str == '6회 이상':
        return 6
    return int(count_str.replace('회', ''))


def encode_data(df):
    """
    Apply encoding functions to the dataset.
    Returns a new encoded DataFrame.
    """
    encoded_df = pd.DataFrame()

    # Example: Numeric columns can be kept as-is if present
    numeric_columns = [
        '총 생성 배아 수', '미세주입된 난자 수', '이식된 배아 수', '저장된 배아 수', 
        '난자 채취 경과일', '배아 이식 경과일', '임신 성공 여부'
    ]
    for col in numeric_columns:
        if col in df.columns:
            encoded_df[col] = df[col]

    # Encode categorical (One-Hot)
    categorical_columns = ['시술 시기 코드', '난자 출처', '정자 출처']
    for col in categorical_columns:
        if col in df.columns:
            dummies = pd.get_dummies(df[col].fillna('알 수 없음'), prefix=col)
            encoded_df = pd.concat([encoded_df, dummies], axis=1)

    # Encode ordinal features
    if '시술 당시 나이' in df.columns:
        encoded_df['시술 당시 나이'] = df['시술 당시 나이'].apply(encode_treatment_age)
    if '난자 기증자 나이' in df.columns:
        encoded_df['난자 기증자 나이'] = df['난자 기증자 나이'].apply(encode_donor_age)

    # Encode count features
    count_columns = ['총 시술 횟수', '총 임신 횟수', '총 출산 횟수']
    for col in count_columns:
        if col in df.columns:
            encoded_df[col] = df[col].apply(encode_count)

    # Encode multi-label binary
    if '특정 시술 유형' in df.columns:
        treatment_encoded = df['특정 시술 유형'].apply(encode_treatment_type)
        for key in treatment_encoded.iloc[0].keys():
            encoded_df[key] = treatment_encoded.apply(lambda x: x[key])
    if '배아 생성 주요 이유' in df.columns:
        purpose_encoded = df['배아 생성 주요 이유'].apply(encode_embryo_purpose)
        for key in purpose_encoded.iloc[0].keys():
            encoded_df[key] = purpose_encoded.apply(lambda x: x[key])

    # Fill missing values with mean
    encoded_df = encoded_df.fillna(encoded_df.mean())

    return encoded_df
