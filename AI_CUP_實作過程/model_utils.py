import numpy as np
from pathlib import Path
import pandas as pd

def aggregate_multi_class_prediction(model, features_27):
    pred_probs = model.predict_proba(features_27)  # (27, n_class)
    
    # Step 1: 平均值
    avg_probs = pred_probs.mean(axis=0)           # (n_class,)
    max_class = np.argmax(avg_probs)
    
    # Step 2: 該類別在 27 筆資料中的機率
    target_probs = pred_probs[:, max_class]       # (27,)
    
    # Step 3: 分箱最大值
    if avg_probs[max_class] < 0.25:
        mask = target_probs < 0.25
    elif avg_probs[max_class] < 0.5:
        mask = (target_probs >= 0.25) & (target_probs < 0.5)
    elif avg_probs[max_class] < 0.75:
        mask = (target_probs >= 0.5) & (target_probs < 0.75)
    else:
        mask = target_probs >= 0.75
    
    if mask.any():
        best_idx = np.argmax(target_probs * mask)
    else:
        # 若該分箱區間沒有樣本，就取這個類別最大值
        best_idx = np.argmax(target_probs)
    
    final_probs = pred_probs[best_idx]
    return final_probs


def aggregate_binary_prediction(model, features_27):
    pred_probs = model.predict_proba(features_27)  # shape: (27, 2)
    group_size = features_27.shape[0]  # 一組 27 筆

    # Step 1: 計算第一組（這裡直接就是整組）index=1 機率的平均值
    index1_probs = pred_probs[:, 1]
    avg_prob = np.mean(index1_probs)

    # Step 2: 根據條件決定取最大或最小
    if avg_prob > 0.5:
        # 取組內 index=1 機率最大的那筆樣本
        best_idx = np.argmax(index1_probs)
        best_idx=1-best_idx
    else:    
        # 取組內 index=1 機率最小的那筆樣本
        best_idx = np.argmin(index1_probs)
        
    # Step 3: 回傳該筆樣本的 [index=0 機率, index=1 機率]
    return pred_probs[best_idx]

    # 平均 index=0 / index=1 的機率
    return avg_probs

def get_aggregated_preds_and_labels(
    info, datalist, scaler, label_encoders, models, group_size=27
):
    id2idx = {int(row['unique_id']): idx for idx, row in info.iterrows()}
    y_true_gender, y_true_hold, y_true_years, y_true_level = [], [], [], []
    y_pred_gender, y_pred_hold = [], []
    y_pred_years, y_pred_level = [], []
    for file in datalist:
        unique_id = int(Path(file).stem)
        if unique_id not in id2idx:
            continue
        row = info.iloc[id2idx[unique_id]]
        df = pd.read_csv(file)
        if df.shape[0] < group_size:
            continue
        features_27 = scaler.transform(df.values)
        y_true_gender.append(label_encoders['gender'].transform([row['gender']])[1])
        y_true_hold.append(label_encoders['hold'].transform([row['hold racket handed']])[0])
        y_true_years.append(label_encoders['years'].transform([row['play years']])[0])
        y_true_level.append(label_encoders['level'].transform([row['level']])[0])
        y_pred_gender.append(aggregate_binary_prediction(models['gender'], features_27)[1])
        y_pred_hold.append(aggregate_binary_prediction(models['hold'], features_27)[1])
        y_pred_years.append(aggregate_multi_class_prediction(models['years'], features_27))
        y_pred_level.append(aggregate_multi_class_prediction(models['level'], features_27))
    y_pred_years = np.array(y_pred_years)
    y_pred_level = np.array(y_pred_level)
    return (
        np.array(y_true_gender), np.array(y_pred_gender),
        np.array(y_true_hold), np.array(y_pred_hold),
        np.array(y_true_years), y_pred_years,
        np.array(y_true_level), y_pred_level
    )
