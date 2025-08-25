from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import os

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb

from feature_engineering import data_generate
from model_utils import aggregate_binary_prediction, aggregate_multi_class_prediction

warnings.filterwarnings('ignore')
os.environ['XGB_VERBOSE'] = '0'  # 屏蔽 XGBoost 日誌

def main():
    # 讀取訓練資料
    info = pd.read_csv('train_info.csv')
    target_mask = ['gender', 'hold racket handed', 'play years', 'level']
    x_all = pd.DataFrame()
    y_all = pd.DataFrame(columns=target_mask)
    player_ids = []

    datapath = './tabular_data_train'
    datalist = list(Path(datapath).glob('**/*.csv'))
    for file in datalist:
        unique_id = int(Path(file).stem)
        row = info[info['unique_id'] == unique_id]
        if row.empty:
            continue
        data = pd.read_csv(file)
        target = row[target_mask]
        target_repeated = pd.concat([target] * len(data))
        x_all = pd.concat([x_all, data], ignore_index=True)
        y_all = pd.concat([y_all, target_repeated], ignore_index=True)
        player_ids.extend([row.iloc[0]['player_id']] * len(data))

    # 特徵標準化 & label 編碼
    scaler = MinMaxScaler()
    X_all = scaler.fit_transform(x_all)
    label_encoders = {}
    y_all_dict = {}
    for col in target_mask:
        le = LabelEncoder()
        y_all_dict[col] = le.fit_transform(y_all[col])
        label_encoders[col] = le

    player_ids = np.array(player_ids)

    # player_id 分割
    all_pids = np.unique(player_ids)
    train_pids, holdout_pids = train_test_split(all_pids, test_size=0.1, random_state=42, shuffle=True)
    print(f"訓練集 player_id: {len(train_pids)} / 測試集 player_id: {len(holdout_pids)}")

    # 根據 player_id 切分資料
    is_holdout = np.isin(player_ids, holdout_pids)
    X_rest, X_holdout = X_all[~is_holdout], X_all[is_holdout]
    y_rest_dict = {col: y[~is_holdout] for col, y in y_all_dict.items()}
    y_holdout_dict = {col: y[is_holdout] for col, y in y_all_dict.items()}

    tasks = ['gender', 'hold racket handed', 'play years', 'level']
    final_models = {}

    for task in tasks:
        print(f"\n{'='*20} 任務: {task} {'='*20}")
        y_rest = y_rest_dict[task]
        y_holdout = y_holdout_dict[task]

        if len(np.unique(y_rest)) == 2:
            print("使用 XGBoost 做 2元分類")
            clf = xgb.XGBClassifier(
                n_estimators=1500,
                learning_rate=0.01,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.5,
                reg_lambda=3,
                eval_metric='auc',  # 放初始化
                use_label_encoder=False,
                verbosity=0,
                random_state=42
            )
        else:
            print("使用 LightGBM 做多分類")
            clf = lgb.LGBMClassifier(
                n_estimators=1500,
                learning_rate=0.01,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=3,
                objective='multiclass',
                num_class=len(np.unique(y_rest)),
                random_state=42
            )

        # 10折交叉驗證
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        aucs, accs = [], []
        for fold, (train_idx, valid_idx) in enumerate(skf.split(X_rest, y_rest)):
            X_train_fold, X_valid_fold = X_rest[train_idx], X_rest[valid_idx]
            y_train_fold, y_valid_fold = y_rest[train_idx], y_rest[valid_idx]

            if len(np.unique(y_rest)) == 2:
                clf.fit(X_train_fold, y_train_fold,
                        eval_set=[(X_valid_fold, y_valid_fold)])
            else:
                clf.fit(X_train_fold, y_train_fold,
                        eval_set=[(X_valid_fold, y_valid_fold)],
                        eval_metric='multi_logloss')

            y_pred_prob = clf.predict_proba(X_valid_fold)
            if len(np.unique(y_rest)) == 2:
                auc = roc_auc_score(y_valid_fold, y_pred_prob[:, 1])
            else:
                auc = roc_auc_score(y_valid_fold, y_pred_prob, multi_class='ovr')
            acc = accuracy_score(y_valid_fold, clf.predict(X_valid_fold))
            aucs.append(auc)
            accs.append(acc)
            print(f"  Fold {fold+1}: AUC={auc:.4f}, ACC={acc:.4f}")

        print(f"=> 10折平均 AUC: {np.mean(aucs):.4f}, ACC: {np.mean(accs):.4f}")

        clf.fit(X_rest, y_rest)
        final_models[task] = clf

        y_pred_holdout_prob = clf.predict_proba(X_holdout)
        y_pred_holdout = clf.predict(X_holdout)
        try:
            auc_holdout = roc_auc_score(
                y_holdout,
                y_pred_holdout_prob[:, 1] if len(np.unique(y_rest)) == 2 else y_pred_holdout_prob,
                multi_class='ovr' if len(np.unique(y_rest)) > 2 else 'raise'
            )
            acc_holdout = accuracy_score(y_holdout, y_pred_holdout)
            print(f"✅ Hold-out 測試集 AUC: {auc_holdout:.4f}, ACC: {acc_holdout:.4f}")
        except Exception as e:
            print(f"❌ Hold-out 測試集 AUC 計算失敗: {str(e)}")
            acc_holdout = accuracy_score(y_holdout, y_pred_holdout)
            print(f"✅ Hold-out 測試集 ACC: {acc_holdout:.4f}")

    print("\n🎯 四個任務的模型都已完成！")

    # 產生 submission.csv
    print("\n========== 產生 submission.csv ==========")
    test_info = pd.read_csv('test_info.csv')
    test_ids = test_info['unique_id'].values
    submission_rows = []

    for unique_id in test_ids:
        file = f'./tabular_data_test/{unique_id}.csv'
        if not Path(file).exists():
            row = {
                "unique_id": unique_id,
                "gender": 0,
                "hold racket handed": 0,
                "play years_0": 0, "play years_1": 0, "play years_2": 0,
                "level_2": 0, "level_3": 0, "level_4": 0, "level_5": 0,
            }
            submission_rows.append(row)
            continue
        df = pd.read_csv(file)
        if df.shape[0] == 0:
            row = {
                "unique_id": unique_id,
                "gender": 0,
                "hold racket handed": 0,
                "play years_0": 0, "play years_1": 0, "play years_2": 0,
                "level_2": 0, "level_3": 0, "level_4": 0, "level_5": 0,
            }
            submission_rows.append(row)
            continue
        features_27 = df.values
        features_27 = scaler.transform(features_27)

        pred_gender = aggregate_binary_prediction(final_models['gender'], features_27)
        pred_hold = aggregate_binary_prediction(final_models['hold racket handed'], features_27)
        pred_years = aggregate_multi_class_prediction(final_models['play years'], features_27)
        pred_level = aggregate_multi_class_prediction(final_models['level'], features_27)

        row = {
            "unique_id": unique_id,
            "gender": pred_gender[0],
            "hold racket handed": pred_hold[0],
            "play years_0": pred_years[0],
            "play years_1": pred_years[1],
            "play years_2": pred_years[2],
            "level_2": pred_level[0],
            "level_3": pred_level[1],
            "level_4": pred_level[2],
            "level_5": pred_level[3],
        }
        submission_rows.append(row)

    submission = pd.DataFrame(submission_rows)
    submission = submission.round(4)
    submission.to_csv('submission_final_mixed.csv', index=False)
    print("✅ submission_final_mixed.csv 已產生，請上傳比賽網站進行評分！")

if __name__ == '__main__':
    main()
