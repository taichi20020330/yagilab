import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC


# 固有の変数設定
DATA_FILE_PATH = "features/extracted_features.csv"
EXCLUDE_IDS = []

# 使用する変数リストと統計量
variable_list = ['x_acc', 'y_acc', 'z_acc']
# statistics_list = ['sum_values', 'median', 'mean', 'length', 'standard_deviation', 'variance', 'root_mean_square', 'maximum', 'absolute_maximum', 'minimum']
statistics_list = ['max', 'min', 'mean', 'median', 'std', 'var', 'sum']
spectal_feature_list = ['freezing_index', 'central_frequency','dominant_frequency','amplitude','relative_amplitude']
parcentile_list = ['percentile25', 'percentile50', 'percentile75']

# データの前処理
def preprocess_data(file_path, exclude_ids, feature_names):
    df = pd.read_csv(file_path, skiprows=0)
    for exclude_id in exclude_ids:
        df = df[df['ID'] != exclude_id]
    df = df.reset_index(drop=True)
    
    parkinson_target_data = df['group']
    parkinson_df = df.loc[:, feature_names]
    return parkinson_df, parkinson_target_data

# PCAによる次元削減
def perform_pca(X_train, X_test, explained_variance_threshold=0.9999):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # PCA適用
    pca_temp = PCA(n_components=explained_variance_threshold)
    pca_temp.fit(X_train_scaled)
    eigenvalue_list = pca_temp.explained_variance_
    
    # 次元数の決定
    use_component_number = next((count for count, eigenvalue in enumerate(eigenvalue_list) if eigenvalue < 1), len(eigenvalue_list))
    pca = PCA(n_components=use_component_number)
    pca.fit(X_train_scaled)
    
    X_train_scaled_pca = pd.DataFrame(pca.transform(X_train_scaled), columns=range(1, len(pca.explained_variance_) + 1))
    X_test_scaled_pca = pd.DataFrame(pca.transform(X_test_scaled), columns=range(1, len(pca.explained_variance_) + 1))
    
    return X_train_scaled_pca, X_test_scaled_pca, len(pca.explained_variance_)

# SVMのグリッドサーチ
def train_svm(X_train, Y_train):
    search_params = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['rbf']}
    clf = SVC(class_weight='balanced')
    skf = StratifiedKFold(n_splits=5)
    
    gs = GridSearchCV(estimator=clf, param_grid=search_params, cv=skf, n_jobs=-1)
    gs.fit(X_train, Y_train)
    
    return gs.best_estimator_

# モデルの評価
def evaluate_model(Y_test, Y_pred, acc_list, pre_list, rec_list, f1_list):
    acc = metrics.accuracy_score(Y_test, Y_pred)
    pre = metrics.precision_score(Y_test, Y_pred, average='macro')
    rec = metrics.recall_score(Y_test, Y_pred, average='macro')
    f1 = metrics.f1_score(Y_test, Y_pred, average='macro')
    
    acc_list.append(acc)
    pre_list.append(pre)
    rec_list.append(rec)
    f1_list.append(f1)

# メイン処理
def main():
    
    # feature_names = [f"{var}__{stat}" for stat in statistics_list for var in variable_list]
    feature_names = [f"{var}__{stat}" for stat in statistics_list  + parcentile_list + spectal_feature_list for var in variable_list] 

    # データの前処理
    parkinson_df, parkinson_target_data = preprocess_data(DATA_FILE_PATH, EXCLUDE_IDS, feature_names)

    # 評価結果のリスト
    acc_list, pre_list, rec_list, f1_list = [], [], [], []
    num_components_list = []

    # 交差検証とモデル学習
    for k in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(parkinson_df, parkinson_target_data, stratify=parkinson_target_data, random_state=k)
        Y_train, Y_test = Y_train.values.ravel(), Y_test.values.ravel()

        # PCAによる次元削減
        X_train_pca, X_test_pca, num_components = perform_pca(X_train, X_test)
        num_components_list.append(num_components)

        # SVMモデルの学習
        clf = train_svm(X_train_pca, Y_train)

        # 予測と評価
        Y_pred = clf.predict(X_test_pca)
        evaluate_model(Y_test, Y_pred, acc_list, pre_list, rec_list, f1_list)

    # 結果を表示（指定されたフォーマット）
    print(format(np.mean(acc_list), '.3f') + "±" + format(np.std(acc_list), '.3f') 
          + "(" + format(np.min(acc_list), '.3f') + "-" + format(np.max(acc_list), '.3f') + ")")
    print(format(np.mean(pre_list), '.3f') + "±" + format(np.std(pre_list), '.3f') 
          + "(" + format(np.min(pre_list), '.3f') + "-" + format(np.max(pre_list), '.3f') + ")")
    print(format(np.mean(rec_list), '.3f') + "±" + format(np.std(rec_list), '.3f') 
          + "(" + format(np.min(rec_list), '.3f') + "-" + format(np.max(rec_list), '.3f') + ")")
    print(format(np.mean(f1_list), '.3f') + "±" + format(np.std(f1_list), '.3f') 
          + "(" + format(np.min(f1_list), '.3f') + "-" + format(np.max(f1_list), '.3f') + ")")

main()
