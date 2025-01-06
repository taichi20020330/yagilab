import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from ccc import concordance_correlation_coefficient

# 固有の変数設定
DATA_FILE_PATH = "features/extracted_features.csv"
TARGET_FILE_PATH = "features/updrs_list.csv"
EXCLUDE_IDS = []

# 定数設定
UPDRS_XMIN = 0
UPDRS_XMAX = 108

# 使用する変数リストと統計量
variable_list = ['x_acc', 'y_acc', 'z_acc']
# statistics_list = ['sum_values', 'median', 'mean', 'length', 'standard_deviation', 'variance', 'root_mean_square', 'maximum', 'absolute_maximum', 'minimum']
statistics_list = ['max', 'min', 'mean', 'median', 'std', 'var', 'sum']
spectal_feature_list = ['freezing_index', 'central_frequency','dominant_frequency','amplitude','relative_amplitude']
parcentile_list = ['percentile25', 'percentile50', 'percentile75']

# データの前処理
def preprocess_data(data_file_path, target_file_path, exclude_ids, feature_names):
    df = pd.read_csv(data_file_path)
    df = df[df['group'] == 1]
    for exclude_id in exclude_ids:
        df = df[df['ID'] != exclude_id]
    df = df.reset_index(drop=True)

    df_updrs_list = pd.read_csv(target_file_path)
    df_updrs_list = df_updrs_list[~df_updrs_list['ID'].isin(exclude_ids)]

    parkinson_target_data = df_updrs_list["UPDRS"]
    parkinson_df = df.loc[:, feature_names]

    return parkinson_df, parkinson_target_data

# データの分割とスケーリング
def split_and_scale_data(X, Y, random_state, test_size=0.25):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    Y_train_scaled = [(x - UPDRS_XMIN) / (UPDRS_XMAX - UPDRS_XMIN) for x in Y_train]
    Y_test_scaled = [(x - UPDRS_XMIN) / (UPDRS_XMAX - UPDRS_XMIN) for x in Y_test]

    return X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled

# PCAによる次元削減
def perform_pca(X_train, X_test, explained_variance_threshold=0.9999):
    pca_temp = PCA(n_components=explained_variance_threshold)
    pca_temp.fit(X_train)
    eigenvalue_list = pca_temp.explained_variance_
    use_component_number = sum(eig > 1 for eig in eigenvalue_list)

    pca = PCA(n_components=use_component_number)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca, use_component_number

# SVRの最適化
def optimize_svr(X_train, Y_train):
    svr_gammas = 2 ** np.arange(-20, 11, dtype=float)
    svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)
    svr_cs = 2 ** np.arange(-5, 11, dtype=float)

    # 最適なgammaを選択
    variance_of_gram_matrix = [
        np.exp(-gamma * ((X_train[:, np.newaxis] - X_train) ** 2).sum(axis=2)).var(ddof=1)
        for gamma in svr_gammas
    ]
    optimal_gamma = svr_gammas[np.argmax(variance_of_gram_matrix)]

    # epsilonを最適化
    model_epsilon = GridSearchCV(SVR(kernel='rbf', C=3, gamma=optimal_gamma), {'epsilon': svr_epsilons}, cv=5)
    model_epsilon.fit(X_train, Y_train)
    optimal_epsilon = model_epsilon.best_params_['epsilon']

    # Cを最適化
    model_c = GridSearchCV(SVR(kernel='rbf', epsilon=optimal_epsilon, gamma=optimal_gamma), {'C': svr_cs}, cv=5)
    model_c.fit(X_train, Y_train)
    optimal_c = model_c.best_params_['C']

    return SVR(kernel='rbf', C=optimal_c, epsilon=optimal_epsilon, gamma=optimal_gamma, max_iter=1000)

# モデル評価
def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    ccc = concordance_correlation_coefficient(Y_test, Y_pred)
    corr = spearmanr(Y_test, Y_pred)[0]

    return r2, mae, ccc, corr

# プロット関数
def plot_results(Y_test, Y_pred, iteration):
    plt.scatter(Y_test, Y_pred)
    plt.plot(np.arange(0, 0.5, 0.01), np.arange(0, 0.5, 0.01), color='red')
    plt.xlabel("Measured value")
    plt.ylabel("Estimated value")
    plt.title(f"Iteration {iteration}")
    plt.show()

# メイン処理
def main():
    feature_names = [f"{var}__{stat}" for stat in statistics_list  + parcentile_list + spectal_feature_list for var in variable_list] 
    parkinson_df, parkinson_target_data = preprocess_data(DATA_FILE_PATH, TARGET_FILE_PATH, EXCLUDE_IDS, feature_names)

    r2_list, mae_list, ccc_list, corr_list = [], [], [], []

    for k in range(50, 60):
        X_train, X_test, Y_train, Y_test = split_and_scale_data(parkinson_df, parkinson_target_data, k)
        X_train_pca, X_test_pca, _ = perform_pca(X_train, X_test)
        svr_model = optimize_svr(X_train_pca, Y_train)
        svr_model.fit(X_train_pca, Y_train)
        
        r2, mae, ccc, corr = evaluate_model(svr_model, X_test_pca, Y_test)
        r2_train, mae_train, ccc_train, corr_train = evaluate_model(svr_model, X_train_pca, Y_train)
        r2_list.append(r2)
        mae_list.append(mae)
        ccc_list.append(ccc)
        corr_list.append(corr)

        plot_results(Y_test, svr_model.predict(X_test_pca), k)

    print("train:")
    print("r2:" + format(np.mean(r2_train), '.3f') + "±" + format(np.std(r2_train), '.3f') 
        + "(" + format(np.min(r2_train), '.3f') + "-" + format(np.max(r2_train), '.3f') + ")")
    print("mae:" + format(np.mean(mae_train), '.3f') + "±" + format(np.std(mae_train), '.3f') 
        + "(" + format(np.min(mae_train), '.3f') + "-" + format(np.max(mae_train), '.3f') + ")")
    print("ccc:" + format(np.mean(ccc_train), '.3f') + "±" + format(np.std(ccc_train), '.3f') 
        + "(" + format(np.min(ccc_train), '.3f') + "-" + format(np.max(ccc_train), '.3f') + ")")
    print("corr:" + format(np.mean(corr_train), '.3f') + "±" + format(np.std(corr_train), '.3f') 
        + "(" + format(np.min(corr_train), '.3f') + "-" + format(np.max(corr_train), '.3f') + ")")

    print("test:")
    print("r2:" + format(np.mean(r2_list), '.3f') + "±" + format(np.std(r2_list), '.3f') 
        + "(" + format(np.min(r2_list), '.3f') + "-" + format(np.max(r2_list), '.3f') + ")")
    print("mae:" + format(np.mean(mae_list), '.3f') + "±" + format(np.std(mae_list), '.3f') 
        + "(" + format(np.min(mae_list), '.3f') + "-" + format(np.max(mae_list), '.3f') + ")")
    print("ccc:" + format(np.mean(ccc_list), '.3f') + "±" + format(np.std(ccc_list), '.3f') 
        + "(" + format(np.min(ccc_list), '.3f') + "-" + format(np.max(ccc_list), '.3f') + ")")
    print("corr:" + format(np.mean(corr_list), '.3f') + "±" + format(np.std(corr_list), '.3f') 
        + "(" + format(np.min(corr_list), '.3f') + "-" + format(np.max(corr_list), '.3f') + ")")


if __name__ == "__main__":
    main()
