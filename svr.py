import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from ccc import concordance_correlation_coefficient

# パスやフォルダ名をまとめる
DATA_PATH = {
    "features": "features/list_cut_zure.csv",
    "target": "target/updrs_list.csv",
    "output_image": "image/regression_updrs2/svr_linear_"
}

# 固定の値
UPDRS_XMIN = 0
UPDRS_XMAX = 108

# 特徴量リストをまとめる
FEATURE_LIST = {
    'variable': ['LeftFootIMU_acc_x', 'LeftFootIMU_acc_y', 'LeftFootIMU_acc_z',
                 'LeftLowerLegIMU_acc_x', 'LeftLowerLegIMU_acc_y', 'LeftLowerLegIMU_acc_z',
                 'LeftUpperLegIMU_acc_x', 'LeftUpperLegIMU_acc_y', 'LeftUpperLegIMU_acc_z',
                 'PelvisIMU_acc_x', 'PelvisIMU_acc_y', 'PelvisIMU_acc_z',
                 'RightFootIMU_acc_x', 'RightFootIMU_acc_y', 'RightFootIMU_acc_z',
                 'RightLowerLegIMU_acc_x', 'RightLowerLegIMU_acc_y', 'RightLowerLegIMU_acc_z',
                 'RightUpperLegIMU_acc_x', 'RightUpperLegIMU_acc_y', 'RightUpperLegIMU_acc_z'],
    'power': ['LeftFoot_mag', 'LeftLowerLeg_mag', 'LeftUpperLeg_mag', 'Pelvis_mag',
              'RightFoot_mag', 'RightLowerLeg_mag', 'RightUpperLeg_mag'],
    'angle': ['LeftAnkle_x', 'LeftAnkle_y', 'LeftAnkle_z', 'LeftKnee_x', 'LeftKnee_y', 'LeftKnee_z',
              'LeftHip_x', 'LeftHip_y', 'LeftHip_z', 'RightHip_x', 'RightHip_y', 'RightHip_z',
              'RightAnkle_x', 'RightAnkle_y', 'RightAnkle_z', 'RightKnee_x', 'RightKnee_y', 'RightKnee_z'],
    'step': ['Ankle_step', 'Knee_step'],
    'dot': ['Ankle_dotproduct', 'Knee_dotproduct', 'Hip_dotproduct']
}

# データの読み込みと前処理
def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH["features"])
    df = df[df['group'] == 1]
    df = df[df['ID'] != 'tone007-1']
    df = df.reset_index(drop=True)

    df_updrs_list = pd.read_csv(DATA_PATH["target"])
    df_updrs_list = df_updrs_list[df_updrs_list['ID'] != 'tone007-1']

    parkinson_target_data = df_updrs_list["UPDRS"]
    feature_names = ['zure_x', 'zure_y', 'zure_z']
    parkinson_df = df.loc[:, feature_names]

    return parkinson_df, parkinson_target_data

# データの分割とスケーリング
def split_and_scale_data(X, Y, test_size=0.25, random_state=42):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    Y_train = Y_train.values.ravel()
    Y_test = Y_test.values.ravel()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    Y_train_scaled = [(x - UPDRS_XMIN) / (UPDRS_XMAX - UPDRS_XMIN) for x in Y_train]
    Y_test_scaled = [(x - UPDRS_XMIN) / (UPDRS_XMAX - UPDRS_XMIN) for x in Y_test]

    return X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled

# PCAで次元削減
def apply_pca(X_train, X_test, threshold=0.9999):
    pca_temp = PCA(n_components=threshold)
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
    fold_number = 5

    variance_of_gram_matrix = [np.exp(-gamma * ((X_train[:, np.newaxis] - X_train) ** 2).sum(axis=2)).var(ddof=1) for gamma in svr_gammas]
    optimal_gamma = svr_gammas[np.argmax(variance_of_gram_matrix)]
    
    model_epsilon = GridSearchCV(SVR(kernel='rbf', C=3, gamma=optimal_gamma), {'epsilon': svr_epsilons}, cv=fold_number)
    model_epsilon.fit(X_train, Y_train)
    optimal_epsilon = model_epsilon.best_params_['epsilon']
    
    model_c = GridSearchCV(SVR(kernel='rbf', epsilon=optimal_epsilon, gamma=optimal_gamma), {'C': svr_cs}, cv=fold_number)
    model_c.fit(X_train, Y_train)
    optimal_c = model_c.best_params_['C']
    
    return SVR(kernel='rbf', C=optimal_c, epsilon=optimal_epsilon, gamma=optimal_gamma, max_iter=1000)

# モデル評価とプロット
def evaluate_and_plot_model(model, X_test, Y_test, X_train, Y_train, iteration):
    Y_pred = model.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    ccc = concordance_correlation_coefficient(Y_test, Y_pred)
    corr = spearmanr(Y_test, Y_pred)[0]

    plt.scatter(Y_test, Y_pred)
    plt.plot(np.arange(0, 0.5, 0.01), np.arange(0, 0.5, 0.01))
    plt.xlabel("Measured value")
    plt.ylabel("Estimated value")
    plt.show()

    print(f"Iteration {iteration}: R2 = {r2}, MAE = {mae}, CCC = {ccc}, Corr = {corr}")
    
    return r2, mae, ccc, corr

# 全体の実行
def main():
    parkinson_df, parkinson_target_data = load_and_preprocess_data()
    r2_list, mae_list, ccc_list, corr_list = [], [], [], []
    
    for k in range(50, 60):
        X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled = split_and_scale_data(parkinson_df, parkinson_target_data, random_state=k)
        X_train_pca, X_test_pca, _ = apply_pca(X_train_scaled, X_test_scaled)
        svr_model = optimize_svr(X_train_pca, Y_train_scaled)
        svr_model.fit(X_train_pca, Y_train_scaled)
        r2, mae, ccc, corr = evaluate_and_plot_model(svr_model, X_test_pca, Y_test_scaled, X_train_pca, Y_train_scaled, k)
        r2_list.append(r2)
        mae_list.append(mae)
        ccc_list.append(ccc)
        corr_list.append(corr)



main()