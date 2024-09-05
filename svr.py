import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from ccc import concordance_correlation_coefficient
import shap
from scipy.stats import spearmanr
import csv

# 場所のリスト　データの種類
variable_list = ['LeftFootIMU_acc_x', 'LeftFootIMU_acc_y', 'LeftFootIMU_acc_z',
                  'LeftLowerLegIMU_acc_x', 'LeftLowerLegIMU_acc_y', 'LeftLowerLegIMU_acc_z',
                  'LeftUpperLegIMU_acc_x', 'LeftUpperLegIMU_acc_y', 'LeftUpperLegIMU_acc_z',
                  'PelvisIMU_acc_x', 'PelvisIMU_acc_y', 'PelvisIMU_acc_z',
                  'RightFootIMU_acc_x', 'RightFootIMU_acc_y', 'RightFootIMU_acc_z',
                  'RightLowerLegIMU_acc_x', 'RightLowerLegIMU_acc_y', 'RightLowerLegIMU_acc_z',
                  'RightUpperLegIMU_acc_x', 'RightUpperLegIMU_acc_y', 'RightUpperLegIMU_acc_z',
                  ]
power_name_list = ['LeftFoot_mag', 'LeftLowerLeg_mag', 'LeftUpperLeg_mag', 'Pelvis_mag',
                    'RightFoot_mag', 'RightLowerLeg_mag', 'RightUpperLeg_mag']
angle_name_list = ['LeftAnkle_x', 'LeftAnkle_y', 'LeftAnkle_z',
                   'LeftKnee_x', 'LeftKnee_y', 'LeftKnee_z',
                   'LeftHip_x', 'LeftHip_y', 'LeftHip_z',
                   'RightHip_x', 'RightHip_y', 'RightHip_z',
                   'RightAnkle_x', 'RightAnkle_y', 'RightAnkle_z',
                   'RightKnee_x', 'RightKnee_y', 'RightKnee_z']

#step_name_list = ['Ankle_step', 'Knee_step', 'Hip_step']
step_name_list = ['Ankle_step', 'Knee_step']

dif_name_list = ['LeftAnkle_dif_x', 'LeftAnkle_dif_y', 'LeftAnkle_dif_z',
                'LeftKnee_dif_x', 'LeftKnee_dif_y', 'LeftKnee_dif_z',
                'LeftHip_dif_x', 'LeftHip_dif_y', 'LeftHip_dif_z']

dot_name_list = ['Ankle_dotproduct', 'Knee_dotproduct', 'Hip_dotproduct']

variable_list.extend(power_name_list)
variable_list.extend(angle_name_list)
variable_list.extend(step_name_list)
#variable_list.extend(dif_name_list)
variable_list.extend(dot_name_list)

# # 統計量のリスト
# statistics_list = ['max', 'min', 'mean', 'var', '25th percentile', '50th percentile', '75th percentile',
#                    'freezing index', 'central frequency', 'dominant frequency', 'amplitude', 'relative amplitude']
# #acti_list = ['Mean_neck', 'Var_neck', 'Cv_neck', 'Mean_waist', 'Var_waist', 'Cv_waist', 'Mean_diff', 'Var_diff', 'Cv_diff']
# #acti_list = ['Mean_neck', 'Mean_waist', 'Mean_diff']
# #acti_list = ['Mean_neck', 'Mean_waist']
# #acti_list = ['Mean_neck', 'Var_neck', 'Mean_waist', 'Var_waist']
# #acti_list = ['Mean_neck', 'Mean_waist']
# acti_list = ['Mean_neck']

# # 特徴量　のリスト
# feature_names = []

# for statistics in statistics_list:
#     for variable in variable_list:
#        feature_names.append(variable + " " + statistics)

feature_names = ['zure_x', 'zure_y', 'zure_z']

#feature_names.extend(acti_list)

#top5を使ったときのベスト
#index_e = 65

#feature_names.extend(lime_list_top_5[:index_e])

#default
#df = pd.read_csv("csv\\list_cut_flt.csv", skiprows = 0)
#edited
## ここに特徴量全部ぶち込んでいく
df = pd.read_csv("features/list_cut_zure.csv", skiprows = 0)


df = df[df['group'] == 1]

#tone07-1を抜く
df = df[df['ID'] != 'tone007-1']
df = df.reset_index(drop=True)

'''
#tone003-1〜005-1を抜く（actiデータなし）
df = df[df['ID'] != 'tone003-1']
df = df.reset_index(drop=True)

df = df[df['ID'] != 'tone004-1']
df = df.reset_index(drop=True)

df = df[df['ID'] != 'tone005-1']
df = df.reset_index(drop=True)

#tone083-1を抜く(外れ値)
df = df[df['ID'] != 'tone083-1']
df = df.reset_index(drop=True)
'''

#default
#df_updrs_list = pd.read_csv("csv\\updrs_list.csv", skiprows = 0)


#edited
# updrs (~108) いじらなくてよい　目的の値
df_updrs_list = pd.read_csv("target/updrs_list.csv", skiprows = 0)

#tone07-1を抜く
df_updrs_list = df_updrs_list[df_updrs_list['ID'] != 'tone007-1']

'''
df_updrs_list = df_updrs_list[df_updrs_list['ID'] != 'tone003-1']
df_updrs_list = df_updrs_list[df_updrs_list['ID'] != 'tone004-1']
df_updrs_list = df_updrs_list[df_updrs_list['ID'] != 'tone005-1']
df_updrs_list = df_updrs_list[df_updrs_list['ID'] != 'tone083-1']
'''
# 目的変数
parkinson_target_data = df_updrs_list["UPDRS"]
## 説明変数
parkinson_df = df.loc[:, feature_names]

# 決定係数　平均　絶対誤差　一致相関係数
# テストは10回やってその評価の平均を取る
r2_list = []
mae_list = []
ccc_list = []
corr_list = []

r2_train_list = []
mae_train_list = []
ccc_train_list = []

num_components_list = []
rank_shap_list = []

max_ccc_avg = 0
min_ccc_std = 1
highest_pt = 0


for k in range(50,60):
    X_train, X_test, Y_train, Y_test = train_test_split(parkinson_df, parkinson_target_data, test_size = 0.25, random_state = k)

    Y_train = Y_train.values.ravel()    #型変換
    Y_test = Y_test.values.ravel()      #型変換
    
    #標準化
    scaler = StandardScaler()
    scaler.fit(X_train)       
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #正規化　updrsを0~1に
    xmin = 0
    xmax = 108
    Y_train_scaled = [(x - xmin)/(xmax - xmin) for x in Y_train]
    Y_test_scaled = [(x - xmin)/(xmax - xmin) for x in Y_test]
    
    
    
    #PCA　似たようなやつを近似していく
    pca_temp = PCA(n_components = 0.9999)
    pca_temp.fit(X_train_scaled)
    use_component_number = 0
    eigenvalue_list = pca_temp.explained_variance_
    for eigenvalue, count in zip(eigenvalue_list, range(len(eigenvalue_list))):
        if(eigenvalue < 1):
            use_component_number = count
            break
    pca = PCA(n_components = use_component_number)
    pca.fit(X_train_scaled)
    X_train_scaled_pca = pca.transform(X_train_scaled)
    X_test_scaled_pca = pca.transform(X_test_scaled)
    
    # print(str(k) + ":")
    # print("使用する主成分の数：" + str(use_component_number))
    # print("使用する主成分の割合：" + str(sum(pca.explained_variance_ratio_)))
    
    #クロスバリデーションを用いたグリッドサーチ
    #総当たり
    # search_params = {'C':[1e-4, 0.001, 0.01, 0.1, 1, 10], 'gamma':2 ** np.arange(-10, 1, dtype=float), 'epsilon':2 ** np.arange(-10, 1, dtype=float), 'kernel':['rbf']}
    # regr = SVR()
    # gs = GridSearchCV(estimator = regr, param_grid = search_params, cv = 5, n_jobs = -1)
    # gs.fit(X_train_scaled_pca, Y_train_scaled)
    
    #高速化手法
    svr_cs = 2 ** np.arange(-5, 11, dtype=float)  # Candidates of C
    svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)  # Candidates of epsilon
    svr_gammas = 2 ** np.arange(-20, 11, dtype=float)  # Candidates of gamma
    fold_number = 5
    
    # Optimize gamma by maximizing variance in Gram matrix
    numpy_autoscaled_Xtrain = np.array(X_train_scaled_pca)
    variance_of_gram_matrix = list()
    for svr_gamma in svr_gammas:
        gram_matrix = np.exp(
            -svr_gamma * ((numpy_autoscaled_Xtrain[:, np.newaxis] - numpy_autoscaled_Xtrain) ** 2).sum(axis=2))
        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
        optimal_svr_gamma = svr_gammas[np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
        
    # Optimize epsilon with cross-validation
    svr_model_in_cv = GridSearchCV(SVR(kernel='rbf', C=3, gamma=optimal_svr_gamma), {'epsilon': svr_epsilons}, cv=fold_number)
    svr_model_in_cv.fit(X_train_scaled_pca, Y_train_scaled)
    optimal_svr_epsilon = svr_model_in_cv.best_params_['epsilon']
    
    # Optimize C with cross-validation
    svr_model_in_cv = GridSearchCV(SVR(kernel='rbf', epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma), {'C': svr_cs}, cv=fold_number)
    svr_model_in_cv.fit(X_train_scaled_pca, Y_train_scaled)
    optimal_svr_c = svr_model_in_cv.best_params_['C']
    
    # Optimize gamma with cross-validation (optional)
    svr_model_in_cv = GridSearchCV(SVR(kernel='rbf', epsilon=optimal_svr_epsilon, C=optimal_svr_c), {'gamma': svr_gammas}, cv=fold_number)
    svr_model_in_cv.fit(X_train_scaled_pca, Y_train_scaled)
    optimal_svr_gamma = svr_model_in_cv.best_params_['gamma']
    
    # Check optimized hyperparameters
    # print("C: {0}, Epsion: {1}, Gamma: {2}".format(optimal_svr_c, optimal_svr_epsilon, optimal_svr_gamma))
    
    #regr = SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma, max_iter=15)
    # regrにモデルが入る

    regr = SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma, max_iter=1000)
    # モデルの訓練
    regr.fit(X_train_scaled_pca, Y_train_scaled)

    
    #てすとする
    Y_pred = regr.predict(X_test_scaled_pca)
    
    r2 = r2_score(Y_test_scaled, Y_pred)
    mae = mean_absolute_error(Y_test_scaled, Y_pred)
    ccc = concordance_correlation_coefficient(Y_test_scaled, Y_pred)
    corr = spearmanr(Y_test_scaled, Y_pred)[0]
    
    r2_list.append(r2)
    mae_list.append(mae)
    ccc_list.append(ccc)
    corr_list.append(corr)
    
    Y_train_pred = regr.predict(X_train_scaled_pca)
    
    r2_train = r2_score(Y_train_scaled, Y_train_pred)
    mae_train = mean_absolute_error(Y_train_scaled, Y_train_pred)
    ccc_train = concordance_correlation_coefficient(Y_train_scaled, Y_train_pred)
    
    r2_train_list.append(r2_train)
    mae_train_list.append(mae_train)
    ccc_train_list.append(ccc_train)
    
    plt.xlabel("Measured value")
    plt.ylabel("Estimated value")
    plt.scatter(Y_test_scaled, Y_pred)
    
    x = np.arange(0, 0.5, 0.01)
    y = x
    plt.plot(x, y)
    
    #default
    #file_name = "image\\regression_updrs2\\svr_linear_" + str(k) + ".png"
    
    #edited
    file_name = "image/regression_updrs2/svr_linear_" + str(k) + ".png"

    
    # plt.savefig(file_name)
    
    plt.show()
    #plt.close()
    
    print("train:")
    print(ccc_train)    
    print("test:")
    print(ccc)
    print()
    
    '''
    if k >= 9:
        recent_ccc_avg = np.mean(ccc_list[k-9:k+1])
        recent_ccc_std = np.std(ccc_list[k-9:k+1])
        print(f"Iteration {k+1}: ccc Recent 10 Avg = {recent_ccc_avg:.3f} ± {recent_ccc_std:.3f}")
        recent_mae_avg = np.mean(mae_list[k-9:k+1])
        recent_mae_std = np.std(mae_list[k-9:k+1])
        print(f"Iteration {k+1}: mae Recent 10 Avg = {recent_mae_avg:.3f} ± {recent_mae_std:.3f}")
        recent_r2_avg = np.mean(r2_list[k-9:k+1])
        recent_r2_std = np.std(r2_list[k-9:k+1])
        print(f"Iteration {k+1}: r2 Recent 10 Avg = {recent_r2_avg:.3f} ± {recent_r2_std:.3f}")

        if max_ccc_avg < recent_ccc_avg:
            max_ccc_avg = recent_ccc_avg
            highest_pt = k
        elif max_ccc_avg == recent_ccc_avg:
            if min_ccc_std > recent_ccc_std:
                min_ccc_std = recent_ccc_std
                highest_pt = k
    '''

    
    # #shap可視化
    # explainer = shap.KernelExplainer(regr.predict, X_train_scaled_pca)
    # shap_values = explainer.shap_values(X_test_scaled_pca)
    # shap.summary_plot(shap_values, X_test_scaled_pca, plot_type = 'bar')
    
    # mean_shap_values = np.mean(np.abs(shap_values), axis = 0)
    # rank_shap_values = np.argsort(- mean_shap_values)
    # rank_shap_list.append(rank_shap_values)

print("train:")
print(format(np.mean(r2_train_list), '.3f') + "±" + format(np.std(r2_train_list), '.3f') 
      + "(" + format(np.min(r2_train_list), '.3f') + "-" + format(np.max(r2_train_list), '.3f') + ")")
print(format(np.mean(mae_train_list), '.3f') + "±" + format(np.std(mae_train_list), '.3f') 
      + "(" + format(np.min(mae_train_list), '.3f') + "-" + format(np.max(mae_train_list), '.3f') + ")")
print(format(np.mean(ccc_train_list), '.3f') + "±" + format(np.std(ccc_train_list), '.3f') 
      + "(" + format(np.min(ccc_train_list), '.3f') + "-" + format(np.max(ccc_train_list), '.3f') + ")")
#print(format(np.mean(r2_train_list), '.3f') + "±" + format(np.std(r2_train_list), '.3f') )
#print(format(np.mean(mae_train_list), '.3f') + "±" + format(np.std(mae_train_list), '.3f') )
#print(format(np.mean(ccc_train_list), '.3f') + "±" + format(np.std(ccc_train_list), '.3f') )
print("test:")
print(format(np.mean(r2_list), '.3f') + "±" + format(np.std(r2_list), '.3f') 
      + "(" + format(np.min(r2_list), '.3f') + "-" + format(np.max(r2_list), '.3f') + ")")
print(format(np.mean(mae_list), '.3f') + "±" + format(np.std(mae_list), '.3f') 
      + "(" + format(np.min(mae_list), '.3f') + "-" + format(np.max(mae_list), '.3f') + ")")
print(format(np.mean(ccc_list), '.3f') + "±" + format(np.std(ccc_list), '.3f') 
      + "(" + format(np.min(ccc_list), '.3f') + "-" + format(np.max(ccc_list), '.3f') + ")")
# print(format(np.mean(corr_list), '.3f') + "±" + format(np.std(corr_list), '.3f') 
#       + "(" + format(np.min(corr_list), '.3f') + "-" + format(np.max(corr_list), '.3f') + ")")

#shap値のランク作成
# rank_shap_list20 = []
# rank_feature_name20 = []
# for rank_list in rank_shap_list:
#     rank_shap_list20.extend(rank_list[0:20])
    
# for rank in rank_shap_list20:
#     rank_feature_name20.append(feature_names[rank])
    
# df_shap = pd.DataFrame(rank_shap_list)
# df_shap.to_excel("regression_shap.xlsx")
'''
header = ['r2', 'mae', 'ccc']

with open('csv/svr_withneck.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    csvwriter.writerow(header)

    rows = zip(r2_train_list, mae_train_list, ccc_train_list)
    for row in rows:
        csvwriter.writerow(row)
'''

'''
k = highest_pt
recent_ccc_avg = np.mean(ccc_list[k-9:k+1])
recent_ccc_std = np.std(ccc_list[k-9:k+1])
print(f"Iteration {k+1}: ccc Recent 10 Avg = {recent_ccc_avg:.3f} ± {recent_ccc_std:.3f}")
recent_mae_avg = np.mean(mae_list[k-9:k+1])
recent_mae_std = np.std(mae_list[k-9:k+1])
print(f"Iteration {k+1}: mae Recent 10 Avg = {recent_mae_avg:.3f} ± {recent_mae_std:.3f}")
recent_r2_avg = np.mean(r2_list[k-9:k+1])
recent_r2_std = np.std(r2_list[k-9:k+1])
print(f"Iteration {k+1}: r2 Recent 10 Avg = {recent_r2_avg:.3f} ± {recent_r2_std:.3f}")
'''

    

       

       