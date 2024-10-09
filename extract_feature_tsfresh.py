import pandas as pd
import glob
import os
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction.settings import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

# データの整形関数
def process_csv(file_path):
    df = pd.read_csv(file_path, header=None, names=['x_acc', 'y_acc', 'z_acc'])
    df['time'] = df.index * 0.01  # 行番号に基づいて時間を作成
    df_melted = pd.melt(df, id_vars=['time'], value_vars=['x_acc', 'y_acc', 'z_acc'],
                        var_name='variable', value_name='value')
    return df_melted

# 特徴量抽出関数
def extract_features_from_file(file_path):
    df = pd.read_csv(file_path)
    df['id'] = 1  # 特定のIDで統一 (1つのセグメントを持つと仮定)
    
    # absolute_sum_of_changesの抽出
    extraction_settings = MinimalFCParameters()
    extracted_features = extract_features(df, column_id='id', column_sort='time', 
                                          column_kind='variable', column_value='value', 
                                          default_fc_parameters=extraction_settings)
    
    impute(extracted_features)  # 欠損値の補完
    return extracted_features[['absolute_sum_of_changes']]  # 必要な特徴量だけを抽出

# CSVファイルの処理と保存関数
def process_and_save_all_csvs(input_folder_path, output_folder_path):
    # if not os.path.exists(output_folder_path):
    #     os.makedirs(output_folder_path)

    csv_files = glob.glob(os.path.join(input_folder_path, '*.csv'))
    print(f'処理するファイル数: {len(csv_files)}')  # 確認のためファイル数を表示
    
    for file in csv_files:
        base_name = os.path.basename(file)
        output_file = os.path.join(output_folder_path, f'{os.path.splitext(base_name)[0]}_features.csv')
        
        # データの整形
        processed_data = process_csv(file)
        processed_data.to_csv(output_file, index=False)
        
        # 特徴量の抽出
        features = extract_features_from_file(output_file)
        print(f"{base_name}の特徴量: \n{features}")

        # 結果を保存
        features.to_csv(output_file, index=False)
        
    print(f'すべてのCSVファイルの処理が完了し、{output_folder_path} に保存しました。')

# メイン処理
def main():
    parkin_input_path = '/content/drive/My Drive/ColabNotebooks/assets/row_data/parkin/'
    normal_input_path = '/content/drive/My Drive/ColabNotebooks/assets/row_data/normal/'
    parkin_output_path = '/content/drive/My Drive/ColabNotebooks/assets/tsfresh/parkin/'
    normal_output_path = '/content/drive/My Drive/ColabNotebooks/assets/tsfresh/normal/'
    
    # parkinデータの処理
    print("parkinデータの処理を開始します")
    process_and_save_all_csvs(parkin_input_path, parkin_output_path)
    
    # normalデータの処理
    print("normalデータの処理を開始します")
    process_and_save_all_csvs(normal_input_path, normal_output_path)

# 実行
main()
