import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tffs import get_frequency_of_feature_by_percent

file_name="Brain.csv"
file_path=file_name
data = df = pd.read_csv(file_path)

def get_features_by_backward_and_tffs(data, percent_tffs, number_run, n_estimators, percent_forward):
    index_TFFS_percent = get_frequency_of_feature_by_percent(df, number_run, percent_tffs, n_estimators)
    X = data.iloc[:, 1:]
    X_original = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    X_new = X_original.iloc[:, index_TFFS_percent]
    total_features = data.shape[1] - 1
    num_selected_features = max(1, round(percent_forward * total_features / 100))  # Lấy 1% số lượng cột, tối thiểu 1 cột
    # Thiết lập số lượng đặc trưng cần giữ lại
    num_features_to_keep = num_selected_features  # Số lượng đặc trưng mong muốn
    # Danh sách đặc trưng ban đầu
    selected_features = list(X.columns)  # Ban đầu chọn tất cả đặc trưng
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Bắt đầu quá trình loại bỏ đặc trưng dần dần
    while len(selected_features) > num_features_to_keep:
        scores = []  # Lưu trữ lỗi MSE sau khi loại từng đặc trưng
        for feature in selected_features:
            # Tạo danh sách đặc trưng sau khi loại bỏ `feature`
            current_features = [f for f in selected_features if f != feature]

            # Huấn luyện mô hình với tập đặc trưng hiện tại
            model = LinearRegression()
            model.fit(X_train[current_features], y_train)
            y_pred = model.predict(X_test[current_features])
            score = mean_squared_error(y_test, y_pred)
            scores.append((feature, score))

        # Tìm đặc trưng có tác động nhỏ nhất (khi loại bỏ mà lỗi tăng ít nhất)
        scores.sort(key=lambda x: x[1])  # Sắp xếp theo MSE tăng dần
        worst_feature = scores[-1][0]  # Lấy đặc trưng có lỗi lớn nhất khi loại bỏ

        # Xóa đặc trưng có ảnh hưởng ít nhất
        selected_features.remove(worst_feature)

        print(f"Loại bỏ đặc trưng: {worst_feature}, MSE: {scores[-1][1]}")

    # Kết quả cuối cùng
    print("\nCác đặc trưng còn lại sau Backward Selection:")
    print(selected_features)
    return selected_features

selected_features = get_features_by_backward_and_tffs(data, 5, 20, 50, 1)
print("\nTop features selected:")
print(selected_features)
print(len(selected_features))
