"""UCI 가정 전력 일 패턴 군집 분석 (ICESI 실험 스크립트).

- get_data: preprocess_uci.py 산출물(dataset.csv) 로드
- classify_pattern: 하루 96개(15분) 사용량 + 기상 + 시간 특징 → t-SNE/UMAP 차원축소 → KMeans 군집
- pattern_plot: 군집별 평균 일 패턴 그림
- split_data / fit_clf_model / fit_pred_model: 군집 결과를 이용한 분류 · 예측 실험 초안 (미완성, 실행 보장 안 함)

사용: python icesi/modeling.py <dataset.csv 와 weather_data.csv 가 있는 폴더>
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

class modeling:

    def __init__(self, current_path="."):
        self.current_path = current_path.rstrip("/") + "/"
        self.data = "not defined yet"
        self.X_train = "not defined yet"
        self.X_test = "not defined yet"
        self.y_train = "not defined yet"
        self.y_test = "not defined yet"
        self.daily_pattern = "not defined yet"
        self.clust_id_set = "not defined yet"

    # 데이터셋 가져오기
    def get_data(self):
        data = pd.read_csv(self.current_path + 'dataset.csv')
        data.DateTime = pd.to_datetime(data.DateTime)
        self.data = data.set_index('DateTime')

        return self.data

    # 패턴 클래스 정의
    def classify_pattern(self, reduce_dim=False, n_dim=2, n_cluster=3):

        # 온전한 하루 단위 데이터만 사용
        elec = self.data.electricity_consumption[27:138267]

        # 96개의 15분 단위 전기사용량 데이터(1440일 분량)
        daily_pattern = []
        for i in range(96):
            daily_pattern.append(elec[i::96])

        # 날씨 데이터 추가
        weather_data = pd.read_csv(self.current_path + "weather_data.csv")
        weather_data.index = pd.to_datetime(weather_data[weather_data.columns[0]], format="%Y-%m-%d")
        weather_data = weather_data.loc[self.data.index[0]:self.data.index[-1]]
        weather_data = weather_data.iloc[1:]

        daily_pattern.append(weather_data.temp_high.values)
        daily_pattern.append(weather_data.temp_low.values)
        daily_pattern.append(weather_data.humid_high.values)
        daily_pattern.append(weather_data.humid_low.values)

        # 시간 데이터 추가
        daily_pattern.append(self.data.minute_x[weather_data.index])
        daily_pattern.append(self.data.minute_y[weather_data.index])
        daily_pattern.append(self.data.hour_x[weather_data.index])
        daily_pattern.append(self.data.hour_y[weather_data.index])
        daily_pattern.append(self.data.day_x[weather_data.index])
        daily_pattern.append(self.data.day_y[weather_data.index])
        daily_pattern.append(self.data.month_x[weather_data.index])
        daily_pattern.append(self.data.month_y[weather_data.index])

        self.daily_pattern = np.array(daily_pattern).T
        print(np.shape(self.daily_pattern))

        # 차원축소
        X = self.daily_pattern
        if reduce_dim:
            if reduce_dim == "tsne":
                X = TSNE(n_components=n_dim).fit_transform(X)
            else:
                X = UMAP(n_components=n_dim).fit_transform(X)
        else:
            X = X
        
        # 클러스터링
        model_clust = KMeans(n_clusters=n_cluster, random_state=0)
        self.clust_id_set = model_clust.fit_predict(X)

        return self.clust_id_set
    
    # 클래스별 패턴 시각화
    def pattern_plot(self):
        df = pd.DataFrame(self.daily_pattern)
        df['clust_id'] = self.clust_id_set

        clust_1_data = df[df['clust_id'] == 0]
        clust_2_data = df[df['clust_id'] == 1]
        clust_3_data = df[df['clust_id'] == 2]

        c1_average = list(clust_1_data.mean())[:96]
        c2_average = list(clust_2_data.mean())[:96]
        c3_average = list(clust_3_data.mean())[:96]

        plt.figure(figsize=(10, 6))
        plt.plot(c1_average, label='Class 1')
        plt.plot(c2_average, label='Class 2')
        plt.plot(c3_average, label='Class 3')
        plt.xlabel('Time(1 Day)')
        plt.ylabel('kW/h')
        plt.legend()
        plt.show()
    
    # 학습 및 테스트용 데이터셋 구성
    def split_data(self):
        data = self.data
        X_train, X_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), [], []

        for dayofweek in ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']:
            
            # 요일별 데이터셋 생성
            data_temp = data[data[dayofweek] == 1]

            # 데이터셋 분리i
            X_t, X_v, y_t, y_v = train_test_split(
                data_temp, data_temp.electricity_consumption, test_size=0.2, random_state=1)
            
            # 최종 데이터셋에 추가
            X_train = pd.concat([X_train, X_t], axis=0)
            X_test = pd.concat([X_test, X_v], axis=0)
            y_train += list(y_t)
            y_test += list(y_v)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)

        return X_train, X_test, y_train, y_test

    def analyze_correlation(self):
        pass

    # 모델 학습
    def fit_clf_model(self, model):
        # 모델 선택
        if model == "nb":
            model = MultinomialNB()
        elif model == "xgb":
            model = XGBClassifier()
        elif model == "svm":
            model = svm.SVC(kernel='linear')

        # 모델 학습
        print(type(self.y_train))
        # print(self.y_train)
        model.fit(self.X_train.values[:-1], self.y_train[1:])

        # 예측 수행
        predictions = model.predict(self.X_test)

        return predictions
    
    def fit_pred_model(self, class_pred):
        # 데이터 로드
        X_train = self.X_train
        X_train['clust_id'] = self.clust_id_set
        X_train_set = X_train.groupby('clust_id')

        X_test = self.X_test
        X_test['clust_id'] = class_pred
        X_test_set = X_train.groupby('clust_id')

        y_pred = []
        for id in range(3):
            X_train = X_train_set[id]
            X_test = X_test_set[id]
            
            X_train = X_train[X_train.columns[:-1]].values
            X_test = X_test[X_test.columns[:-1]].values

            # ANN 모델 생성
            mlp = MLPClassifier(hidden_layer_sizes=(7,), activation='logistic', random_state=42)

            # 모델 학습
            mlp.fit(X_train[:-1], self.y_train[1:])

            # 예측 수행
            y_pred += mlp.predict(X_test)

        # 정확도 평가
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", accuracy)
            


if __name__ == "__main__":
    import sys
    import matplotlib
    matplotlib.use("Agg")
    m = modeling(sys.argv[1] if len(sys.argv) > 1 else ".")
    m.get_data()
    clusters = m.classify_pattern(reduce_dim="tsne", n_dim=2, n_cluster=3)
    print("cluster sizes:", {int(k): int(v) for k, v in zip(*__import__("numpy").unique(clusters, return_counts=True))})
    m.pattern_plot()
