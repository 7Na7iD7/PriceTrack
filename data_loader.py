import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoadError(Exception):
    pass


class DataLoader:
    def __init__(self, csv_path, target_column="price_range", test_size=0.3):
        self.csv_path = csv_path
        self.target_column = target_column
        self.test_size = test_size
        self.scaler = StandardScaler()

    def load_raw(self):
        if not os.path.exists(self.csv_path):
            raise DataLoadError(f"فایل دیتاست در مسیر '{self.csv_path}' یافت نشد.")
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as exc:
            raise DataLoadError(f"خطا در خواندن فایل CSV: {exc}")
        if self.target_column not in df.columns:
            raise DataLoadError(f"ستون هدف '{self.target_column}' در دیتاست وجود ندارد.")
        return df

    def split_and_scale(self):
        df = self.load_raw()
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train.values, y_test.values, list(X.columns)
