import pandas as pd
from sklearn.metrics import f1_score

# Функция для оценки на приватном лидерборде (все данные)
def calc_f1_score_from_files(test_url: str, prediction_url: str) -> float:
    true_labels = pd.read_csv(test_url)
    pred_labels = pd.read_csv(prediction_url)
    # Таргет для месячного прогноза
    true_labels_month = true_labels['target_month'].values
    pred_labels_month = pred_labels['target_month'].values

    # Таргет для 10 дневного прогноза
    true_labels_day = true_labels['target_day'].values
    pred_labels_day = pred_labels['target_day'].values

    # Посчитаем метрику для месяца и 10 дней
    score_month = f1_score(true_labels_month, pred_labels_month)
    score_day = f1_score(true_labels_day, pred_labels_day)
    # Посчитаем метрику с весом для двух таргетов
    score = 0.5 * score_month + 0.5 * score_day
    return score


def calc_f1_score_from_data(test, prediction) -> float:
    true_labels = test
    pred_labels = prediction
    # Таргет для месячного прогноза
    true_labels_month = true_labels['target_month'].astype(int).values
    pred_labels_month = pred_labels['target_month'].astype(int).values

    # Таргет для 10 дневного прогноза
    true_labels_day = true_labels['target_day'].astype(int).values
    pred_labels_day = pred_labels['target_day'].astype(int).values

    # Посчитаем метрику для месяца и 10 дней
    score_month = f1_score(true_labels_month, pred_labels_month)
    score_day = f1_score(true_labels_day, pred_labels_day)
    # Посчитаем метрику с весом для двух таргетов
    score = 0.5 * score_month + 0.5 * score_day
    return score
