# -*- coding: utf-8 -*-
"""CV04 — Trâm: causal rolling features and recursive 24-hour forecasting.

Only CV04 artifacts are written. Train/test boundaries come from CV01.
Rolling features use shift(1); recursive features use predictions within each
forecast window. Test targets never select hyperparameters or tree counts.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
TRAIN_DATA = DATA_PROCESSED_DIR / 'train.csv'
TEST_DATA = DATA_PROCESSED_DIR / 'test.csv'
FIGURES_DIR = PROJECT_ROOT / 'outputs' / 'figures'
TABLES_DIR = PROJECT_ROOT / 'outputs' / 'tables'
MODELS_DIR = PROJECT_ROOT / 'outputs' / 'models'
CONFIG = {'target_col': 'AEP_MW', 'rolling_windows': [24, 168], 'forecast_horizon': 24}
LAGS = list(range(1, 25)) + [168]
CALENDAR = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'is_weekend']
BASE_FEATURES = [f'lag_{lag}' for lag in LAGS] + CALENDAR
ROLL_FEATURES = [f'rolling_{stat}_{window}' for window in (24, 168) for stat in ('mean', 'std')]


def calendar(index):
    return pd.DataFrame({'hour': index.hour, 'dayofweek': index.dayofweek,
                         'quarter': index.quarter, 'month': index.month,
                         'year': index.year, 'dayofyear': index.dayofyear,
                         'is_weekend': (index.dayofweek >= 5).astype(int)}, index=index)


def validate_series(df, target_col='AEP_MW'):
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.has_duplicates:
        raise ValueError('Cần DatetimeIndex duy nhất.')
    if not df.index.is_monotonic_increasing:
        raise ValueError('Dữ liệu phải tăng dần theo thời gian.')
    if len(df) < 2 or not (df.index.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise ValueError('Chuỗi phải có tần suất giờ liên tục.')
    if not np.isfinite(df[target_col].to_numpy(dtype=float)).all():
        raise ValueError('Target chứa NaN hoặc giá trị vô hạn.')


def step1_create_rolling_features(df, target_col='AEP_MW'):
    """Build all lags and four rolling statistics using past observations only."""
    validate_series(df, target_col)
    result = df[[target_col]].copy()
    for lag in LAGS:
        result[f'lag_{lag}'] = df[target_col].shift(lag)
    result = result.join(calendar(df.index))
    past = df[target_col].shift(1)
    for window in CONFIG['rolling_windows']:
        result[f'rolling_mean_{window}'] = past.rolling(window).mean()
        result[f'rolling_std_{window}'] = past.rolling(window).std(ddof=1)
    result = result.dropna()
    if result.empty:
        raise ValueError('Cần nhiều hơn 168 giờ dữ liệu.')
    return result, FIGURES_DIR / 'cv04_rolling_features_impact.png'


def train_model(train, features, target_col='AEP_MW', gap=0):
    """Chronological validation inside training, then refit on all training rows."""
    cut = int(len(train) * 0.9)
    if cut < 168 or len(train) - cut < 24:
        raise ValueError('Tập train quá nhỏ để tách validation theo thời gian.')
    params = dict(n_estimators=1000, learning_rate=0.01, max_depth=5,
                  subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror',
                  eval_metric='rmse', random_state=42, n_jobs=4, tree_method='hist')
    selector = XGBRegressor(**params, early_stopping_rounds=50)
    selector.fit(train.iloc[:cut-gap][features], train.iloc[:cut-gap][target_col],
                 eval_set=[(train.iloc[cut:][features], train.iloc[cut:][target_col])], verbose=False)
    params['n_estimators'] = selector.best_iteration + 1
    model = XGBRegressor(**params)
    model.fit(train[features], train[target_col], verbose=False)
    model.cv04_training_info = {
        'best_n_estimators': params['n_estimators'],
        'hit_tree_budget': params['n_estimators'] == 1000,
        'validation_rmse': float(selector.best_score),
        'validation_start': str(train.index[cut]),
        'fit_rows_before_validation': cut-gap,
        'validation_rows': len(train)-cut, 'purge_gap': gap,
        'validation_history': selector.evals_result()['validation_0']['rmse']}
    return model


def recursive_predict(model, history, start_times, features, horizon=24):
    """Batch independent origins. Never access actual values after an origin."""
    history = np.array(history, dtype=float, copy=True)
    if history.ndim != 2 or history.shape[1] < 168 or horizon < 1:
        raise ValueError('Mỗi origin cần ít nhất 168 giờ quá khứ; horizon >= 1.')
    if len(history) != len(start_times) or not np.isfinite(history).all():
        raise ValueError('History không hợp lệ.')
    predictions = np.empty((len(history), horizon))
    for h in range(horizon):
        times = pd.DatetimeIndex(start_times) + pd.Timedelta(hours=h)
        row = calendar(times)
        for lag in LAGS:
            row[f'lag_{lag}'] = history[:, -lag]
        for window in (24, 168):
            row[f'rolling_mean_{window}'] = history[:, -window:].mean(axis=1)
            row[f'rolling_std_{window}'] = history[:, -window:].std(axis=1, ddof=1)
        predictions[:, h] = model.predict(row[features])
        history = np.column_stack([history[:, 1:], predictions[:, h]])
    return predictions


def errors(actual, predicted):
    residual = np.asarray(predicted) - np.asarray(actual)
    return float(np.abs(residual).mean()), float(np.sqrt(np.mean(residual ** 2)))


def step2_multistep_recursive_forecast(df, horizon=24, target_col='AEP_MW',
                                      model=None, test_start=None):
    """Daily rolling origins: observed history is available again at next origin."""
    validate_series(df, target_col)
    if model is None or test_start is None:
        raise ValueError('Cần model đã huấn luyện và test_start của CV01.')
    first = df.index.searchsorted(pd.Timestamp(test_start))
    if first < 168 or len(df) - first < horizon:
        raise ValueError('Không đủ lịch sử hoặc không đủ test cho horizon.')
    origins = np.arange(first, len(df) - horizon + 1, horizon)
    values = df[target_col].to_numpy()
    history = np.array([values[i-168:i] for i in origins])
    predicted = recursive_predict(model, history, df.index[origins], BASE_FEATURES + ROLL_FEATURES, horizon)
    actual = np.array([values[i:i+horizon] for i in origins])
    timestamps = np.array([df.index[i:i+horizon].to_numpy() for i in origins])
    predictions = pd.DataFrame({'origin': np.repeat(df.index[origins] - pd.Timedelta(hours=1), horizon),
                                'Datetime': timestamps.ravel(), 'horizon': np.tile(np.arange(1, horizon+1), len(origins)),
                                'actual': actual.ravel(), 'predicted': predicted.ravel()})
    rows = []
    for h in range(horizon):
        mae, rmse = errors(actual[:, h], predicted[:, h])
        cumulative_mae, cumulative_rmse = errors(actual[:, :h+1], predicted[:, :h+1])
        rows.append({'horizon': h+1, 'n_origins': len(origins), 'MAE': mae, 'RMSE': rmse,
                     'cumulative_MAE': cumulative_mae, 'cumulative_RMSE': cumulative_rmse})
    return predictions, pd.DataFrame(rows)


def direct_forecast(rolled, hourly, test_start, target_times):
    # At timestamp t, X_t uses only observations up to t-1. Model h learns y_(t+h-1).
    features = BASE_FEATURES + ROLL_FEATURES
    origins = pd.DatetimeIndex(target_times.origin.unique()) + pd.Timedelta(hours=1)
    result = target_times[['origin', 'Datetime', 'horizon', 'actual']].copy()
    forecasts = np.empty((len(origins), 24))
    for h in range(1, 25):
        labelled = rolled[features].copy()
        labelled['AEP_MW'] = hourly.AEP_MW.shift(-(h-1)).reindex(rolled.index)
        label_times = labelled.index + pd.Timedelta(hours=h-1)
        labelled = labelled.loc[label_times < test_start].dropna()
        print(f'Training Direct h={h}/24', flush=True)
        model = train_model(labelled, features, gap=h-1)
        forecasts[:, h-1] = model.predict(rolled.loc[origins, features])
    result['predicted'] = forecasts.ravel()
    return result


def horizon_metrics(predictions):
    rows = []
    for h in range(1, 25):
        step = predictions[predictions.horizon == h]
        pooled = predictions[predictions.horizon <= h]
        mae, rmse = errors(step.actual, step.predicted)
        cmae, crmse = errors(pooled.actual, pooled.predicted)
        rows.append(dict(horizon=h, n_origins=len(step), MAE=mae, RMSE=rmse,
                         cumulative_MAE=cmae, cumulative_RMSE=crmse))
    return pd.DataFrame(rows)


def plot_results(rolled, predictions, metrics, impact, direct, direct_metrics):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    sample = rolled.iloc[-336:]
    axes[0].plot(sample.index, sample.AEP_MW, label='Actual', color='#263b53', lw=1)
    for window, color in [(24, '#007f86'), (168, '#dd8734')]:
        mean = sample[f'rolling_mean_{window}']
        std = sample[f'rolling_std_{window}']
        axes[0].plot(sample.index, mean, label=f'Past {window}h mean', color=color)
        axes[0].fill_between(sample.index, (mean-std).to_numpy(), (mean+std).to_numpy(), color=color, alpha=.13)
    axes[0].set(title='Causal rolling features: mean ± 1 sample std (last 14 days)', ylabel='Load (MW)')
    axes[0].legend(ncol=3)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    axes[0].xaxis.set_major_locator(locator)
    axes[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    x = np.arange(len(impact))
    axes[1].bar(x-.18, impact.MAE, width=.36, label='MAE', color='#007f86')
    axes[1].bar(x+.18, impact.RMSE, width=.36, label='RMSE', color='#dd8734')
    axes[1].set(xticks=x, xticklabels=impact.model, ylabel='Error (MW)', title='One-step test: identical rows and training protocol')
    axes[1].legend()
    fig.savefig(FIGURES_DIR / 'cv04_rolling_features_impact.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    sample = predictions.iloc[:24]
    axes[0].plot(sample.Datetime, sample.actual, 'o-', label='Actual')
    axes[0].plot(sample.Datetime, sample.predicted, 'o-', label='Recursive forecast')
    axes[0].plot(direct.iloc[:24].Datetime, direct.iloc[:24].predicted, 'o-', label='Direct forecast')
    axes[0].set(title=f'First test origin: {sample.origin.iloc[0]} — next 24 hours', ylabel='Load (MW)')
    axes[0].legend()
    for column in ['MAE', 'RMSE']:
        axes[1].plot(metrics.horizon, metrics[column], 'o-', label=f'Recursive {column}')
        axes[1].plot(direct_metrics.horizon, direct_metrics[column], '--', label=f'Direct {column}')
        axes[2].plot(metrics.horizon, metrics[f'cumulative_{column}'], 'o-', label=f'Recursive {column}')
        axes[2].plot(direct_metrics.horizon, direct_metrics[f'cumulative_{column}'], '--', label=f'Direct {column}')
    axes[1].set(title='Error at each horizon across daily test origins', ylabel='Error (MW)')
    axes[2].set(title='Error pooled over steps 1 through h', ylabel='Error (MW)', xlabel='Forecast horizon h (hours)')
    for ax in axes:
        ax.grid(alpha=.2)
    for ax in axes[1:]:
        ax.set_xticks(range(1, 25))
        ax.legend()
    fig.savefig(FIGURES_DIR / 'cv04_multistep_forecast.png', dpi=150)
    plt.close(fig)


def main():
    for path in [FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_DATA, index_col=0, parse_dates=True)
    test = pd.read_csv(TEST_DATA, index_col=0, parse_dates=True)
    if train.index.max() >= test.index.min():
        raise ValueError('Train/test chồng lấn hoặc sai thứ tự.')
    source = pd.concat([train[['AEP_MW']], test[['AEP_MW']]]).sort_index()
    if source.index.has_duplicates:
        raise ValueError('CV01 phải xử lý timestamp trùng lặp trước.')
    # DST/missing hours: forward fill only from earlier observations, never interpolate.
    hourly = source.reindex(pd.date_range(source.index.min(), source.index.max(), freq='h', name='Datetime'))
    missing = int(hourly.AEP_MW.isna().sum())
    hourly = hourly.ffill()
    rolled, _ = step1_create_rolling_features(hourly)
    train_roll = rolled.loc[rolled.index < test.index.min()]
    # Score only observed test timestamps; inserted hours provide continuous history.
    test_roll = rolled.loc[rolled.index.isin(test.index)]
    impact_rows = []
    rolling_model = None
    for name, features in [('Lags + calendar', BASE_FEATURES), ('Lags + calendar + rolling', BASE_FEATURES + ROLL_FEATURES)]:
        print(f'Training {name}: {len(train_roll):,} train / {len(test_roll):,} test', flush=True)
        model = train_model(train_roll, features)
        mae, rmse = errors(test_roll.AEP_MW, model.predict(test_roll[features]))
        impact_rows.append({'model': name, 'MAE': mae, 'RMSE': rmse, 'n_test': len(test_roll), 'n_estimators': model.n_estimators})
        if features == BASE_FEATURES + ROLL_FEATURES:
            rolling_model = model
    rolling_model.save_model(MODELS_DIR / 'cv04_xgboost_rolling.json')
    predictions, metrics = step2_multistep_recursive_forecast(hourly, model=rolling_model, test_start=test.index.min())
    # Do not score imputed targets in recursive evaluation either.
    observed = source.index[source.AEP_MW.notna()]
    valid_origins = predictions.groupby('origin').Datetime.apply(lambda times: times.isin(observed).all())
    predictions = predictions[predictions.origin.isin(valid_origins[valid_origins].index)].copy()
    if predictions.empty:
        raise ValueError('Không có cửa sổ test 24 giờ đầy đủ quan sát thật.')
    metrics = horizon_metrics(predictions)
    direct = direct_forecast(rolled, hourly, test.index.min(), predictions)
    direct_metrics = horizon_metrics(direct)
    direct.to_csv(TABLES_DIR / 'cv04_direct_predictions.csv', index=False)
    direct_metrics.to_csv(TABLES_DIR / 'cv04_direct_metrics.csv', index=False)
    impact = pd.DataFrame(impact_rows)
    for frame, name in [(metrics, 'cv04_multistep_metrics.csv'), (predictions, 'cv04_multistep_predictions.csv'), (impact, 'cv04_rolling_features_impact.csv')]:
        frame.to_csv(TABLES_DIR / name, index=False)
    rolled.to_csv(DATA_PROCESSED_DIR / 'AEP_features_rolling.csv', index_label='Datetime')
    plot_results(rolled, predictions, metrics, impact, direct, direct_metrics)
    metadata = dict(train_rows=len(train_roll), test_rows=len(test_roll), test_start=str(test.index.min()),
                    hourly_rows=len(hourly), forward_filled_hours=missing, origins=int(metrics.n_origins.iloc[0]),
                    first_forecast=str(predictions.Datetime.min()), last_forecast=str(predictions.Datetime.max()),
                    origin_stride_hours=24, validation='last 10% of training; refit full training at best tree count',
                    target_imputation='causal forward fill; imputed targets excluded from scoring',
                    python=sys.version, xgboost=__import__('xgboost').__version__)
    (TABLES_DIR / 'cv04_run_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(impact.to_string(index=False))
    print(metrics.iloc[[0, 23]].to_string(index=False))
    print('CV04 complete.', flush=True)


if __name__ == '__main__':
    main()
