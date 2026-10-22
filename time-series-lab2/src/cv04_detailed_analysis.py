# -*- coding: utf-8 -*-
"""Reproducible CV04 diagnostics; run after cv04_rolling_and_multistep.py.

Ablations share rows and training protocol. All bootstrap comparisons are paired
and use blocks of ordered origins. Analysis does not tune models on test results.
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cv04_rolling_and_multistep as cv

ROOT = cv.PROJECT_ROOT
TABLES = cv.TABLES_DIR
FIGURES = cv.FIGURES_DIR


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.size == 0 or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Metrics require nonempty, finite arrays.')
    error = predicted - actual
    absolute = np.abs(error)
    nonzero = np.abs(actual) > 1e-12
    sst = np.sum((actual - actual.mean())**2)
    return {'n': actual.size, 'MAE': absolute.mean(), 'RMSE': np.sqrt(np.mean(error**2)),
            'MAPE_pct': np.mean(absolute[nonzero] / np.abs(actual[nonzero]))*100 if nonzero.any() else np.nan,
            'WAPE_pct': absolute.sum()/np.abs(actual).sum()*100 if np.abs(actual).sum() else np.nan,
            'R2': 1-np.sum(error**2)/sst if sst > 0 else np.nan,
            'Bias_MW': error.mean(), 'Median_AE': np.median(absolute),
            'P90_AE': np.quantile(absolute, .90), 'P95_AE': np.quantile(absolute, .95),
            'Max_AE': absolute.max()}


def paired_block_bootstrap(reference, candidate, block_length=7, repetitions=2000, seed=42):
    """Circular blocks of daily origins; positive delta means candidate better.

Inputs are mean absolute errors within each origin. The uncertainty concerns the
observed fitted forecasts, not retraining variability or calibrated prediction CIs.
"""
    reference, candidate = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape or reference.ndim != 1 or len(reference) < 2:
        raise ValueError('Bootstrap requires aligned one-dimensional origin losses.')
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError('Losses must be finite.')
    if block_length < 1 or repetitions < 1:
        raise ValueError('Positive block length and repetition count required.')
    rng = np.random.default_rng(seed)
    n = len(reference)
    starts = rng.integers(0, n, size=(repetitions, int(np.ceil(n/block_length))))
    indices = ((starts[..., None] + np.arange(block_length)) % n).reshape(repetitions, -1)[:, :n]
    delta = (reference[indices]-candidate[indices]).mean(axis=1)
    return {'block_length': block_length, 'repetitions': repetitions,
            'MAE_reduction_MW': float((reference-candidate).mean()),
            'CI95_low_MW': float(np.quantile(delta, .025)),
            'CI95_high_MW': float(np.quantile(delta, .975)),
            'candidate_win_rate_pct': float(np.mean(candidate < reference)*100)}


def grouped_metrics(frame, column):
    rows = []
    for (strategy, group), subset in frame.groupby(['strategy', column], observed=True):
        rows.append({'strategy': strategy, column: str(group), **metrics(subset.actual, subset.predicted)})
    return pd.DataFrame(rows)


def save(frame, name):
    frame.to_csv(TABLES / f'cv04_{name}.csv', index=False)
    return frame


def load_source():
    train = pd.read_csv(cv.TRAIN_DATA, index_col=0, parse_dates=True)
    test = pd.read_csv(cv.TEST_DATA, index_col=0, parse_dates=True)
    source = pd.concat([train[['AEP_MW']], test[['AEP_MW']]]).sort_index()
    hourly = source.reindex(pd.date_range(source.index.min(), source.index.max(), freq='h', name='Datetime'))
    missing = hourly[hourly.AEP_MW.isna()].reset_index()[['Datetime']]
    missing['partition'] = np.where(missing.Datetime < test.index.min(), 'train', 'test')
    save(missing, 'missing_hours_audit')
    return train, test, hourly.ffill()


def ablation_analysis(rolled, test_start, test_index):
    train = rolled.loc[rolled.index < test_start]
    test = rolled.loc[rolled.index.isin(test_index)]
    variants = {'lags_calendar': cv.BASE_FEATURES,
                'plus_rolling_24': cv.BASE_FEATURES + ['rolling_mean_24', 'rolling_std_24'],
                'plus_rolling_168': cv.BASE_FEATURES + ['rolling_mean_168', 'rolling_std_168'],
                'plus_all_rolling': cv.BASE_FEATURES + cv.ROLL_FEATURES}
    summaries, forecasts, histories = [], [], []
    model_all = None
    for name, features in variants.items():
        print(f'CV04 ablation: {name}', flush=True)
        begin = time.perf_counter()
        model = cv.train_model(train, features)
        fit_seconds = time.perf_counter()-begin
        begin = time.perf_counter()
        predicted = model.predict(test[features])
        predict_seconds = time.perf_counter()-begin
        info = model.cv04_training_info
        summaries.append({'variant': name, 'n_features': len(features), 'fit_seconds': fit_seconds,
                          'predict_seconds': predict_seconds,
                          'best_n_estimators': info['best_n_estimators'],
                          'hit_tree_budget': info['hit_tree_budget'],
                          'validation_rmse': info['validation_rmse'],
                          **metrics(test.AEP_MW, predicted)})
        forecasts.append(pd.DataFrame({'Datetime': test.index, 'variant': name,
                                      'actual': test.AEP_MW.to_numpy(), 'predicted': predicted}))
        histories.extend({'variant': name, 'iteration': i+1, 'validation_RMSE': value}
                         for i, value in enumerate(info['validation_history']))
        if name == 'plus_all_rolling':
            model_all = model
    summary = save(pd.DataFrame(summaries), 'ablation_summary')
    one_step = save(pd.concat(forecasts, ignore_index=True), 'one_step_predictions')
    save(pd.DataFrame(histories), 'ablation_validation_history')
    importance = pd.DataFrame({'feature': cv.BASE_FEATURES + cv.ROLL_FEATURES,
                               'normalized_gain': model_all.feature_importances_})
    importance['group'] = importance.feature.map(lambda x: 'rolling' if x.startswith('rolling') else 'lags' if x.startswith('lag_') else 'calendar')
    save(importance.sort_values('normalized_gain', ascending=False), 'rolling_model_importance')
    corr = train[cv.ROLL_FEATURES + ['lag_1', 'lag_24', 'lag_168', 'AEP_MW']].corr()
    corr.to_csv(TABLES / 'cv04_train_feature_correlations.csv')
    one_step['day'] = one_step.Datetime.dt.normalize()
    one_step['AE'] = (one_step.predicted-one_step.actual).abs()
    day_loss = one_step.groupby(['day', 'variant']).AE.mean().unstack()
    cis = []
    for variant in list(variants)[1:]:
        for block in (1, 7, 14):
            cis.append({'candidate': variant,
                        **paired_block_bootstrap(day_loss.lags_calendar.to_numpy(), day_loss[variant].to_numpy(), block_length=block)})
    save(pd.DataFrame(cis), 'rolling_bootstrap')
    return summary, one_step, importance


def forecast_analysis(hourly, training_target):
    recursive = pd.read_csv(TABLES/'cv04_multistep_predictions.csv', parse_dates=['origin', 'Datetime'])
    direct = pd.read_csv(TABLES/'cv04_direct_predictions.csv', parse_dates=['origin', 'Datetime'])
    keys = ['origin', 'Datetime', 'horizon', 'actual']
    if not recursive[keys].equals(direct[keys]):
        raise ValueError('Recursive/Direct forecasts do not share evaluation targets.')
    frames = [recursive.assign(strategy='Recursive'), direct.assign(strategy='Direct')]
    for name, lag in [('SeasonalNaive24', 24), ('SeasonalNaive168', 168)]:
        baseline = recursive[keys].copy()
        timestamps = pd.DatetimeIndex(baseline.Datetime)-pd.Timedelta(hours=lag)
        assert (timestamps.to_numpy() <= baseline.origin.to_numpy()).all()
        baseline['predicted'] = hourly.AEP_MW.reindex(timestamps).to_numpy()
        baseline['strategy'] = name
        frames.append(baseline)
    frame = pd.concat(frames, ignore_index=True)
    frame['error'] = frame.predicted-frame.actual
    frame['AE'] = frame.error.abs()
    frame['year'] = frame.Datetime.dt.year
    frame['month'] = frame.Datetime.dt.month
    frame['hour'] = frame.Datetime.dt.hour
    frame['day_type'] = np.where(frame.Datetime.dt.dayofweek >= 5, 'weekend', 'weekday')
    q25, q75 = training_target.quantile([.25, .75])
    frame['load_regime'] = pd.cut(frame.actual, [-np.inf, q25, q75, np.inf], labels=['low', 'middle', 'high'])
    frame['horizon_band'] = pd.cut(frame.horizon, [0,6,12,18,24], labels=['01-06','07-12','13-18','19-24'])
    overall = save(grouped_metrics(frame.assign(scope='all'), 'scope'), 'strategy_summary')
    for dimension in ['horizon', 'horizon_band', 'year', 'month', 'hour', 'day_type', 'load_regime']:
        save(grouped_metrics(frame, dimension), f'error_by_{dimension}')
    daily = frame.groupby(['origin', 'strategy']).AE.mean().unstack().sort_index()
    save(daily.reset_index(), 'origin_mae')
    cis = []
    for reference, candidate in [('Recursive','Direct'), ('SeasonalNaive24','Recursive'),
                                  ('SeasonalNaive24','Direct'), ('SeasonalNaive168','Direct')]:
        for block in (1,7,14):
            cis.append({'reference': reference, 'candidate': candidate,
                        **paired_block_bootstrap(daily[reference], daily[candidate], block_length=block)})
    save(pd.DataFrame(cis), 'strategy_bootstrap')
    save(frame.sort_values('AE', ascending=False).groupby('strategy', sort=False).head(10), 'largest_errors')
    save(frame.groupby(['strategy','origin']).agg(MAE=('AE','mean'), Bias_MW=('error','mean')).reset_index(), 'origin_diagnostics')
    audit = {'train_load_q25': float(q25), 'train_load_q75': float(q75),
             'origin_hours': sorted(int(v) for v in frame.origin.dt.hour.unique()),
             'horizon_to_target_hour': {str(h): sorted(int(v) for v in g.hour.unique()) for h,g in frame.groupby('horizon')},
             'residual_definition': 'predicted - actual',
             'daily_mae_lag1_autocorrelation': {col: float(daily[col].autocorr()) for col in daily},
             'bootstrap_seed': 42, 'bootstrap_repetitions': 2000,
             'rows_per_strategy': int(len(recursive)), 'origins': int(len(daily)),
             'limitations': ['one chronological holdout', 'fixed origin hour confounds horizon and target hour',
                             'no model retraining uncertainty in bootstrap', 'no weather or holiday variables']}
    (TABLES/'cv04_analysis_metadata.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return frame, overall, pd.DataFrame(cis), audit


def create_figures(frame, ablations, importance):
    colors = {'Recursive':'#007f86', 'Direct':'#d97928', 'SeasonalNaive24':'#65788c', 'SeasonalNaive168':'#a18bb9'}
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2,2,figsize=(14,10), constrained_layout=True)
    for name,g in frame.groupby('strategy'):
        errors = g.groupby('horizon').AE.mean()
        axes[0,0].plot(errors.index, errors, label=name, color=colors[name])
        by_month = g.groupby('month').AE.mean()
        axes[0,1].plot(by_month.index, by_month, 'o-', label=name, color=colors[name])
    axes[0,0].set(title='Horizon MAE vs seasonal-naive benchmarks', xlabel='Horizon (hours)', ylabel='MAE (MW)')
    axes[0,1].set(title='Calendar-month errors (unequal coverage)', xlabel='Target month', ylabel='MAE (MW)', xticks=range(1,13))
    order = ['Recursive','Direct','SeasonalNaive24','SeasonalNaive168']
    axes[1,0].boxplot([frame.loc[frame.strategy == s,'error'].to_numpy() for s in order], showfliers=False)
    axes[1,0].set_xticks(np.arange(1, len(order)+1), order)
    axes[1,0].axhline(0, color='#333', linestyle='--')
    axes[1,0].set(title='Signed residual distributions (outliers hidden)', ylabel='Predicted − actual (MW)')
    rec = frame[frame.strategy == 'Recursive'].groupby('origin').AE.mean()
    direct = frame[frame.strategy == 'Direct'].groupby('origin').AE.mean()
    delta = rec-direct
    axes[1,1].hist(delta, bins=45, color='#007f86', alpha=.85)
    axes[1,1].axvline(0, color='#333', linestyle='--')
    axes[1,1].set(title='Daily origin gain: positive favors Direct', xlabel='Recursive MAE − Direct MAE (MW)', ylabel='Number of origins')
    for ax in axes.flat: ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8)
    axes[0,1].legend(fontsize=8)
    fig.savefig(FIGURES/'cv04_diagnostic_dashboard.png',dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(14,5), constrained_layout=True)
    labels = ['Lags + calendar','+ Rolling 24h','+ Rolling 168h','+ Both windows']
    axes[0].barh(labels, ablations.MAE, color=['#8295a5','#5aada4','#dea167','#007f86'])
    for i,value in enumerate(ablations.MAE): axes[0].text(value+1,i,f'{value:.2f}',va='center')
    axes[0].set(xlim=(0,ablations.MAE.max()*1.18), xlabel='One-step MAE (MW)', title='Rolling-window ablation; identical test rows')
    top = importance.sort_values('normalized_gain').tail(12)
    axes[1].barh(top.feature, top.normalized_gain, color=['#d97928' if f.startswith('rolling') else '#007f86' for f in top.feature])
    axes[1].set(title='Top 12 normalized mean-gain importances', xlabel='Share of summed feature mean gains')
    fig.savefig(FIGURES/'cv04_ablation_importance.png',dpi=150)
    plt.close(fig)
    # Representative first, median-error and worst Recursive windows selected deterministically.
    rec = frame[frame.strategy == 'Recursive'].groupby('origin').AE.mean().sort_values()
    origins = [frame.origin.min(), rec.index[len(rec)//2], rec.index[-1]]
    fig,axes = plt.subplots(3,1,figsize=(13,11),constrained_layout=True)
    for ax,origin,label in zip(axes,origins,['First valid origin','Median Recursive MAE origin','Worst Recursive MAE origin']):
        subset = frame[frame.origin == origin]
        actual = subset[subset.strategy == 'Recursive']
        ax.plot(actual.horizon,actual.actual,'o-',color='#263b53',label='Actual')
        for strategy in ['Recursive','Direct','SeasonalNaive24']:
            group = subset[subset.strategy == strategy]
            ax.plot(group.horizon,group.predicted,label=strategy,color=colors[strategy])
        ax.set(title=f'{label}: {origin}',xlabel='Horizon (hours)',ylabel='Load (MW)')
        ax.legend(ncol=4,fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(FIGURES/'cv04_case_studies.png',dpi=150)
    plt.close(fig)
    # Holdout errors are diagnostics, not prediction intervals.
    pivot = frame[frame.strategy == 'Recursive'].pivot_table(index='month',columns='horizon',values='AE',aggfunc='mean')
    fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
    im=ax.imshow(pivot.to_numpy(),aspect='auto',cmap='YlOrRd',origin='lower')
    ax.set(xticks=range(24),xticklabels=range(1,25),yticks=range(len(pivot)),yticklabels=pivot.index,
           xlabel='Horizon (hours)',ylabel='Target calendar month',title='Recursive MAE by month and horizon (MW; unequal monthly coverage)')
    fig.colorbar(im,ax=ax,label='MAE (MW)')
    fig.savefig(FIGURES/'cv04_month_horizon_heatmap.png',dpi=150)
    plt.close(fig)


def main():
    _, test, hourly = load_source()
    rolled, _ = cv.step1_create_rolling_features(hourly)
    ablations, _, importance = ablation_analysis(rolled, test.index.min(), test.index)
    training_target = hourly.loc[hourly.index < test.index.min(),'AEP_MW']
    frame, summary, bootstrap, audit = forecast_analysis(hourly, training_target)
    create_figures(frame, ablations, importance)
    print(summary[['strategy','MAE','RMSE','MAPE_pct','Bias_MW']].to_string(index=False))
    print(bootstrap[bootstrap.block_length == 7].to_string(index=False))
    print('CV04 detailed analysis complete.',flush=True)


if __name__ == '__main__':
    main()
