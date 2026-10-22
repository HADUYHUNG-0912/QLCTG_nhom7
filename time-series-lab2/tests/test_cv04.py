"""Regression checks for causal features, recursive feedback, and horizon alignment."""
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import cv04_rolling_and_multistep as cv
import cv04_detailed_analysis as analyst


class CV04Tests(unittest.TestCase):
    def test_current_and_future_targets_cannot_change_current_features(self):
        df = pd.DataFrame({'AEP_MW': np.arange(400, dtype=float)}, index=pd.date_range('2020-01-01', periods=400, freq='h'))
        original, _ = cv.step1_create_rolling_features(df)
        df.iloc[250:, 0] = 1e9
        changed, _ = cv.step1_create_rolling_features(df)
        features = cv.BASE_FEATURES + cv.ROLL_FEATURES
        pd.testing.assert_frame_equal(original.loc[:df.index[250], features], changed.loc[:df.index[250], features])
        self.assertEqual(original.loc[df.index[250], 'rolling_mean_24'], np.arange(226, 250).mean())
        self.assertAlmostEqual(original.loc[df.index[250], 'rolling_std_168'], np.arange(82, 250).std(ddof=1))

    def test_recursive_feedback_and_rolling_are_recomputed(self):
        class Spy:
            def __init__(self): self.calls = []
            def predict(self, frame):
                self.calls.append(frame.copy())
                return frame.lag_1.to_numpy() + 10
        model = Spy()
        history = np.arange(168, dtype=float)[None, :]
        start = pd.DatetimeIndex(['2020-01-01 23:00'])
        predicted = cv.recursive_predict(model, history, start, cv.BASE_FEATURES + cv.ROLL_FEATURES, horizon=24)
        np.testing.assert_allclose(predicted[0], 167 + 10*np.arange(1, 25))
        self.assertEqual(model.calls[1].hour.iloc[0], 0)
        self.assertEqual(model.calls[1].lag_1.iloc[0], 177)
        self.assertAlmostEqual(model.calls[1].rolling_mean_24.iloc[0], np.r_[np.arange(145, 168), 177].mean())
        np.testing.assert_array_equal(history[0], np.arange(168))

    def test_hourly_gaps_rejected(self):
        df = pd.DataFrame({'AEP_MW': [1., 2.]}, index=pd.to_datetime(['2020-01-01 00:00', '2020-01-01 02:00']))
        with self.assertRaises(ValueError): cv.step1_create_rolling_features(df)

    def test_metrics_use_all_origins_and_pooled_squared_errors(self):
        rows = [{'horizon': h, 'actual': 0., 'predicted': error} for h in range(1,25) for error in [h, 2*h]]
        metrics = cv.horizon_metrics(pd.DataFrame(rows))
        self.assertEqual(metrics.iloc[0].MAE, 1.5)
        self.assertAlmostEqual(metrics.iloc[0].RMSE, np.sqrt(2.5))
        self.assertAlmostEqual(metrics.iloc[1].cumulative_RMSE, np.sqrt((1+4+4+16)/4))


    def test_direct_labels_and_validation_gap(self):
        index = pd.date_range('2020-01-01', periods=450, freq='h')
        hourly = pd.DataFrame({'AEP_MW': np.arange(450, dtype=float)}, index=index)
        rolled, _ = cv.step1_create_rolling_features(hourly)
        start = index[400]
        rows = pd.DataFrame({'origin': [index[399]]*24,
                             'Datetime': index[400:424], 'horizon': range(1,25),
                             'actual': np.arange(400,424, dtype=float)})
        gaps = []
        def fake_train(training, features, target_col='AEP_MW', gap=0):
            gaps.append(gap)
            expected = hourly.AEP_MW.shift(-gap).reindex(training.index)
            np.testing.assert_array_equal(training.AEP_MW.to_numpy(), expected.to_numpy())
            self.assertTrue(((training.index+pd.Timedelta(hours=gap)) < start).all())
            class Model:
                def predict(self, features_frame):
                    return features_frame.lag_1.to_numpy()+1+gap
            return Model()
        with patch.object(cv, 'train_model', side_effect=fake_train):
            forecast = cv.direct_forecast(rolled, hourly, start, rows)
        self.assertEqual(gaps, list(range(24)))
        np.testing.assert_array_equal(forecast.predicted, forecast.actual)

    def test_block_bootstrap_preserves_pairing_and_constant_gain(self):
        reference = np.arange(1,31,dtype=float)+10
        candidate = reference-2
        result = analyst.paired_block_bootstrap(reference,candidate,block_length=7,repetitions=100)
        self.assertEqual(result['MAE_reduction_MW'],2)
        self.assertEqual(result['CI95_low_MW'],2)
        self.assertEqual(result['CI95_high_MW'],2)
        self.assertEqual(result['candidate_win_rate_pct'],100)
        self.assertEqual(result,analyst.paired_block_bootstrap(reference,candidate,block_length=7,repetitions=100))

    def test_metric_conventions_and_zero_actual(self):
        result=analyst.metrics([0,2],[1,1])
        self.assertEqual(result['MAE'],1)
        self.assertEqual(result['RMSE'],1)
        self.assertEqual(result['Bias_MW'],0)
        self.assertEqual(result['MAPE_pct'],50)
        self.assertEqual(result['WAPE_pct'],100)

    def test_saved_forecasts_and_metrics_agree(self):
        path=cv.TABLES_DIR/'cv04_multistep_predictions.csv'
        if not path.exists():
            self.skipTest('Run CV04 to create integration artifacts.')
        rec=pd.read_csv(path)
        direct=pd.read_csv(cv.TABLES_DIR/'cv04_direct_predictions.csv')
        self.assertTrue(rec[['origin','Datetime','horizon','actual']].equals(direct[['origin','Datetime','horizon','actual']]))
        self.assertTrue(rec.groupby('origin').horizon.nunique().eq(24).all())
        for frame,name in [(rec,'multistep'),(direct,'direct')]:
            computed=cv.horizon_metrics(frame)
            saved=pd.read_csv(cv.TABLES_DIR/f'cv04_{name}_metrics.csv')
            np.testing.assert_allclose(computed.select_dtypes('number'),saved.select_dtypes('number'),rtol=1e-10)


if __name__ == '__main__': unittest.main()
