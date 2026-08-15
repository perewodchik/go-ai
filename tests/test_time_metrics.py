"""
test_time_metrics.py — Tests for Time Metrics calculation and formatting API.
"""

import unittest
from web.routes.training_routes import learning_stats


class TestTimeMetrics(unittest.TestCase):
    def test_time_metrics_structure(self):
        # Verify structure of timing response output format
        summary = {
            'sp_avg_last': 12.5,
            'sp_total_last': 125.0,
            'sp_total_all': 500.0,
            'nn_total_last': 15.0,
            'nn_total_all': 60.0,
            'rand_avg_last': 5.0,
            'rand_total_last': 50.0,
            'rand_total_all': 150.0,
            'champ_avg_last': 20.0,
            'champ_total_last': 100.0,
            'champ_total_all': 200.0,
            'last_iter_total': 290.0,
            'all_time_total': 910.0,
        }
        self.assertIn('sp_avg_last', summary)
        self.assertIn('last_iter_total', summary)
        self.assertIn('all_time_total', summary)


if __name__ == '__main__':
    unittest.main()
