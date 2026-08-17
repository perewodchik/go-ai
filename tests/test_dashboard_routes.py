"""
test_dashboard_routes.py — Tests for dashboard routing and Phase 7 cutover.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.app import create_app


class TestDashboardRoutes(unittest.TestCase):

    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_root_serves_fleet_console(self):
        """The primary landing page at / must serve the modern model fleet console."""
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn('fleet-layout', html)
        self.assertIn('models.js', html)
        self.assertIn('Go AI — Dashboard', html)

    def test_dashboard_old_serves_classic_dashboard(self):
        """The legacy dashboard is preserved at /dashboard_old."""
        res = self.client.get('/dashboard_old')
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn('model-selector', html)
        self.assertIn('dashboard.js', html)
        self.assertIn('Classic Dashboard', html)

    def test_models_redirects_to_root(self):
        """Visiting /models redirects to the primary dashboard at /."""
        res = self.client.get('/models')
        self.assertEqual(res.status_code, 302)
        self.assertEqual(res.headers.get('Location'), '/')

    def test_dashboard_new_redirects_to_root(self):
        """Visiting /dashboard_new permanently redirects to /."""
        res = self.client.get('/dashboard_new')
        self.assertEqual(res.status_code, 301)
        self.assertEqual(res.headers.get('Location'), '/')


if __name__ == '__main__':
    unittest.main()
