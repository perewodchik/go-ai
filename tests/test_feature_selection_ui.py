"""
test_feature_selection_ui.py — the encoding must be CHOOSABLE, not just supported.

Written after shipping `v2_12` with the registry, the config plumbing, the API
and the docs all in place — and no control in the create-model form. The backend
accepted `input_features` and the presets endpoint advertised the options, so
every backend test passed while the feature was unreachable from the UI.

These tests close that gap from both ends: the API round-trips the choice, and
the markup and scripts that let a person make it actually exist.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECT_ID = "new-model-input-features"

# Templates carrying the create-model form, and the script that drives each.
FORM_TEMPLATES = ("models.html", "dashboard_old.html", "index.html")
FORM_SCRIPTS = ("models.js", "dashboard.js")


def _read(*parts):
    with open(os.path.join(ROOT, *parts)) as fh:
        return fh.read()


class TestTheControlExists:
    @pytest.mark.parametrize("template", FORM_TEMPLATES)
    def test_every_create_form_has_the_selector(self, template):
        html = _read("web", "templates", template)
        if "new-model-net-size" not in html:
            pytest.skip(f"{template} has no create-model form")
        assert SELECT_ID in html, (
            f"{template} lets you pick a network size but not an input "
            f"encoding — the setting would be unreachable from this page"
        )

    @pytest.mark.parametrize("template", FORM_TEMPLATES)
    def test_the_selector_is_a_real_input(self, template):
        html = _read("web", "templates", template)
        if SELECT_ID not in html:
            pytest.skip(f"{template} has no create-model form")
        assert re.search(rf'<select[^>]*id="{SELECT_ID}"', html), \
            f"{template}: {SELECT_ID} must be a <select>"

    @pytest.mark.parametrize("template", FORM_TEMPLATES)
    def test_the_freeze_is_explained_in_the_form(self, template):
        html = _read("web", "templates", template)
        if SELECT_ID not in html:
            pytest.skip(f"{template} has no create-model form")
        assert "net-features-locked" in html, \
            f"{template}: no note explaining the setting is frozen after creation"


class TestTheControlIsWired:
    @pytest.mark.parametrize("script", FORM_SCRIPTS)
    def test_the_choice_is_sent_on_create(self, script):
        js = _read("web", "static", "js", script)
        if "api/create" not in js:
            pytest.skip(f"{script} does not create models")
        assert "input_features" in js, \
            f"{script}: the chosen encoding is never put in the create payload"

    @pytest.mark.parametrize("script", FORM_SCRIPTS)
    def test_the_choice_is_locked_after_creation(self, script):
        js = _read("web", "static", "js", script)
        if SELECT_ID not in js:
            pytest.skip(f"{script} does not drive the create form")
        assert "net-features-locked" in js, \
            f"{script}: the selector is never locked for edit/fork"

    @pytest.mark.parametrize("script", FORM_SCRIPTS)
    def test_the_options_are_populated_from_the_api(self, script):
        js = _read("web", "static", "js", script)
        if SELECT_ID not in js:
            pytest.skip(f"{script} does not drive the create form")
        assert "feature_sets" in js, \
            f"{script}: options are not read from /api/network_presets"


class TestPresetsEndpointAdvertisesThem:
    def test_feature_sets_are_offered_with_plane_counts(self):
        from web.app import create_app
        app = create_app()
        with app.test_client() as client:
            data = client.get('/models/api/network_presets?board_size=9').get_json()

        assert data['default_features'] == 'v1_10'
        keys = {f['key'] for f in data['feature_sets']}
        assert {'v1_10', 'v2_12'} <= keys
        by_key = {f['key']: f for f in data['feature_sets']}
        assert by_key['v1_10']['num_planes'] == 10
        assert by_key['v2_12']['num_planes'] == 12
        for spec in data['feature_sets']:
            assert spec['label'] and spec['summary']

    def test_parameter_counts_reflect_the_chosen_encoding(self):
        """A wider encoding widens the first convolution, so the price differs."""
        from web.app import create_app
        app = create_app()
        with app.test_client() as client:
            v1 = client.get('/models/api/network_presets'
                            '?board_size=9&input_features=v1_10').get_json()
            v2 = client.get('/models/api/network_presets'
                            '?board_size=9&input_features=v2_12').get_json()

        small_v1 = next(p for p in v1['presets'] if p['key'] == 'small')['params']
        small_v2 = next(p for p in v2['presets'] if p['key'] == 'small')['params']
        assert small_v2 > small_v1, "12 planes must cost more than 10"

    def test_params_delta_is_the_actual_difference_not_a_guess(self):
        """
        params_delta must equal params(chosen) - params(default), for real —
        computed by diffing two real networks, not a filters*planes*9 formula
        kept in sync by hand on both ends of the API.
        """
        from web.app import create_app
        app = create_app()
        with app.test_client() as client:
            v1 = client.get('/models/api/network_presets'
                            '?board_size=9&input_features=v1_10').get_json()
            v2 = client.get('/models/api/network_presets'
                            '?board_size=9&input_features=v2_12').get_json()

        by_key_v1 = {p['key']: p for p in v1['presets']}
        for preset in v2['presets']:
            base = by_key_v1[preset['key']]['params']
            assert preset['params_delta'] == preset['params'] - base, preset['key']

    def test_default_encoding_has_zero_delta(self):
        from web.app import create_app
        app = create_app()
        with app.test_client() as client:
            data = client.get('/models/api/network_presets?board_size=9').get_json()
        assert all(p['params_delta'] == 0 for p in data['presets'])

    def test_delta_scales_with_preset_filter_width(self):
        """Bigger presets have wider input convolutions, so a bigger delta."""
        from web.app import create_app
        app = create_app()
        with app.test_client() as client:
            data = client.get('/models/api/network_presets'
                              '?board_size=9&input_features=v2_12').get_json()

        by_key = {p['key']: p for p in data['presets']}
        assert by_key['tiny']['params_delta'] < by_key['small']['params_delta']
        assert by_key['small']['params_delta'] < by_key['large']['params_delta']

    def test_delta_is_exposed_to_the_ui_scripts(self):
        """The label must read params_delta from the server, not recompute it."""
        for script in FORM_SCRIPTS:
            js = _read("web", "static", "js", script)
            if SELECT_ID not in js:
                continue
            assert "params_delta" in js, \
                f"{script}: the params label does not surface the server's delta"


class TestCreateApiRoundTrip:
    def test_creating_with_v2_12_stores_it(self, tmp_path, monkeypatch):
        import model_manager
        monkeypatch.setattr(model_manager, "MODELS_ROOT", str(tmp_path))

        manager = model_manager.ModelManager()
        info = manager.create_model(
            name="probe",
            network_params={'size_preset': 'small', 'num_res_blocks': 4,
                            'num_filters': 64, 'value_head_hidden': 64,
                            'input_features': 'v2_12'})

        assert info.network.input_features == 'v2_12'

        with open(os.path.join(manager.get_model_dir(info.id), 'config.json')) as fh:
            assert json.load(fh)['network']['input_features'] == 'v2_12'

        # And it must survive a reload, then size the network correctly.
        from ai.network import GoNetwork
        from config import Config
        reloaded = model_manager.ModelManager().get_model(info.id)
        config = Config.from_model(reloaded, manager.get_model_dir(info.id))
        assert config.network.num_input_planes == 12
        net = GoNetwork(board_size=config.board.size,
                        input_features=config.network.input_features)
        assert net.input_conv.in_channels == 12

    def test_an_unknown_encoding_is_rejected(self):
        from web.routes.model_routes import _resolve_network_params
        params, error = _resolve_network_params({'input_features': 'v9_bogus'}, 9)
        assert params is None
        assert 'Invalid input features' in error

    def test_omitting_it_defaults_to_the_legacy_encoding(self):
        from web.routes.model_routes import _resolve_network_params
        params, error = _resolve_network_params({}, 9)
        assert error is None
        assert params['input_features'] == 'v1_10'
