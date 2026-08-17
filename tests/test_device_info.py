"""
test_device_info.py — Hardware detection, worker bounds, and the device the
trainer actually resolves onto.

The behaviour worth pinning here is not "does this machine have a GPU" — that
is not a property of the code — but the FALLBACKS: a torch that will not
import, a CUDA device that claims to exist and then fails, and a worker ceiling
that must track the host without ever dropping below the bound this project
shipped with.
"""

import os
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import device_info


class TestWorkerBounds(unittest.TestCase):

    def test_ceiling_tracks_the_host(self):
        with patch.object(device_info, 'logical_cores', return_value=20):
            self.assertEqual(device_info.worker_ceiling(), 20)
            # Two cores held back for the server and trainer threads.
            self.assertEqual(device_info.recommended_workers(), 18)

    def test_ceiling_never_drops_below_the_historical_maximum(self):
        # A 2-core laptop still offers the 8 this project has always offered:
        # the pools clamp to the core count at run time anyway, and a tighter
        # slider would push a stored value of 8 out of range.
        with patch.object(device_info, 'logical_cores', return_value=2):
            self.assertEqual(device_info.worker_ceiling(),
                             device_info.MIN_WORKER_CEILING)
            self.assertEqual(device_info.recommended_workers(), 1)

    def test_ceiling_is_capped(self):
        with patch.object(device_info, 'logical_cores', return_value=512):
            self.assertEqual(device_info.worker_ceiling(),
                             device_info.MAX_WORKER_CEILING)
            self.assertEqual(device_info.recommended_workers(),
                             device_info.MAX_WORKER_CEILING)

    def test_recommendation_is_at_least_one(self):
        with patch.object(device_info, 'logical_cores', return_value=1):
            self.assertGreaterEqual(device_info.recommended_workers(), 1)

    def test_param_bounds_offers_the_hosts_cores(self):
        from param_bounds import PARAM_BOUNDS, sanitize_params
        spec = PARAM_BOUNDS['num_parallel_workers']
        self.assertEqual(spec['max'], device_info.worker_ceiling())
        self.assertGreaterEqual(spec['max'], 8)
        self.assertLessEqual(spec['default'], spec['max'])
        # And the clamp agrees with the slider, so an API caller cannot store a
        # value the UI would refuse to produce.
        clamped = sanitize_params({'num_parallel_workers': 9999})
        self.assertEqual(clamped['num_parallel_workers'], spec['max'])


class TestDetection(unittest.TestCase):

    def _fake_torch(self, cuda=False, mps=False, cuda_build='12.1'):
        torch = types.ModuleType('torch')
        torch.__version__ = '2.4.0'
        torch.version = types.SimpleNamespace(cuda=cuda_build)

        props = types.SimpleNamespace(
            name='NVIDIA GeForce RTX 3070 Ti Laptop GPU',
            total_memory=8 * 1024 ** 3, major=8, minor=6,
            multi_processor_count=46,
        )
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: cuda,
            current_device=lambda: 0,
            device_count=lambda: 1,
            get_device_properties=lambda idx: props,
        )
        torch.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps))
        return torch

    def _detect_with(self, torch_module):
        with patch.dict(sys.modules, {'torch': torch_module}):
            return device_info.detect(refresh=True)

    def tearDown(self):
        # The module caches its answer; leave it holding the real machine's.
        device_info.detect(refresh=True)

    def test_cuda_is_described_in_full(self):
        info = self._detect_with(self._fake_torch(cuda=True))
        self.assertEqual(info.kind, 'cuda')
        self.assertEqual(info.torch_device, 'cuda')
        self.assertTrue(info.is_gpu)
        self.assertEqual(info.total_memory_gb, 8.0)
        self.assertEqual(info.capability, '8.6')
        self.assertIn('RTX 3070 Ti', info.detail)

    def test_cuda_wins_over_mps_when_both_claim_to_exist(self):
        info = self._detect_with(self._fake_torch(cuda=True, mps=True))
        self.assertEqual(info.kind, 'cuda')

    def test_mps_is_used_when_there_is_no_cuda(self):
        info = self._detect_with(self._fake_torch(cuda=False, mps=True))
        self.assertEqual(info.kind, 'mps')
        self.assertTrue(info.is_gpu)

    def test_cpu_only_build_says_so(self):
        # The single most common "why is my GPU idle" case on Windows: an
        # NVIDIA card and a torch wheel with no CUDA in it.
        info = self._detect_with(self._fake_torch(cuda=False, cuda_build=None))
        self.assertEqual(info.kind, 'cpu')
        self.assertIn('CPU-only', info.note)

    def test_driver_problem_is_named_separately(self):
        info = self._detect_with(self._fake_torch(cuda=False, cuda_build='12.1'))
        self.assertEqual(info.kind, 'cpu')
        self.assertIn('driver', info.note)

    def test_a_raising_mps_probe_does_not_take_detection_down(self):
        torch = self._fake_torch(cuda=False, cuda_build=None)

        def boom():
            raise RuntimeError('MPS backend is not built')

        torch.backends.mps.is_available = boom
        info = self._detect_with(torch)
        self.assertEqual(info.kind, 'cpu')

    def test_missing_torch_degrades_to_cpu(self):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == 'torch':
                raise ImportError('No module named torch')
            return real_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=fake_import):
            info = device_info.detect(refresh=True)
        self.assertEqual(info.kind, 'cpu')
        self.assertFalse(info.torch_available)
        self.assertIn('PyTorch', info.note)


class TestSummary(unittest.TestCase):

    def test_summary_has_what_the_ui_needs(self):
        s = device_info.summary()
        self.assertIn('device', s)
        self.assertIn('cpu', s)
        self.assertEqual(s['cpu']['worker_ceiling'], device_info.worker_ceiling())
        # The one sentence every surface reuses instead of inventing its own.
        self.assertIn('explanation', s['roles'])
        self.assertNotIn('benchmark', s)   # opt-in; it costs real compute

    def test_benchmark_is_skipped_without_a_gpu(self):
        with patch.object(device_info, 'detect',
                          return_value=device_info._cpu_info()):
            out = device_info.benchmark()
        self.assertEqual(out['verdict'], 'no_gpu')
        self.assertIsNone(out['gpu_ms'])


if __name__ == '__main__':
    unittest.main()
