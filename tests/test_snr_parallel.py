"""Unit tests for SNR worker argument and row-result handling."""
import unittest
from unittest.mock import patch

import numpy as np

from skysim_gw.stages import snr


class SnrParallelTests(unittest.TestCase):
    def test_worker_options_default_to_serial(self):
        arguments = snr.parse_arguments([])
        self.assertEqual(arguments.workers, 1)
        self.assertEqual(arguments.native_threads, 1)

    def test_worker_options_must_be_positive(self):
        with self.assertRaises(SystemExit):
            snr.parse_arguments(["--workers", "0"])
        with self.assertRaises(SystemExit):
            snr.parse_arguments(["--native-threads", "0"])

    def test_source_task_returns_tabular_success_payload(self):
        detector_snrs = np.array([3.0, 4.0, 5.0, 6.0])
        with patch.object(snr, "_compute_snr", return_value=(detector_snrs, 9.274)):
            result = snr._compute_snr_task(
                17, {"m1": 30.0}, [], "test", 10.0, 100.0, 1000,
            )
        self.assertEqual(result[0], 17)
        np.testing.assert_array_equal(result[1], detector_snrs)
        self.assertEqual(result[2], 9.274)
        self.assertEqual(result[3], "")

    def test_source_task_records_error_without_raising(self):
        with patch.object(snr, "_compute_snr", side_effect=RuntimeError("bad PSD")):
            result = snr._compute_snr_task(
                17, {"m1": 30.0}, [], "test", 10.0, 100.0, 1000,
            )
        self.assertEqual(result[0], 17)
        self.assertIsNone(result[1])
        self.assertTrue(np.isnan(result[2]))
        self.assertEqual(result[3], "RuntimeError: bad PSD")


if __name__ == "__main__":
    unittest.main()
