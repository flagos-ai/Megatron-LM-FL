"""Tests for KunLunXin debug logging helpers."""

import os
import unittest

from megatron.plugin.kunlunxin.debug import debug_enabled, debug_patch, log_patch


class TestKunLunXinDebug(unittest.TestCase):
    """Validate the KunLunXin patch debug logging switch."""

    def tearDown(self):
        """Clear debug environment variables after each test."""
        os.environ.pop("MG_FL_KUNLUNXIN_DEBUG", None)

    def test_debug_enabled_parses_truthy_values(self):
        """Accept the supported truthy values for the debug switch."""
        for value in ("1", "true", "TRUE", "yes", "on"):
            os.environ["MG_FL_KUNLUNXIN_DEBUG"] = value
            self.assertTrue(debug_enabled())

    def test_log_patch_is_silent_by_default(self):
        """Do not emit KunLunXin patch logs unless the debug switch is enabled."""
        os.environ.pop("MG_FL_KUNLUNXIN_DEBUG", None)
        with self.assertNoLogs("megatron.plugin.kunlunxin.debug", level="WARNING"):
            log_patch("unit.test")

    def test_debug_patch_logs_when_enabled(self):
        """Log patch entry and preserve the wrapped function result."""
        os.environ["MG_FL_KUNLUNXIN_DEBUG"] = "1"

        @debug_patch("unit.test.patch")
        def patched(value):
            """Return a deterministic value for wrapper validation."""
            return value + 1

        with self.assertLogs("megatron.plugin.kunlunxin.debug", level="WARNING") as captured:
            self.assertEqual(patched(1), 2)

        self.assertTrue(any("[KunLunXin Override] unit.test.patch" in line for line in captured.output))


if __name__ == "__main__":
    unittest.main(verbosity=2)
