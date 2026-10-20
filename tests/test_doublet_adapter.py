"""Small dependency-free checks for tagged Doublet provenance helpers."""
import unittest

from skysim_gw.doublet_adapter import realization_tag


class DoubletAdapterTests(unittest.TestCase):
    def test_extracts_tagged_realization(self):
        self.assertEqual(
            realization_tag(
                "/data/skysim5000_BBH_gwdali_fh1_r0009_z0.0000_1.0000.npz"
            ),
            "fh1_r0009",
        )

    def test_uses_stem_when_no_realization_tag_exists(self):
        self.assertEqual(
            realization_tag("/data/skysim5000_BBH_gwdali_z0.0000_1.0000.npz"),
            "skysim5000_BBH_gwdali_z0.0000_1.0000",
        )


if __name__ == "__main__":
    unittest.main()
