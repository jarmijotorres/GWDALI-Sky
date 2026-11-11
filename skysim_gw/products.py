"""Product filenames shared by stages and diagnostics."""
from pathlib import Path

def snr_output_path(output_dir, source_type, zi, zf):
    return Path(output_dir) / (
        f"skysim5000_{source_type}_lvk_snr_radec_deg_"
        f"z{zi:.4f}_{zf:.4f}.npz"
    )

def localization_output_path(output_dir, source_type, zi, zf):
    return Path(output_dir) / (
        f"skysim5000_{source_type}_lvk_fisher_mc_eta_inv_dL_"
        f"tc_fixed_radec_deg_localisation_"
        f"z{zi:.4f}_{zf:.4f}.npz"
    )

