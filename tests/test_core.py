"""
Verification tests for the VDA-HyperQuant research library.

Run: pytest tests/test_core.py -x -q
     (or: python tests/test_core.py for the pretty-printed CLI report)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def test_hadamard_orthogonality():
    """Verify that H @ H^T = d * I (Hadamard orthogonality)."""
    from research.transforms.hadamard import build_hadamard_matrix
    for d in [4, 8, 16, 64, 256]:
        H = build_hadamard_matrix(d)
        product = H @ H.T
        expected = d * torch.eye(d)
        error = (product - expected).abs().max().item()
        status = "OK" if error < 1e-4 else "FAIL"
        print(f"  [{status}] Hadamard orthogonality d={d}: max error = {error:.2e}")
        assert error < 1e-4, f"Hadamard orthogonality violated at d={d}: max error {error:.2e}"


def test_fht_matches_matrix():
    """Verify that Fast Hadamard Transform matches matrix multiplication."""
    from research.transforms.hadamard import build_hadamard_matrix, fast_hadamard_transform
    d = 64
    H = build_hadamard_matrix(d).float()
    x = torch.randn(8, d)  # batch of 8 vectors

    # Matrix multiply (ground truth)
    y_mat = (x @ H.T) / (d ** 0.5)

    # Fast butterfly transform
    y_fht = fast_hadamard_transform(x, normalize=True)

    error = (y_mat - y_fht).abs().max().item()
    status = "OK" if error < 1e-4 else "FAIL"
    print(f"  [{status}] FHT matches matrix multiply (d={d}): max error = {error:.2e}")
    assert error < 1e-4, f"FHT does not match matrix multiply: max error {error:.2e}"


def test_fht_does_not_mutate_input():
    """Verify fast_hadamard_transform never mutates its input tensor in place."""
    from research.transforms.hadamard import fast_hadamard_transform
    for d in [64, 384]:
        x = torch.randn(4, 6, d)
        x_before = x.clone()
        _ = fast_hadamard_transform(x, normalize=True)
        mutated = not torch.equal(x, x_before)
        status = "FAIL" if mutated else "OK"
        print(f"  [{status}] FHT purity (d={d}): input mutated = {mutated}")
        assert not mutated, f"fast_hadamard_transform mutated its input tensor at d={d}"


def test_fht_fp16_precision():
    """Verify FHT upcasts fp16 internally and round-trips within fp16 tolerance."""
    from research.transforms.hadamard import fast_hadamard_transform
    d = 64
    x = torch.randn(8, d, dtype=torch.float16)
    y = fast_hadamard_transform(x, normalize=True)
    assert y.dtype == torch.float16, f"FHT should return fp16 for fp16 input, got {y.dtype}"

    # Round-trip: FHT is its own inverse (up to normalization it already handles)
    x_back = fast_hadamard_transform(y, normalize=True)
    error = (x.float() - x_back.float()).abs().max().item()
    status = "OK" if error < 2e-2 else "FAIL"
    print(f"  [{status}] FHT fp16 round-trip (d={d}): max error = {error:.2e}")
    assert error < 2e-2, f"FHT fp16 round-trip error too large: {error:.2e}"


def test_rht_invertibility():
    """Verify that RHT is perfectly invertible: x == inv(RHT(x))."""
    from research.transforms.hadamard import HadamardRotation

    # Test 1: Power-of-2 dimension (like ViT-Small head_dim=64) — exact match
    rot64 = HadamardRotation(dim=64)
    x = torch.randn(2, 6, 50, 64)  # (B, heads, tokens, head_dim)
    x_rot = rot64(x)
    x_back = rot64.inverse(x_rot)
    error = (x - x_back).abs().max().item()
    status = "OK" if error < 1e-4 else "FAIL"
    print(f"  [{status}] RHT invertibility (64-dim, power-of-2): max error = {error:.2e}")
    assert error < 1e-4, f"RHT round-trip failed at dim=64: max error {error:.2e}"

    # Test 2: Non-power-of-2 (384) — output is padded to 512, inverse recovers 384
    rot384 = HadamardRotation(dim=384)
    x2 = torch.randn(2, 6, 50, 384)
    x2_rot = rot384(x2)  # shape: (..., 512)
    x2_back = rot384.inverse(x2_rot)  # shape: (..., 384)
    error2 = (x2 - x2_back).abs().max().item()
    status2 = "OK" if error2 < 1e-4 else "FAIL"
    print(f"  [{status2}] RHT invertibility (384->512->384): max error = {error2:.2e}")
    assert error2 < 1e-4, f"RHT round-trip failed at dim=384: max error {error2:.2e}"


def test_rht_outlier_suppression():
    """
    KEY EXPERIMENT: Verify that RHT suppresses outliers.
    Simulate the DinoV2 outlier problem: 2 channels have 100x larger values.
    After RHT, the max/mean ratio should be dramatically reduced.
    """
    from research.transforms.hadamard import HadamardRotation
    d = 64
    rot = HadamardRotation(dim=d)

    # Create input with extreme outlier channels (like DinoV2 LayerNorm)
    x = torch.randn(100, d) * 0.1  # small baseline
    x[:, 0] = 50.0   # Outlier channel 1
    x[:, 1] = -30.0  # Outlier channel 2

    before_ratio = x.abs().max().item() / x.abs().mean().item()

    x_rot = rot(x)
    after_ratio = x_rot.abs().max().item() / x_rot.abs().mean().item()

    reduction = before_ratio / after_ratio
    status = "OK" if reduction > 3.0 else "FAIL"
    print(f"  [{status}] Outlier suppression: before max/mean = {before_ratio:.1f}, "
          f"after = {after_ratio:.1f}, reduction = {reduction:.1f}x")
    assert reduction > 3.0, f"RHT outlier reduction insufficient: {reduction:.1f}x (need >3.0x)"


def test_d4_lattice_parity():
    """Verify that D4 quantizer always produces even-sum integer coordinates,
    and that every coordinate stays within the representable range, even for
    inputs that push the scaled magnitude near/over the clamp boundary."""
    from research.quantizers.lattice_vq import LatticeD4Quantizer
    q = LatticeD4Quantizer(bits=4)
    half_levels = q.half_levels

    torch.manual_seed(0)
    x_normal = torch.randn(1000, 64)
    x_extreme = torch.randn(1000, 64)
    x_extreme[:, ::4] *= 5.0  # push some group maxima to scale up the whole group

    for name, x in [("normal", x_normal), ("extreme", x_extreme)]:
        x_q, info = q(x)
        scale = info['scale'].unsqueeze(-1)  # (..., d//4, 1)
        x_q_grouped = x_q.reshape(x.shape[0], 16, 4)
        z_int = (x_q_grouped / scale).round()

        sums = z_int.sum(dim=-1)
        even_frac = (sums.long() % 2 == 0).float().mean().item()
        in_range = ((z_int >= -half_levels) & (z_int <= half_levels - 1)).all().item()

        status = "OK" if (even_frac == 1.0 and in_range) else "FAIL"
        print(f"  [{status}] D4 lattice parity ({name}): even_sum_frac={even_frac:.4f}, "
              f"in_range={in_range}, output shape {x_q.shape}, method={info['method']}")
        assert even_frac == 1.0, f"D4 lattice produced odd-parity points ({name}): {even_frac:.4f} even"
        assert in_range, f"D4 lattice produced out-of-range points ({name})"

    # Effective-bits accounting must not claim an unearned savings without index coding.
    assert info['bits'] == q.bits, "D4 info['bits'] must equal nominal bits (no unearned -0.25 claim)"
    assert 'effective_bits_per_scalar' not in info or info['effective_bits_per_scalar'] == info['bits'], \
        "D4 quantizer must not report effective_bits_per_scalar below nominal bits without index coding"


def test_d4_boundary_edge_case():
    """
    Targeted regression test for the pre-existing clamp-after-decode bug:
    a coordinate whose scaled value sits exactly at the round-to-boundary
    tie point (x_scaled = half_levels - 1.5) must never end up outside
    [-half_levels, half_levels-1] after the parity correction is applied,
    regardless of which coordinate the parity flip lands on.
    """
    from research.quantizers.lattice_vq import LatticeD4Quantizer
    q = LatticeD4Quantizer(bits=4)
    hl = q.half_levels  # 8 for bits=4

    # Construct a group where coordinate 0 sits exactly at the upper
    # boundary tie (scaled value = hl - 1.5) and the other three coordinates
    # are chosen so a parity flip is forced and could target coordinate 0.
    alpha = float(hl - 1)  # so scale = alpha / (hl - 1) = 1.0 -> x_scaled == x
    x = torch.tensor([[
        hl - 1.5,   # ties at the upper boundary
        0.5,        # also a tie, smaller magnitude
        0.0,
        alpha,      # sets the group's max-abs (defines scale=1.0)
    ]])

    x_q, info = q(x)
    scale = info['scale'].unsqueeze(-1)
    z_int = (x_q / scale).round()
    in_range = ((z_int >= -hl) & (z_int <= hl - 1)).all().item()
    even_sum = (int(z_int.sum().item()) % 2 == 0)

    status = "OK" if (in_range and even_sum) else "FAIL"
    print(f"  [{status}] D4 boundary tie case: z={z_int.tolist()}, in_range={in_range}, even_sum={even_sum}")
    assert in_range, f"D4 boundary tie case produced out-of-range point: {z_int.tolist()}"
    assert even_sum, f"D4 boundary tie case left odd parity: {z_int.tolist()}"


def test_d4_scale_bits_option():
    """Verify the scale_bits=8 option still produces valid, in-range D4 points,
    and that its reported metadata overhead is exactly half of scale_bits=16."""
    from research.quantizers.lattice_vq import LatticeD4Quantizer

    torch.manual_seed(1)
    x = torch.randn(200, 64)

    q16 = LatticeD4Quantizer(bits=4, scale_bits=16)
    q8 = LatticeD4Quantizer(bits=4, scale_bits=8)

    x_q16, info16 = q16(x)
    x_q8, info8 = q8(x)

    for name, x_q, info, hl in [("scale_bits=16", x_q16, info16, q16.half_levels),
                                 ("scale_bits=8", x_q8, info8, q8.half_levels)]:
        scale = info['scale'].unsqueeze(-1)
        z_int = (x_q.reshape(200, 16, 4) / scale).round()
        even_frac = (z_int.sum(dim=-1).long() % 2 == 0).float().mean().item()
        in_range = ((z_int >= -hl) & (z_int <= hl - 1)).all().item()
        status = "OK" if (even_frac == 1.0 and in_range) else "FAIL"
        print(f"  [{status}] D4 {name}: even_sum_frac={even_frac:.4f}, in_range={in_range}, "
              f"scale_overhead_bits_per_scalar={info['scale_overhead_bits_per_scalar']}")
        assert even_frac == 1.0 and in_range, f"D4 {name} produced invalid lattice points"

    assert info16['scale_overhead_bits_per_scalar'] == 4.0, info16
    assert info8['scale_overhead_bits_per_scalar'] == 2.0, info8


def test_quantizer_comparison():
    """
    Compare MSE of scalar vs vector vs lattice quantizers at same bit-rate.
    After RHT rotation, lattice should have lowest MSE (validates the coding-gain
    direction empirically, not just claims it).
    """
    from research.transforms.hadamard import HadamardRotation
    from research.quantizers.lattice_vq import (
        ScalarRoundQuantizer, UniformVectorQuantizer, LatticeD4Quantizer
    )

    d = 64
    rot = HadamardRotation(dim=d)
    bits = 4

    # Create realistic ViT-like activations with outliers
    torch.manual_seed(0)
    x = torch.randn(1000, d)
    x[:, 0] *= 20  # outlier channel

    # Rotate
    x_rot = rot(x)

    # Quantize with each method
    scalar_q = ScalarRoundQuantizer(bits=bits)
    vector_q = UniformVectorQuantizer(bits=bits, group_size=4)
    lattice_q = LatticeD4Quantizer(bits=bits)

    results = {}
    for name, quantizer in [('Scalar', scalar_q), ('Vector-4', vector_q), ('Lattice-D4', lattice_q)]:
        x_q, info = quantizer(x_rot)
        mse = ((x_rot - x_q) ** 2).mean().item()
        results[name] = mse

    # Also test without rotation (to show rotation helps)
    x_q_no_rot, _ = scalar_q(x)
    mse_no_rot = ((x - x_q_no_rot) ** 2).mean().item()
    results['Scalar (NO rotation)'] = mse_no_rot

    print(f"\n  Quantizer MSE Comparison @ {bits}-bit:")
    print(f"  {'Method':<25} {'MSE':>10}")
    print(f"  {'-'*35}")
    for name, mse in sorted(results.items(), key=lambda t: t[1], reverse=True):
        marker = " <- WORST" if mse == max(results.values()) else \
                 " <- BEST" if mse == min(results.values()) else ""
        print(f"  {name:<25} {mse:>10.6f}{marker}")

    assert results['Lattice-D4'] < results['Scalar'], (
        f"D4 lattice MSE ({results['Lattice-D4']:.6f}) should be lower than "
        f"scalar MSE ({results['Scalar']:.6f}) at equal bit-rate post-RHT"
    )
    assert results['Scalar'] < results['Scalar (NO rotation)'], (
        "RHT rotation should reduce scalar-quantization MSE vs no rotation "
        f"(rotated={results['Scalar']:.6f}, unrotated={results['Scalar (NO rotation)']:.6f})"
    )


def test_qjl_bias_correction():
    """
    Verify QJL correction reduces attention score error, using enough
    projections (m=512) for the corrected estimator to actually pay off.

    IMPORTANT FINDING (docs/optimization_ledger.md, F3 addendum): after RHT,
    Q and the per-vector quantization error eps are close to ORTHOGONAL
    (empirically theta ~ 90 +/- a few degrees for ViT-Small-like d=64 data).
    The corrected cosine map cos(pi/2*(1-rho)) has derivative ~pi/2 (its
    steepest point) exactly at rho=0 (theta=90deg), so fixing the estimator's
    systematic bias AMPLIFIES its sampling variance precisely in the regime
    that dominates this data. At the module's default n_projections (128 for
    d=64), this variance amplification currently outweighs the bias fix:
    the corrected estimator is empirically WORSE than no correction at all
    (see test_qjl_default_projections_insufficient_at_low_m below). The
    corrected estimator only becomes net-positive vs. no correction beyond
    roughly m=200-256, and only beats the old (mathematically wrong, but
    lower-variance) raw-rho estimator beyond roughly m=700-1000. This is WHY
    docs/optimization_ledger.md T8's headline low-bit configuration disables
    QJL entirely (use_qjl=False) rather than relying on it at low overhead.
    """
    from research.quantizers.qjl_bias import QJLBiasCorrection
    from research.transforms.hadamard import HadamardRotation
    from research.quantizers.lattice_vq import ScalarRoundQuantizer

    d = 64
    n_q, n_k = 10, 20
    rot = HadamardRotation(dim=d)
    quantizer = ScalarRoundQuantizer(bits=3)  # aggressive 3-bit

    torch.manual_seed(0)
    Q = torch.randn(1, n_q, d)
    K = torch.randn(1, n_k, d)
    qjl = QJLBiasCorrection(dim=d, n_projections=512)  # enough to actually pay off; see docstring

    Q_rot = rot(Q)
    K_rot = rot(K)

    # True scores (using unquantized K)
    true_scores = Q_rot @ K_rot.transpose(-2, -1)

    # Quantized scores (no correction)
    K_q, _ = quantizer(K_rot)
    quant_scores = Q_rot @ K_q.transpose(-2, -1)

    # QJL-corrected scores
    K_error_signs, K_error_norms = qjl.encode(K_rot, K_q)
    corrected_scores = qjl.correct_scores(quant_scores, Q_rot, K_error_signs, K_error_norms)

    error_before = (true_scores - quant_scores).abs().mean().item()
    error_after = (true_scores - corrected_scores).abs().mean().item()
    improvement = (1 - error_after / error_before) * 100

    status = "OK" if error_after < error_before else "FAIL"
    print(f"\n  [{status}] QJL bias correction @ 3-bit (m=512):")
    print(f"    Score error before QJL: {error_before:.4f}")
    print(f"    Score error after QJL:  {error_after:.4f}")
    print(f"    Improvement: {improvement:.1f}%")
    assert error_after < error_before, (
        f"QJL correction did not improve score error: before={error_before:.4f}, "
        f"after={error_after:.4f}"
    )


def test_qjl_default_projections_insufficient_at_low_m():
    """
    Documents (rather than hides) the bias-variance tradeoff found while
    validating T3: at the module's DEFAULT n_projections (128 for d=64 —
    docs/optimization_ledger.md T3's own prescribed 'min(2d,128)' default),
    the mathematically-corrected cosine estimator does NOT yet beat doing
    no correction at all on this RHT+3-bit-quantization data, because Q and
    the quantization error are close to orthogonal and the correction's
    variance is highest exactly there. This test pins that finding down
    (rather than letting the default silently regress unnoticed) and checks
    that correction quality improves monotonically-ish as m grows, providing
    a regression guard on the theoretical asymptotic-correctness claim.
    """
    from research.quantizers.qjl_bias import QJLBiasCorrection, default_qjl_projections
    from research.transforms.hadamard import HadamardRotation
    from research.quantizers.lattice_vq import ScalarRoundQuantizer

    d = 64
    rot = HadamardRotation(dim=d)
    quantizer = ScalarRoundQuantizer(bits=3)

    torch.manual_seed(0)
    Q = torch.randn(1, 10, d)
    K = torch.randn(1, 20, d)
    Q_rot = rot(Q)
    K_rot = rot(K)
    true_scores = Q_rot @ K_rot.transpose(-2, -1)
    K_q, _ = quantizer(K_rot)
    quant_scores = Q_rot @ K_q.transpose(-2, -1)
    error_before = (true_scores - quant_scores).abs().mean().item()

    default_m = default_qjl_projections(d)
    assert default_m == 128, default_m

    errors = {}
    for m in (default_m, 1024):
        torch.manual_seed(2)
        qjl = QJLBiasCorrection(dim=d, n_projections=m)
        K_error_signs, K_error_norms = qjl.encode(K_rot, K_q)
        corrected = qjl.correct_scores(quant_scores, Q_rot, K_error_signs, K_error_norms)
        errors[m] = (true_scores - corrected).abs().mean().item()

    print(f"  error_before={error_before:.4f}, error@m={default_m}: {errors[default_m]:.4f}, "
          f"error@m=1024: {errors[1024]:.4f}")
    # The known finding: the default is under-provisioned for this geometry.
    assert errors[default_m] > error_before, (
        "Expected the current default n_projections to be insufficient for net-positive "
        "QJL correction on near-orthogonal Q/eps geometry (a known, documented tradeoff); "
        "if this now passes, the geometry or default has changed and this test (and the "
        "surrounding docs) should be revisited rather than silently deleted."
    )
    # But correction quality must improve substantially with more projections,
    # confirming the fix is asymptotically correct (not just differently wrong).
    assert errors[1024] < errors[default_m], (
        f"More projections should reduce corrected-score error: "
        f"m={default_m} gave {errors[default_m]:.4f}, m=1024 gave {errors[1024]:.4f}"
    )
    assert errors[1024] < error_before, (
        f"At m=1024 QJL correction should beat no correction: "
        f"error_before={error_before:.4f}, error@1024={errors[1024]:.4f}"
    )


def test_qjl_cosine_calibration():
    """
    The QJL sign-correlation statistic rho = (1/m) sign(RQ)^T sign(Re) is a
    SimHash collision statistic: E[rho] = 1 - 2*theta/pi, NOT cos(theta).
    correct_scores() must map rho through cos(pi/2*(1-rho)) before using it
    as a cosine estimate. This test constructs vector pairs at known angles
    and checks the mapped estimate is close to the true cosine, and strictly
    closer than using raw rho directly.
    """
    from research.quantizers.qjl_bias import QJLBiasCorrection

    d = 64
    m = 128
    torch.manual_seed(0)
    qjl = QJLBiasCorrection(dim=d, n_projections=m)

    def make_pair(theta):
        a = torch.randn(d)
        a = a / a.norm()
        b_rand = torch.randn(d)
        b_rand = b_rand - (b_rand @ a) * a
        b_rand = b_rand / b_rand.norm()
        b = torch.cos(torch.tensor(theta)) * a + torch.sin(torch.tensor(theta)) * b_rand
        return a, b

    for theta_deg in [30.0, 45.0, 60.0, 90.0]:
        theta = np.deg2rad(theta_deg)
        true_cos = float(np.cos(theta))

        raw_rho_samples = []
        mapped_samples = []
        n_trials = 50
        for _ in range(n_trials):
            a, b = make_pair(theta)
            a_sign = (a @ qjl.R.T).sign()
            b_sign = (b @ qjl.R.T).sign()
            a_sign[a_sign == 0] = 1.0
            b_sign[b_sign == 0] = 1.0
            rho = (a_sign * b_sign).mean().item()
            raw_rho_samples.append(rho)
            mapped_samples.append(np.cos(np.pi / 2 * (1 - np.clip(rho, -1, 1))))

        mean_raw = float(np.mean(raw_rho_samples))
        mean_mapped = float(np.mean(mapped_samples))
        raw_bias = abs(mean_raw - true_cos)
        mapped_bias = abs(mean_mapped - true_cos)

        status = "OK" if mapped_bias < 0.05 else "FAIL"
        print(f"  [{status}] theta={theta_deg:.0f} deg: true_cos={true_cos:.3f}, "
              f"raw_rho={mean_raw:.3f} (bias={raw_bias:.3f}), "
              f"mapped={mean_mapped:.3f} (bias={mapped_bias:.3f})")
        assert mapped_bias < 0.05, (
            f"Mapped cosine estimate bias too large at theta={theta_deg}: {mapped_bias:.3f}"
        )
        if theta_deg == 45.0:
            assert mapped_bias < raw_bias, (
                "Mapped cosine estimate should be strictly less biased than raw rho at theta=45deg"
            )


def test_qjl_correct_scores_bias_direction():
    """
    Regression test that exercises QJLBiasCorrection.correct_scores() itself
    (not a standalone reimplementation of the math) at three known angles
    between Q and the quantization error eps: near-parallel (theta=10deg,
    cos~0.98), orthogonal (theta=90deg, cos=0), near-anti-parallel
    (theta=170deg, cos~-0.98). Q and eps are both unit vectors, so the bias
    estimate should track cos(theta) closely. This would fail if
    correct_scores used the raw sign-correlation rho directly (linear in
    theta) instead of the mapped cosine, since at these three angles the raw
    statistic is off by a large, easily detectable margin relative to the
    true cosine (see docs/optimization_ledger.md finding F3).
    """
    from research.quantizers.qjl_bias import QJLBiasCorrection

    d = 64
    torch.manual_seed(3)
    qjl = QJLBiasCorrection(dim=d, n_projections=256)  # large m for a tight single-draw estimate

    def unit(v):
        return v / v.norm()

    q_dir = unit(torch.randn(d))

    def make_eps(theta_deg):
        theta = np.deg2rad(theta_deg)
        perp = torch.randn(d)
        perp = unit(perp - (perp @ q_dir) * q_dir)
        return np.cos(theta) * q_dir + np.sin(theta) * perp

    Q = q_dir.unsqueeze(0).unsqueeze(0)  # (1, 1, d), ||Q|| = 1

    for theta_deg, expect in [(10.0, "positive"), (90.0, "zero"), (170.0, "negative")]:
        eps = make_eps(theta_deg).unsqueeze(0).unsqueeze(0)  # (1, 1, d), ||eps|| = 1
        K_q = torch.zeros_like(eps)  # encode() computes epsilon = x_original - x_quantized == eps exactly
        signs, norms = qjl.encode(eps, K_q)
        attn_scores = torch.zeros(1, 1, 1)
        bias = qjl.correct_scores(attn_scores, Q, signs, norms).item()

        true_cos = float(np.cos(np.deg2rad(theta_deg)))
        status = "OK"
        if expect == "positive" and not (bias > 0.3):
            status = "FAIL"
        if expect == "negative" and not (bias < -0.3):
            status = "FAIL"
        if expect == "zero" and not (abs(bias) < 0.3):
            status = "FAIL"
        print(f"  [{status}] theta={theta_deg} deg: bias_est={bias:.3f} (true cos={true_cos:.3f})")

        if expect == "positive":
            assert bias > 0.3, f"Expected strongly positive bias at theta={theta_deg}, got {bias:.3f}"
        elif expect == "negative":
            assert bias < -0.3, f"Expected strongly negative bias at theta={theta_deg}, got {bias:.3f}"
        else:
            assert abs(bias) < 0.3, f"Expected near-zero bias at theta=90, got {bias:.3f}"


def test_qjl_ablation_flag_reduces_error_when_disabled():
    """
    Sanity check that QJL is genuinely ablatable at the module level: with
    use_qjl=False, RotatedSelfAttention must skip QJL entirely (self.qjl is
    None), so results differ from the use_qjl=True configuration. This
    guards against the CLI-level '--use_qjl action=store_true default=True'
    bug where QJL could never actually be turned off (docs/optimization_ledger.md F8).
    """
    from research.models.rotated_attention import RotatedSelfAttention

    torch.manual_seed(0)
    attn_on = RotatedSelfAttention(dim=64, num_heads=1, bits=3, quantizer='lattice_d4', use_qjl=True)
    attn_off = RotatedSelfAttention(dim=64, num_heads=1, bits=3, quantizer='lattice_d4', use_qjl=False)

    assert attn_on.qjl is not None, "use_qjl=True should construct a QJLBiasCorrection module"
    assert attn_off.qjl is None, "use_qjl=False should leave self.qjl as None (ablatable)"
    print("  [OK] QJL module presence correctly toggled by use_qjl flag")


def test_rotated_self_attention_rejects_attn_bias():
    """
    RotatedSelfAttention must raise NotImplementedError (not silently ignore)
    when an attn_bias/attention_mask/mask kwarg is passed non-None, since its
    quantized-K pipeline has no path to apply one (docs/optimization_ledger.md
    T7, finding F9).
    """
    from research.models.rotated_attention import RotatedSelfAttention

    attn = RotatedSelfAttention(dim=64, num_heads=4, bits=4, quantizer='lattice_d4', use_qjl=False)
    attn.eval()
    x = torch.randn(2, 10, 64)

    for kwarg_name in ('attn_bias', 'attention_mask', 'mask'):
        raised = False
        try:
            with torch.no_grad():
                attn(x, **{kwarg_name: torch.zeros(2, 10, 10)})
        except NotImplementedError:
            raised = True
        print(f"  [{'OK' if raised else 'FAIL'}] RotatedSelfAttention rejects non-None '{kwarg_name}': raised={raised}")
        assert raised, f"Expected NotImplementedError when '{kwarg_name}' is non-None"

    # Sanity: still works fine with no bias/mask kwargs at all.
    with torch.no_grad():
        out = attn(x)
    assert out.shape == x.shape


def test_rotated_self_attention_e8_headline_config():
    """
    Verify RotatedSelfAttention runs end-to-end with the T8 headline
    ≤4.0-effective-bits/scalar configuration: lattice_e8, 3-bit, 8-bit
    scales, QJL disabled (docs/optimization_ledger.md T8).
    """
    from research.models.rotated_attention import RotatedSelfAttention

    dim, heads = 384, 6  # head_dim = 64, divisible by 8 for E8 groups
    attn = RotatedSelfAttention(
        dim=dim, num_heads=heads, bits=3, quantizer='lattice_e8', use_qjl=False, scale_bits=8,
    )
    attn.eval()

    x = torch.randn(2, 50, dim)
    with torch.no_grad():
        out = attn(x)

    status = "OK" if out.shape == x.shape else "FAIL"
    print(f"\n  [{status}] RotatedSelfAttention (E8 headline config): input {x.shape} -> output {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    assert not torch.isnan(out).any(), "E8-config output contains NaN"
    assert not torch.isinf(out).any(), "E8-config output contains Inf"
    assert attn.k_quantizer.group_size == 8, "K quantizer should be using E8's group_size=8"
    assert attn.qjl is None, "QJL should be disabled in the headline config"


def test_rotated_self_attention():
    """Verify RotatedSelfAttention runs end-to-end with no NaN/Inf."""
    from research.models.rotated_attention import RotatedSelfAttention

    dim, heads = 384, 6
    attn = RotatedSelfAttention(
        dim=dim, num_heads=heads, bits=4, quantizer='lattice_d4', use_qjl=True
    )
    attn.eval()

    x = torch.randn(2, 50, dim)  # (batch, tokens, dim)
    with torch.no_grad():
        out = attn(x)

    status = "OK" if out.shape == x.shape else "FAIL"
    print(f"\n  [{status}] RotatedSelfAttention: input {x.shape} -> output {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    assert not torch.isnan(out).any(), "RotatedSelfAttention output contains NaN"
    assert not torch.isinf(out).any(), "RotatedSelfAttention output contains Inf"


def test_rotated_temporal_attention():
    """Verify RotatedTemporalAttention runs end-to-end with no NaN/Inf."""
    from research.models.rotated_attention import RotatedTemporalAttention

    dim, heads = 256, 8
    attn = RotatedTemporalAttention(
        dim=dim, num_heads=heads, bits=4, quantizer='lattice_d4', use_qjl=True
    )
    attn.eval()

    query = torch.randn(2, 100, dim)    # current frame
    context = torch.randn(2, 100, dim)  # previous frame

    with torch.no_grad():
        out = attn(query, context)

    status = "OK" if out.shape == query.shape else "FAIL"
    print(f"  [{status}] RotatedTemporalAttention: query {query.shape} + context {context.shape} -> {out.shape}")
    assert out.shape == query.shape, f"Shape mismatch: {out.shape} != {query.shape}"
    assert not torch.isnan(out).any(), "RotatedTemporalAttention output contains NaN"
    assert not torch.isinf(out).any(), "RotatedTemporalAttention output contains Inf"


if __name__ == '__main__':
    print("=" * 60)
    print("  VDA-HyperQuant -- Core Verification Tests")
    print("=" * 60)

    tests = [
        ("[1] Hadamard Orthogonality", test_hadamard_orthogonality),
        ("[2] Fast Hadamard Transform vs Matrix", test_fht_matches_matrix),
        ("[3] FHT Does Not Mutate Input", test_fht_does_not_mutate_input),
        ("[4] FHT fp16 Precision", test_fht_fp16_precision),
        ("[5] RHT Invertibility", test_rht_invertibility),
        ("[6] RHT Outlier Suppression (THE KEY EXPERIMENT)", test_rht_outlier_suppression),
        ("[7] D4 Lattice Quantizer Parity", test_d4_lattice_parity),
        ("[7b] D4 Boundary Edge Case", test_d4_boundary_edge_case),
        ("[7c] D4 scale_bits Option", test_d4_scale_bits_option),
        ("[8] Quantizer MSE Comparison", test_quantizer_comparison),
        ("[9] QJL Bias Correction", test_qjl_bias_correction),
        ("[9b] QJL Default-m Bias-Variance Tradeoff", test_qjl_default_projections_insufficient_at_low_m),
        ("[10] QJL Cosine Calibration", test_qjl_cosine_calibration),
        ("[10b] QJL correct_scores Bias Direction", test_qjl_correct_scores_bias_direction),
        ("[10c] QJL Ablation Flag", test_qjl_ablation_flag_reduces_error_when_disabled),
        ("[10d] RotatedSelfAttention Rejects attn_bias", test_rotated_self_attention_rejects_attn_bias),
        ("[11] RotatedSelfAttention End-to-End", test_rotated_self_attention),
        ("[11b] RotatedSelfAttention E8 Headline Config", test_rotated_self_attention_e8_headline_config),
        ("[12] RotatedTemporalAttention End-to-End", test_rotated_temporal_attention),
    ]

    n_failed = 0
    for label, fn in tests:
        print(f"\n{label}")
        try:
            fn()
        except AssertionError as e:
            n_failed += 1
            print(f"  !!! ASSERTION FAILED: {e}")

    print(f"\n{'=' * 60}")
    if n_failed == 0:
        print("  All tests passed!")
    else:
        print(f"  {n_failed} test(s) FAILED")
    print(f"{'=' * 60}")

    sys.exit(1 if n_failed else 0)
