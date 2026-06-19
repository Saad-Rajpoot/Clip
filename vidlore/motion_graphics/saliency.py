"""Fast subject-point (saliency) detection for footage callouts / spotlights.

RC4 anchor-precision fix. Pure numpy-FFT + OpenCV **spectral-residual** saliency
(Hou & Zhang 2007) — NO ML model, NO extra deps (`cv2.saliency`/contrib not
required). It returns the normalised (x, y) of the visual SUBJECT of a frame so a
pointer no longer defaults to the geometric centre (which landed on film-strip
sprocket holes / flat wood when the subject was off-centre).

Why spectral-residual + post-processing beats naive contrast here:
  * A large Gaussian blur of the saliency map MERGES fine repetitive structure
    (sprocket-hole rows, grain, dither) into low smooth bands, so a compact unique
    subject out-scores the repetition.
  * A border safe-zone suppresses frame edges / UI strips.
  * A gentle centre bias breaks ties toward the middle WITHOUT forcing centre — a
    strongly-salient off-centre subject still wins.
  * A robust weighted centre-of-mass of the salient core (not raw argmax) ignores
    isolated hot pixels (noise).

Deterministic, ~1-3 ms on a 128px frame, degrades to centre on ANY failure so a
render is never blocked.

    from vidlore.motion_graphics.saliency import salient_point
    nx, ny = salient_point(bg_image)          # PIL.Image | np.ndarray | path
"""
from __future__ import annotations

import numpy as np

_DEFAULT = (0.5, 0.5)


def _to_gray_small(img, width: int = 128):
    import cv2
    from PIL import Image
    if isinstance(img, Image.Image):
        a = np.asarray(img.convert("L"))
    elif isinstance(img, np.ndarray):
        a = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        a = np.asarray(Image.open(str(img)).convert("L"))
    h, w = a.shape[:2]
    if w > width:
        nh = max(1, int(round(h * width / float(w))))
        a = cv2.resize(a, (width, nh), interpolation=cv2.INTER_AREA)
    return a.astype(np.float32) / 255.0


def _spectral_residual(gray):
    import cv2
    f = np.fft.fft2(gray)
    amp = np.abs(f)
    phase = np.angle(f)
    log_amp = np.log(amp + 1e-8)
    smooth = cv2.blur(log_amp, (3, 3))
    residual = log_amp - smooth
    recon = np.fft.ifft2(np.exp(residual + 1j * phase))
    return recon.real ** 2 + recon.imag ** 2


def _sharpness_map(gray, sig):
    """A normalised FOCUS map: high where the image is sharp / in-focus (the real
    subject), low where it is smooth (out-of-focus bokeh, blown-out backdrops, flat
    sky/wall). |Laplacian| is local high-frequency energy; a blur at the saliency
    scale turns per-pixel edges into a smooth focus field. This is the key signal
    that stops a bright BLURRED bokeh highlight from being mistaken for the subject:
    bokeh is bright but has near-zero high-frequency energy, so its focus weight is
    ~0. A uniformly-sharp frame (e.g. a clean AI macro) yields a near-flat map, so
    it never hurts — saliency + centre bias still decide there."""
    import cv2
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    energy = cv2.GaussianBlur(np.abs(lap), (0, 0),
                              sigmaX=max(2.0, sig), sigmaY=max(2.0, sig))
    m = float(energy.max())
    if m > 1e-9:
        energy = energy / m
    return energy


def saliency_map(img, width: int = 96):
    """Return the post-processed saliency map (float32, max-normalised) + its gray
    frame, or (None, None) on failure. Exposed for the test harness. A coarse 96px
    grid + a wide merge blur dissolves fine repetitive structure (sprocket rows,
    grain) so the COM settles on the subject mass, not a bright repeat-line."""
    try:
        import cv2
        g = _to_gray_small(img, width)
        h, w = g.shape
        if h < 8 or w < 8:
            return None, None
        sal = _spectral_residual(g)
        sig = max(2.0, 0.06 * max(h, w))             # merge repetitive fine structure
        sal = cv2.GaussianBlur(sal, (0, 0), sigmaX=sig, sigmaY=sig)
        if sal.max() > 1e-12:
            sal = sal / sal.max()
        # FOCUS GATE (permanent anti-bokeh fix): the in-focus subject is sharp; an
        # out-of-focus background (green foliage bokeh, blown-out backdrop) is
        # smooth. Spectral-residual saliency alone ranks the bright blurred bokeh
        # highlights as "subject" — which is exactly how a callout ring landed on
        # empty green background instead of the fence post. Gating saliency by a
        # sharpness map suppresses anything that is bright-but-blurred, so only a
        # genuinely sharp region can win. Soft floor (0.18) keeps it from zeroing
        # a slightly-soft subject; a uniformly-sharp frame gets a flat gate.
        sharp = _sharpness_map(g, sig)
        sal = sal * (0.18 + 0.82 * sharp)
        if sal.max() > 1e-12:
            sal = sal / sal.max()
        return sal, g
    except Exception:                                # noqa: BLE001
        return None, None


def salient_point(img, *, safe: float = 0.10, center_bias: float = 0.45,
                  default=_DEFAULT):
    """Normalised (x, y) in [0.08,0.92]/[0.10,0.90] of the frame's visual subject.

    `safe` = border fraction suppressed (edges/UI). `center_bias` in [0,1] = how
    much a central location is favoured on ties (0 = none). Falls back to `default`
    (centre) if the subject is ambiguous or anything errors."""
    sal, g = saliency_map(img)
    if sal is None:
        return default
    try:
        h, w = g.shape
        # border safe-zone
        by, bx = int(h * safe), int(w * safe)
        mask = np.zeros_like(sal)
        mask[by:h - by, bx:w - bx] = 1.0
        sal = sal * mask
        if sal.max() <= 1e-9:
            return default
        sal = sal / sal.max()
        # gentle centre bias (broad gaussian; never forces centre)
        yy, xx = np.mgrid[0:h, 0:w]
        cxg, cyg = (w - 1) / 2.0, (h - 1) / 2.0
        rad2 = ((xx - cxg) / (w * 0.5)) ** 2 + ((yy - cyg) / (h * 0.5)) ** 2
        cgauss = np.exp(-rad2 / (2 * 0.70 ** 2))
        sal = sal * (1.0 - center_bias + center_bias * cgauss)
        # ambiguity guard: if the salient mass is spread over most of the frame
        # (no clear subject), don't pretend — use centre.
        pos = sal[sal > 0]
        if pos.size < 4:
            return default
        thr = np.percentile(pos, 82)
        core = sal >= thr
        if core.sum() < 3 or core.sum() > 0.55 * mask.sum():
            return default
        ws = sal[core]
        ys, xs = np.nonzero(core)
        cx = float((xs * ws).sum() / ws.sum())
        cy = float((ys * ws).sum() / ws.sum())
        nx = min(0.92, max(0.08, (cx + 0.5) / w))
        ny = min(0.90, max(0.10, (cy + 0.5) / h))
        return (round(nx, 4), round(ny, 4))
    except Exception:                                # noqa: BLE001
        return default
