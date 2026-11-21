"""imgfeat.py - turns a banknote photo into a fixed-length feature vector.
The same function is used for training and for the website. Needs only numpy + Pillow."""
import numpy as np
from PIL import Image

SIZE = (480, 240)


def _prep(img):
    img = img.convert("RGB")
    if img.height > img.width:
        img = img.rotate(90, expand=True)
    return img.resize(SIZE, Image.BILINEAR)


def _entropy(g):
    hist, _ = np.histogram(g, bins=256, range=(0, 255))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _moments(x):
    m = float(x.mean())
    s = float(x.std()) + 1e-9
    z = (x - m) / s
    return m, s, float((z ** 3).mean()), float((z ** 4).mean() - 3.0)


def _haar(a):
    a = a[: a.shape[0] // 2 * 2, : a.shape[1] // 2 * 2]
    p, q, r, s = a[0::2, 0::2], a[0::2, 1::2], a[1::2, 0::2], a[1::2, 1::2]
    return (p + q + r + s) / 4.0, (p + q - r - s) / 4.0, (p - q + r - s) / 4.0, (p - q - r + s) / 4.0


def _lbp_hist(g):
    c = g[1:-1, 1:-1]
    code = np.zeros(c.shape, dtype=np.uint8)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    h, w = g.shape
    for i, (dy, dx) in enumerate(shifts):
        nb = g[1 + dy: h - 1 + dy, 1 + dx: w - 1 + dx]
        code |= ((nb >= c).astype(np.uint8) << i)
    hist, _ = np.histogram(code, bins=32, range=(0, 256))
    return hist / max(hist.sum(), 1)


def _fft_bands(g, bands=8):
    small = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((256, 128), Image.BILINEAR), dtype=float)
    small = small - small.mean()
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(small))))
    h, w = mag.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    out = []
    for i in range(bands):
        sel = (rr >= i / bands * 1.4) & (rr < (i + 1) / bands * 1.4)
        out.append(float(mag[sel].mean()) if sel.any() else 0.0)
    return out


def extract(pil_img):
    """Return (feature_vector, report_dict) for one PIL image."""
    ow, oh = pil_img.size
    img = _prep(pil_img)
    g = np.asarray(img.convert("L"), dtype=float)
    feats = []
    m, s, sk, ku = _moments(g)
    feats += [m, s, sk, ku, _entropy(g)]
    gx = np.diff(g, axis=1)[:-1, :]
    gy = np.diff(g, axis=0)[:, :-1]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_density = float((mag > 25).mean())
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    sharp = float(lap.var())
    feats += [edge_density, float(mag.mean()), float(np.log1p(sharp))]
    a = g
    for _ in range(2):
        a, lh, hl, hh = _haar(a)
        for band in (lh, hl, hh):
            feats += [float(np.log1p((band ** 2).mean())), float(band.std())]
    feats += list(_lbp_hist(g))
    feats += _fft_bands(g)
    hsv = np.asarray(img.convert("HSV"), dtype=float)
    for ch in range(3):
        feats += [float(hsv[..., ch].mean()), float(hsv[..., ch].std())]
    colourful = hsv[..., 1] > 40
    if colourful.sum() > 50:
        hh_, _ = np.histogram(hsv[..., 0][colourful], bins=12, range=(0, 256))
        hh_ = hh_ / max(hh_.sum(), 1)
    else:
        hh_ = np.zeros(12)
    feats += list(hh_)
    vec = np.nan_to_num(np.array(feats, dtype=float))

    long_s, short_s = max(ow, oh), max(min(ow, oh), 1)
    aspect = long_s / short_s
    warnings = []
    if short_s < 240:
        warnings.append("Low resolution: use a larger or closer photo.")
    if sharp < 25:
        warnings.append("The photo looks blurry: hold the camera steady and refocus.")
    if m < 55:
        warnings.append("The photo is too dark: use brighter, even light.")
    elif m > 215:
        warnings.append("The photo is overexposed: avoid glare and direct flash.")
    if aspect < 1.3 or aspect > 3.2:
        warnings.append("This does not look like a banknote shape: crop tightly around one note.")
    report = {
        "width": ow, "height": oh, "aspect": round(aspect, 2),
        "sharpness": round(min(sharp / 400.0, 1.0) * 100, 1),
        "brightness": round(m / 255.0 * 100, 1),
        "contrast": round(min(s / 80.0, 1.0) * 100, 1),
        "edge_detail": round(min(edge_density / 0.25, 1.0) * 100, 1),
        "texture": round(min(_entropy(g) / 8.0, 1.0) * 100, 1),
        "warnings": warnings,
    }
    return vec, report
