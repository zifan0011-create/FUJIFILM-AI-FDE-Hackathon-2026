import numpy as np
from PIL import Image
from scipy import ndimage as ndi

BLOCKS = ("rings", "components", "local")
RING_SCALES = (2, 4, 7, 12)
ANGLES = np.arange(12) * (2 * np.pi / 12)
COS, SIN = np.cos(ANGLES), np.sin(ANGLES)
SCALES = (1.5, 2.5, 4., 6.5, 10.)
ANGLE = np.arange(16) * np.pi / 8
DY, DX = np.sin(ANGLE), np.cos(ANGLE)


def tails(values):
    flat = values.ravel()
    k = max(1, flat.size // 100)
    ordered = np.partition(flat, (flat.size // 2, flat.size - k))
    return [float(ordered[flat.size // 2]), float(ordered[-k:].mean()), float(flat.max())]


def visual_features(gray):
    # Signed DoG responses capture holes and bright fills at several apparent sizes.
    smooth = [ndi.gaussian_filter(gray, scale) for scale in (1, 2.5, 5, 10, 18)]
    contrast = []
    for fine, coarse in zip(smooth, smooth[1:]):
        response = fine - coarse
        for signed in (response, -response):
            contrast.extend(tails(signed))
            contrast.extend(tails(np.concatenate(
                [signed[:16].ravel(), signed[-16:].ravel(),
                 signed[16:-16, :16].ravel(), signed[16:-16, -16:].ravel()])))

    # Describe connected dark responses, including components touching the tile edge.
    geometry = []
    for response in (smooth[2] - smooth[0], smooth[3] - smooth[1], smooth[4] - smooth[2]):
        mask = response > max(0.015, float(response.max()) * 0.35)
        components, count = ndi.label(mask)
        areas = np.bincount(components.ravel())
        strength = np.bincount(components.ravel(), weights=response.ravel())
        boxes = ndi.find_objects(components)
        indices = sorted((i for i in range(1, count + 1) if areas[i] >= 6),
                         key=lambda i: strength[i], reverse=True)[:2]
        for index in indices:
            ys, xs = boxes[index - 1]
            height, width = ys.stop - ys.start, xs.stop - xs.start
            border = min(ys.start, xs.start, 128 - ys.stop, 128 - xs.stop)
            geometry.extend([np.log1p(areas[index]), min(height, width) / max(height, width),
                             areas[index] / (height * width), strength[index] / areas[index],
                             border / 128])
        geometry.extend([0.0] * (5 * (2 - len(indices))))

    # Grain and directional edge energy help distinguish concrete, cracks and pipe rims.
    texture = tails(np.abs(gray - smooth[0]))
    gy, gx = np.gradient(smooth[0])
    energy = gx * gx + gy * gy
    xx = ndi.uniform_filter(gx * gx, 9)
    yy = ndi.uniform_filter(gy * gy, 9)
    xy = ndi.uniform_filter(gx * gy, 9)
    anisotropy = np.sqrt((xx - yy) ** 2 + 4 * xy ** 2)
    texture.extend(tails(energy))
    texture.extend(tails(anisotropy))
    texture.extend(tails(np.maximum(0, xx + yy - anisotropy)))
    for patch in (gray, gray[:16], gray[-16:], gray[:, :16], gray[:, -16:]):
        texture.extend([float(patch.mean()), float(patch.std())])
    return contrast, geometry, texture


def image_features(path):
    with Image.open(path) as image:
        if image.size != (256, 256):
            raise ValueError(f"Unexpected dimensions: {path}")
        mode = image.mode
        gray = np.asarray(image.convert("L").resize((128, 128), Image.Resampling.BILINEAR),
                          dtype=np.float32) / 255
    return visual_features(gray), mode


def blocks(path):
    with Image.open(path) as image:
        if image.size != (256, 256):
            raise ValueError(f"Unexpected dimensions for {path}")
        rgb = np.asarray(image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR),
                         dtype=np.float32) / 255
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    fine = ndi.gaussian_filter(gray, 1)
    rings, components, local = [], [], []
    for scale in RING_SCALES:
        narrow = ndi.gaussian_filter(gray, scale)
        broad = ndi.gaussian_filter(gray, scale * 1.65)
        residual = narrow - broad
        variance = np.maximum(0, ndi.gaussian_filter(gray * gray, scale * 1.65) - broad * broad)
        normalized = residual / (np.sqrt(variance) + 0.015)
        for sign in (-1, 1):
            response = sign * normalized
            peak = np.argmax(response)
            y, x = np.unravel_index(peak, gray.shape)
            radius = 2 * scale
            yy, xx = y + radius * SIN, x + radius * COS
            visible = (yy >= 0) & (yy < 128) & (xx >= 0) & (xx < 128)
            ring = ndi.map_coordinates(fine, [yy[visible], xx[visible]], order=1, prefilter=False)
            delta = sign * (fine[y, x] - ring)
            rings.extend([response[y, x], sign * residual[y, x], np.mean(delta),
                          np.median(delta), np.mean(delta > 0.012), np.std(delta),
                          np.mean(visible)])
            for region in (response[12:-12, 12:-12], np.concatenate(
                    (response[:12].ravel(), response[-12:].ravel(),
                     response[12:-12, :12].ravel(), response[12:-12, -12:].ravel()))):
                rings.extend([float(region.max()), float(np.quantile(region, 0.99))])
            mask = response >= max(0.28, 0.45 * response[y, x])
            labels, _ = ndi.label(mask)
            component = labels[y, x]
            area = labels == component if component else np.zeros_like(mask)
            ys, xs = np.nonzero(area)
            if len(ys) >= 2:
                height, width = np.ptp(ys) + 1, np.ptp(xs) + 1
                cov = np.cov(np.vstack((ys, xs)), bias=True)
                axis = np.linalg.eigvalsh(cov)
                perimeter = np.count_nonzero(area ^ ndi.binary_erosion(area))
                components.extend([np.log1p(len(ys)), min(height, width) / max(height, width),
                                   len(ys) / (height * width), np.sqrt(axis[0] / (axis[1] + 1e-9)),
                                   perimeter / np.sqrt(len(ys)), np.mean(response[area]),
                                   np.mean((ys == 0) | (ys == 127) | (xs == 0) | (xs == 127)),
                                   np.mean(variance[area])])
            else:
                components.extend([0.] * 8)
            pad = max(3, radius)
            section = np.s_[max(0, y-pad):min(128, y+pad+1), max(0, x-pad):min(128, x+pad+1)]
            colors = rgb[section]
            grain = (gray - fine)[section]
            local.extend([np.mean(colors[..., 0] - colors[..., 2]),
                          np.std(colors[..., 0] - colors[..., 2]),
                          np.mean(np.ptp(colors, axis=2)), np.std(grain),
                          np.std(gray[section]), np.mean(grain**2)])
    return dict(zip(BLOCKS, (rings, components, local)))


def targeted_features(path):
    with Image.open(path) as image:
        if image.size != (256, 256):
            raise ValueError(f"Wrong size: {path}")
        gray = np.asarray(image.convert("L").resize((128, 128), Image.Resampling.BILINEAR),
                          dtype=np.float32) / 255
    smooth = ndi.gaussian_filter(gray, 1)
    py, px = np.gradient(smooth)
    magnitude = np.hypot(px, py)
    responses = []
    for scale in SCALES:
        hxx = ndi.gaussian_filter(gray, scale, order=(0, 2)) * scale**2
        hyy = ndi.gaussian_filter(gray, scale, order=(2, 0)) * scale**2
        hxy = ndi.gaussian_filter(gray, scale, order=(1, 1)) * scale**2
        trace = hxx + hyy
        strength = np.sqrt(np.maximum(0, hxx*hyy-hxy*hxy))
        responses.append(np.sign(trace) * strength)
    proposed = []
    for sign in (1, -1):
        for border in (False, True):
            limits = np.zeros(gray.shape, dtype=bool)
            limits[18:-18, 18:-18] = True
            if border:
                limits = ~limits
            cube = np.where(limits[None, ...], sign*np.stack(responses), -np.inf)
            stats = []
            for j in range(3):
                level, y, x = np.unravel_index(np.argmax(cube), cube.shape)
                strength = cube[level, y, x]
                if not np.isfinite(strength) or strength <= 0:
                    stats.append(np.zeros(9))
                    break
                size = SCALES[level]
                radius = 2.5*size
                yy, xx = y+radius*DY, x+radius*DX
                visible = (yy >= 0) & (yy <= 127) & (xx >= 0) & (xx <= 127)
                sample = [yy[visible], xx[visible]]
                around = ndi.map_coordinates(smooth, sample, order=1, prefilter=False)
                gy = ndi.map_coordinates(py, sample, order=1, prefilter=False)
                gx = ndi.map_coordinates(px, sample, order=1, prefilter=False)
                radial = sign*(gy*DY[visible]+gx*DX[visible])
                support = np.hypot(gy, gx)
                contrast = sign*(around-smooth[y, x])
                stats.append(np.array([strength, size, np.mean(contrast), np.std(contrast),
                                       np.mean(radial), np.mean(radial/(support+.004)),
                                       np.mean(radial > .004), np.mean(visible),
                                       min(x, 127-x, y, 127-y)/128]))
                delta = max(5, int(radius))
                cube[:, max(0,y-delta):y+delta+1, max(0,x-delta):x+delta+1] = -np.inf
            stats = np.array(stats)
            proposed.extend(np.r_[stats[0], stats.mean(axis=0), stats.max(axis=0)])
    shade = ndi.gaussian_filter(gray, 8)
    q10, q25, q50, q90 = np.quantile(shade, [.1, .25, .5, .9])
    dark = shade < q50 - max(.055, .3*(q90-q10))
    edge_stats = []
    # Rotation-invariant summaries across the four border strips.
    for width in (6, 16, 32):
        stats = []
        for k in range(4):
            stripe = np.rot90(shade, k)[:width]
            mask = np.rot90(dark, k)[:width]
            g = np.rot90(magnitude, k)[:width]
            _, label_runs = ndi.label(mask.mean(axis=0) > .55)
            labels = ndi.label(mask.mean(axis=0) > .55)[0]
            longest = np.bincount(labels)[1:].max(initial=0) if label_runs else 0
            along = np.abs(np.diff(stripe, axis=1)).mean()
            across = np.abs(np.diff(stripe, axis=0)).mean()
            stats.append([mask.mean(), longest/128, stripe.mean()-q50, stripe.std(),
                          g.mean(), across/(along+1e-4)])
        edge_stats.extend(np.r_[np.max(stats, axis=0), np.mean(stats, axis=0), np.min(stats, axis=0)])
    return np.asarray([*proposed, *edge_stats, q25-q10, q90-q10, dark.mean()], dtype=np.float32)


def extract_row(path):
    base, _ = image_features(path)
    ring_blocks = blocks(path)
    return np.concatenate([
        *(np.asarray(block, dtype=np.float32) for block in base),
        *(np.asarray(ring_blocks[key], dtype=np.float32) for key in BLOCKS),
        targeted_features(path),
    ])
