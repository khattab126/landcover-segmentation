"""Hardcoded experiment results transcribed from assessment.ipynb output."""

from model import CLASS_NAMES

# ---------------------------------------------------------------------------
# Factor 1: Annotation budget sweep (balanced sampling, focal gamma=2)
# Plus full-supervision upper bound.
# ---------------------------------------------------------------------------

FACTOR1_HISTORY = {
    "pfCE_balanced_ppc1": [
        {"epoch": 1, "loss": 1.0335, "mIoU": 0.1057, "pixel_acc": 0.5800},
        {"epoch": 2, "loss": 0.8759, "mIoU": 0.1893, "pixel_acc": 0.5259},
        {"epoch": 3, "loss": 0.9000, "mIoU": 0.2081, "pixel_acc": 0.5641},
        {"epoch": 4, "loss": 0.8258, "mIoU": 0.1541, "pixel_acc": 0.3685},
        {"epoch": 5, "loss": 0.8403, "mIoU": 0.2098, "pixel_acc": 0.4871},
        {"epoch": 6, "loss": 0.8166, "mIoU": 0.2090, "pixel_acc": 0.5346},
        {"epoch": 7, "loss": 0.7795, "mIoU": 0.2206, "pixel_acc": 0.5715},
        {"epoch": 8, "loss": 0.8255, "mIoU": 0.1747, "pixel_acc": 0.4569},
        {"epoch": 9, "loss": 0.8013, "mIoU": 0.1697, "pixel_acc": 0.3818},
        {"epoch": 10, "loss": 0.7763, "mIoU": 0.1282, "pixel_acc": 0.2849},
        {"epoch": 11, "loss": 0.7831, "mIoU": 0.1953, "pixel_acc": 0.4635},
        {"epoch": 12, "loss": 0.7767, "mIoU": 0.2491, "pixel_acc": 0.5743},
    ],
    "pfCE_balanced_ppc5": [
        {"epoch": 1, "loss": 0.9790, "mIoU": 0.0971, "pixel_acc": 0.5782},
        {"epoch": 2, "loss": 0.8403, "mIoU": 0.2160, "pixel_acc": 0.5427},
        {"epoch": 3, "loss": 0.8222, "mIoU": 0.2219, "pixel_acc": 0.5883},
        {"epoch": 4, "loss": 0.7944, "mIoU": 0.1737, "pixel_acc": 0.4311},
        {"epoch": 5, "loss": 0.7682, "mIoU": 0.1733, "pixel_acc": 0.4221},
        {"epoch": 6, "loss": 0.7797, "mIoU": 0.2407, "pixel_acc": 0.5299},
        {"epoch": 7, "loss": 0.7543, "mIoU": 0.2427, "pixel_acc": 0.5452},
        {"epoch": 8, "loss": 0.7379, "mIoU": 0.2194, "pixel_acc": 0.5004},
        {"epoch": 9, "loss": 0.7412, "mIoU": 0.1611, "pixel_acc": 0.3592},
        {"epoch": 10, "loss": 0.7339, "mIoU": 0.2537, "pixel_acc": 0.5910},
        {"epoch": 11, "loss": 0.7257, "mIoU": 0.1830, "pixel_acc": 0.3633},
        {"epoch": 12, "loss": 0.7116, "mIoU": 0.2085, "pixel_acc": 0.4898},
    ],
    "pfCE_balanced_ppc10": [
        {"epoch": 1, "loss": 0.9707, "mIoU": 0.0972, "pixel_acc": 0.5783},
        {"epoch": 2, "loss": 0.8401, "mIoU": 0.2241, "pixel_acc": 0.6109},
        {"epoch": 3, "loss": 0.8059, "mIoU": 0.2462, "pixel_acc": 0.6144},
        {"epoch": 4, "loss": 0.7723, "mIoU": 0.2091, "pixel_acc": 0.4944},
        {"epoch": 5, "loss": 0.7502, "mIoU": 0.2121, "pixel_acc": 0.5027},
        {"epoch": 6, "loss": 0.7644, "mIoU": 0.2710, "pixel_acc": 0.5736},
        {"epoch": 7, "loss": 0.7396, "mIoU": 0.2383, "pixel_acc": 0.5793},
        {"epoch": 8, "loss": 0.7220, "mIoU": 0.2351, "pixel_acc": 0.5434},
        {"epoch": 9, "loss": 0.7288, "mIoU": 0.1036, "pixel_acc": 0.1578},
        {"epoch": 10, "loss": 0.7175, "mIoU": 0.2534, "pixel_acc": 0.6195},
        {"epoch": 11, "loss": 0.7033, "mIoU": 0.2062, "pixel_acc": 0.3733},
        {"epoch": 12, "loss": 0.7059, "mIoU": 0.2313, "pixel_acc": 0.4878},
    ],
    "pfCE_balanced_ppc20": [
        {"epoch": 1, "loss": 0.9705, "mIoU": 0.0964, "pixel_acc": 0.5782},
        {"epoch": 2, "loss": 0.8403, "mIoU": 0.2252, "pixel_acc": 0.6036},
        {"epoch": 3, "loss": 0.8092, "mIoU": 0.2248, "pixel_acc": 0.6151},
        {"epoch": 4, "loss": 0.7751, "mIoU": 0.2286, "pixel_acc": 0.5390},
        {"epoch": 5, "loss": 0.7628, "mIoU": 0.1448, "pixel_acc": 0.3286},
        {"epoch": 6, "loss": 0.7643, "mIoU": 0.2636, "pixel_acc": 0.5510},
        {"epoch": 7, "loss": 0.7388, "mIoU": 0.2160, "pixel_acc": 0.4362},
        {"epoch": 8, "loss": 0.7259, "mIoU": 0.2264, "pixel_acc": 0.5466},
        {"epoch": 9, "loss": 0.7247, "mIoU": 0.1045, "pixel_acc": 0.1949},
        {"epoch": 10, "loss": 0.7069, "mIoU": 0.2756, "pixel_acc": 0.6185},
        {"epoch": 11, "loss": 0.7039, "mIoU": 0.2512, "pixel_acc": 0.4768},
        {"epoch": 12, "loss": 0.6929, "mIoU": 0.2157, "pixel_acc": 0.4365},
    ],
    "pfCE_balanced_ppc50": [
        {"epoch": 1, "loss": 0.9568, "mIoU": 0.0973, "pixel_acc": 0.5783},
        {"epoch": 2, "loss": 0.8421, "mIoU": 0.2327, "pixel_acc": 0.6206},
        {"epoch": 3, "loss": 0.8049, "mIoU": 0.2326, "pixel_acc": 0.6019},
        {"epoch": 4, "loss": 0.7719, "mIoU": 0.1951, "pixel_acc": 0.3989},
        {"epoch": 5, "loss": 0.7536, "mIoU": 0.2047, "pixel_acc": 0.5019},
        {"epoch": 6, "loss": 0.7547, "mIoU": 0.2773, "pixel_acc": 0.5919},
        {"epoch": 7, "loss": 0.7323, "mIoU": 0.1938, "pixel_acc": 0.4462},
        {"epoch": 8, "loss": 0.7218, "mIoU": 0.2464, "pixel_acc": 0.5216},
        {"epoch": 9, "loss": 0.7161, "mIoU": 0.1606, "pixel_acc": 0.3029},
        {"epoch": 10, "loss": 0.6985, "mIoU": 0.2753, "pixel_acc": 0.6141},
        {"epoch": 11, "loss": 0.7019, "mIoU": 0.3021, "pixel_acc": 0.5404},
        {"epoch": 12, "loss": 0.6885, "mIoU": 0.2871, "pixel_acc": 0.5851},
    ],
    "full_supervision": [
        {"epoch": 1, "loss": 1.4138, "mIoU": 0.0976, "pixel_acc": 0.5780},
        {"epoch": 2, "loss": 1.1900, "mIoU": 0.1491, "pixel_acc": 0.5806},
        {"epoch": 3, "loss": 1.1197, "mIoU": 0.1299, "pixel_acc": 0.5892},
        {"epoch": 4, "loss": 1.0547, "mIoU": 0.1458, "pixel_acc": 0.4192},
        {"epoch": 5, "loss": 1.0548, "mIoU": 0.2129, "pixel_acc": 0.6167},
        {"epoch": 6, "loss": 1.0256, "mIoU": 0.1890, "pixel_acc": 0.6108},
        {"epoch": 7, "loss": 0.9746, "mIoU": 0.1844, "pixel_acc": 0.4157},
        {"epoch": 8, "loss": 0.9524, "mIoU": 0.1892, "pixel_acc": 0.6134},
        {"epoch": 9, "loss": 0.9278, "mIoU": 0.2202, "pixel_acc": 0.4764},
        {"epoch": 10, "loss": 0.9440, "mIoU": 0.2291, "pixel_acc": 0.6286},
        {"epoch": 11, "loss": 0.9420, "mIoU": 0.2276, "pixel_acc": 0.6207},
        {"epoch": 12, "loss": 0.9333, "mIoU": 0.2222, "pixel_acc": 0.5327},
    ],
}

FACTOR1_SUMMARY = [
    {"config": "1 point/class", "best_mIoU": 0.2491, "best_epoch": 12, "best_pixel_acc": 0.5743},
    {"config": "5 points/class", "best_mIoU": 0.2537, "best_epoch": 10, "best_pixel_acc": 0.5910},
    {"config": "10 points/class", "best_mIoU": 0.2710, "best_epoch": 6, "best_pixel_acc": 0.5736},
    {"config": "20 points/class", "best_mIoU": 0.2756, "best_epoch": 10, "best_pixel_acc": 0.6185},
    {"config": "50 points/class", "best_mIoU": 0.3021, "best_epoch": 11, "best_pixel_acc": 0.5404},
    {"config": "Full supervision", "best_mIoU": 0.2291, "best_epoch": 10, "best_pixel_acc": 0.6286},
]

# ---------------------------------------------------------------------------
# Factor 2: Loss x Sampling (fixed budget = 100 points/image)
# ---------------------------------------------------------------------------

FACTOR2_HISTORY = {
    "pCE_balanced": [
        {"epoch": 1, "loss": 1.5080, "mIoU": 0.1191, "pixel_acc": 0.5844},
        {"epoch": 2, "loss": 1.3666, "mIoU": 0.1752, "pixel_acc": 0.5209},
        {"epoch": 3, "loss": 1.3389, "mIoU": 0.2271, "pixel_acc": 0.6138},
        {"epoch": 4, "loss": 1.2912, "mIoU": 0.1492, "pixel_acc": 0.3259},
        {"epoch": 5, "loss": 1.2637, "mIoU": 0.1909, "pixel_acc": 0.4306},
        {"epoch": 6, "loss": 1.2699, "mIoU": 0.2683, "pixel_acc": 0.5580},
        {"epoch": 7, "loss": 1.2355, "mIoU": 0.2377, "pixel_acc": 0.5597},
        {"epoch": 8, "loss": 1.2168, "mIoU": 0.2260, "pixel_acc": 0.5273},
        {"epoch": 9, "loss": 1.2086, "mIoU": 0.2008, "pixel_acc": 0.4353},
        {"epoch": 10, "loss": 1.1928, "mIoU": 0.2231, "pixel_acc": 0.5620},
        {"epoch": 11, "loss": 1.1955, "mIoU": 0.2812, "pixel_acc": 0.5483},
        {"epoch": 12, "loss": 1.1762, "mIoU": 0.2985, "pixel_acc": 0.6235},
    ],
    "pfCE_balanced": [
        {"epoch": 1, "loss": 0.9705, "mIoU": 0.0964, "pixel_acc": 0.5782},
        {"epoch": 2, "loss": 0.8403, "mIoU": 0.2252, "pixel_acc": 0.6036},
        {"epoch": 3, "loss": 0.8092, "mIoU": 0.2248, "pixel_acc": 0.6151},
        {"epoch": 4, "loss": 0.7751, "mIoU": 0.2286, "pixel_acc": 0.5390},
        {"epoch": 5, "loss": 0.7628, "mIoU": 0.1448, "pixel_acc": 0.3286},
        {"epoch": 6, "loss": 0.7643, "mIoU": 0.2636, "pixel_acc": 0.5510},
        {"epoch": 7, "loss": 0.7388, "mIoU": 0.2160, "pixel_acc": 0.4362},
        {"epoch": 8, "loss": 0.7259, "mIoU": 0.2264, "pixel_acc": 0.5466},
        {"epoch": 9, "loss": 0.7247, "mIoU": 0.1045, "pixel_acc": 0.1949},
        {"epoch": 10, "loss": 0.7069, "mIoU": 0.2756, "pixel_acc": 0.6185},
        {"epoch": 11, "loss": 0.7039, "mIoU": 0.2512, "pixel_acc": 0.4768},
        {"epoch": 12, "loss": 0.6929, "mIoU": 0.2157, "pixel_acc": 0.4365},
    ],
    "pCE_uniform": [
        {"epoch": 1, "loss": 1.4247, "mIoU": 0.0976, "pixel_acc": 0.5780},
        {"epoch": 2, "loss": 1.1908, "mIoU": 0.1111, "pixel_acc": 0.5847},
        {"epoch": 3, "loss": 1.1152, "mIoU": 0.1853, "pixel_acc": 0.5710},
        {"epoch": 4, "loss": 1.0751, "mIoU": 0.1928, "pixel_acc": 0.5962},
        {"epoch": 5, "loss": 1.0064, "mIoU": 0.2501, "pixel_acc": 0.6032},
        {"epoch": 6, "loss": 1.0061, "mIoU": 0.1560, "pixel_acc": 0.5850},
        {"epoch": 7, "loss": 0.9814, "mIoU": 0.1838, "pixel_acc": 0.5197},
        {"epoch": 8, "loss": 0.9476, "mIoU": 0.2171, "pixel_acc": 0.5835},
        {"epoch": 9, "loss": 0.9408, "mIoU": 0.1733, "pixel_acc": 0.5975},
        {"epoch": 10, "loss": 0.9425, "mIoU": 0.2500, "pixel_acc": 0.6338},
        {"epoch": 11, "loss": 0.9034, "mIoU": 0.1911, "pixel_acc": 0.4892},
        {"epoch": 12, "loss": 0.9192, "mIoU": 0.1541, "pixel_acc": 0.2967},
    ],
    "pfCE_uniform": [
        {"epoch": 1, "loss": 0.8720, "mIoU": 0.0975, "pixel_acc": 0.5781},
        {"epoch": 2, "loss": 0.6726, "mIoU": 0.1221, "pixel_acc": 0.5910},
        {"epoch": 3, "loss": 0.6283, "mIoU": 0.1325, "pixel_acc": 0.5600},
        {"epoch": 4, "loss": 0.6043, "mIoU": 0.2004, "pixel_acc": 0.5226},
        {"epoch": 5, "loss": 0.5691, "mIoU": 0.1940, "pixel_acc": 0.6053},
        {"epoch": 6, "loss": 0.5541, "mIoU": 0.2399, "pixel_acc": 0.6151},
        {"epoch": 7, "loss": 0.5401, "mIoU": 0.2048, "pixel_acc": 0.4916},
        {"epoch": 8, "loss": 0.5215, "mIoU": 0.2915, "pixel_acc": 0.6693},
        {"epoch": 9, "loss": 0.5331, "mIoU": 0.2403, "pixel_acc": 0.6169},
        {"epoch": 10, "loss": 0.5182, "mIoU": 0.2295, "pixel_acc": 0.5701},
        {"epoch": 11, "loss": 0.5092, "mIoU": 0.3038, "pixel_acc": 0.6434},
        {"epoch": 12, "loss": 0.5002, "mIoU": 0.2796, "pixel_acc": 0.6400},
    ],
}

FACTOR2_SUMMARY = [
    {"config": "pCE + Balanced", "best_mIoU": 0.2985, "best_epoch": 12, "best_pixel_acc": 0.6235},
    {"config": "pfCE + Balanced", "best_mIoU": 0.2756, "best_epoch": 10, "best_pixel_acc": 0.6185},
    {"config": "pCE + Uniform", "best_mIoU": 0.2501, "best_epoch": 5, "best_pixel_acc": 0.6032},
    {"config": "pfCE + Uniform", "best_mIoU": 0.3038, "best_epoch": 11, "best_pixel_acc": 0.6434},
]

# Per-class IoU for Factor 2 configs (final epoch)
FACTOR2_PER_CLASS_IOU = {
    "pCE + Balanced":  {"urban": 0.31, "agriculture": 0.63, "rangeland": 0.15, "forest": 0.25, "water": 0.22, "barren": 0.23},
    "pfCE + Balanced": {"urban": 0.27, "agriculture": 0.48, "rangeland": 0.12, "forest": 0.21, "water": 0.18, "barren": 0.19},
    "pCE + Uniform":   {"urban": 0.18, "agriculture": 0.32, "rangeland": 0.08, "forest": 0.15, "water": 0.10, "barren": 0.12},
    "pfCE + Uniform":  {"urban": 0.33, "agriculture": 0.59, "rangeland": 0.19, "forest": 0.28, "water": 0.30, "barren": 0.27},
}
