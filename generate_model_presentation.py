"""Generate technical ML Model deep-dive presentation."""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

PURPLE_DARK  = RGBColor(0x1E, 0x1B, 0x4B)
PURPLE_MID   = RGBColor(0x7C, 0x3A, 0xED)
PURPLE_LIGHT = RGBColor(0xED, 0xE9, 0xFE)
WHITE        = RGBColor(0xFF, 0xFF, 0xFF)
GRAY_DARK    = RGBColor(0x1F, 0x29, 0x37)
GRAY_MID     = RGBColor(0x6B, 0x72, 0x80)
GRAY_LIGHT   = RGBColor(0xF3, 0xF4, 0xF6)
CODE_BG      = RGBColor(0x0F, 0x17, 0x2A)
CODE_FG      = RGBColor(0x93, 0xC5, 0xFD)
CODE_GREEN   = RGBColor(0x6E, 0xE7, 0xB7)
CODE_YELLOW  = RGBColor(0xFD, 0xE6, 0x8A)
CODE_COMMENT = RGBColor(0x6B, 0x72, 0x80)
GREEN        = RGBColor(0x05, 0x96, 0x69)
ORANGE       = RGBColor(0xEA, 0x58, 0x0C)
RED          = RGBColor(0xDC, 0x26, 0x26)
BLUE         = RGBColor(0x25, 0x63, 0xEB)
TEAL         = RGBColor(0x0D, 0x94, 0x88)
CYAN         = RGBColor(0x06, 0xB6, 0xD4)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)


def new_prs():
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def fill_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def rect(slide, l, t, w, h, color, line_color=None):
    s = slide.shapes.add_shape(1, l, t, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = color
    if line_color:
        s.line.color.rgb = line_color
        s.line.width = Pt(1)
    else:
        s.line.fill.background()
    return s


def txt(slide, text, l, t, w, h, size=14, bold=False,
        color=GRAY_DARK, align=PP_ALIGN.LEFT, italic=False, font="Calibri"):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    r.font.name = font
    return tb


def code_block(slide, lines, l, t, w, h, size=11):
    """Render a code-like block with dark background."""
    rect(slide, l, t, w, h, CODE_BG)
    tb = slide.shapes.add_textbox(
        l + Inches(0.15), t + Inches(0.12),
        w - Inches(0.3), h - Inches(0.2)
    )
    tf = tb.text_frame
    tf.word_wrap = False
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.color.rgb = CODE_FG
        r.font.name = "Consolas"


def slide_header(slide, title, subtitle=None, accent=PURPLE_MID):
    rect(slide, 0, 0, SLIDE_W, Inches(1.0), PURPLE_DARK)
    rect(slide, 0, Inches(1.0), SLIDE_W, Inches(0.05), accent)
    txt(slide, title,
        Inches(0.5), Inches(0.15), Inches(12.3), Inches(0.6),
        size=26, bold=True, color=WHITE)
    if subtitle:
        txt(slide, subtitle,
            Inches(0.5), Inches(0.68), Inches(12.3), Inches(0.35),
            size=12, color=RGBColor(0xA5, 0xB4, 0xFC))


def shape_box(slide, l, t, w, h, bg, title, title_color=WHITE,
              title_size=13, body=None, body_size=11, body_color=WHITE):
    rect(slide, l, t, w, Inches(0.38), bg)
    txt(slide, title, l + Inches(0.1), t + Inches(0.04),
        w - Inches(0.2), Inches(0.3),
        size=title_size, bold=True, color=title_color, align=PP_ALIGN.CENTER)
    if body:
        txt(slide, body, l + Inches(0.08), t + Inches(0.42),
            w - Inches(0.16), h - Inches(0.45),
            size=body_size, color=body_color)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 1: Cover
# ─────────────────────────────────────────────────────────────────────────────
def slide_01_cover(prs):
    sl = blank_slide(prs)
    fill_bg(sl, PURPLE_DARK)
    rect(sl, 0, Inches(3.5), SLIDE_W, Inches(0.07), PURPLE_MID)
    rect(sl, 0, Inches(6.5), SLIDE_W, Inches(0.07), PURPLE_MID)

    txt(sl, "Deep-Dive: Arsitektur Model ML",
        Inches(0.8), Inches(0.9), Inches(11.7), Inches(0.9),
        size=40, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    txt(sl, "Air Quality Forecast System",
        Inches(0.8), Inches(1.85), Inches(11.7), Inches(0.55),
        size=22, color=RGBColor(0xA5, 0xB4, 0xFC), align=PP_ALIGN.CENTER)

    tags = [
        ("Preprocessing", BLUE),
        ("SSA Decomposition", TEAL),
        ("BiLSTM", PURPLE_MID),
        ("LightGBM Ensemble", GREEN),
        ("Quantile Regression", ORANGE),
    ]
    total_w = Inches(10.5)
    spacing = total_w / len(tags)
    for i, (tag, color) in enumerate(tags):
        x = Inches(1.4) + int(i * spacing)
        rect(sl, x, Inches(4.0), Inches(1.9), Inches(0.4), color)
        txt(sl, tag, x, Inches(4.04), Inches(1.9), Inches(0.32),
            size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    txt(sl, "Python 3.11  ·  TensorFlow/Keras  ·  LightGBM  ·  scikit-learn  ·  NumPy",
        Inches(0.8), Inches(4.7), Inches(11.7), Inches(0.4),
        size=13, color=RGBColor(0x6B, 0x72, 0x80), align=PP_ALIGN.CENTER)

    txt(sl, "Input: 24-jam × 14 sensor features  →  Output: 6-step × 14 predicted values + confidence band",
        Inches(0.8), Inches(5.2), Inches(11.7), Inches(0.4),
        size=14, bold=True, color=RGBColor(0xC4, 0xB5, 0xFD), align=PP_ALIGN.CENTER)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 2: Pipeline Overview
# ─────────────────────────────────────────────────────────────────────────────
def slide_02_pipeline(prs):
    sl = blank_slide(prs)
    fill_bg(sl, WHITE)
    slide_header(sl, "Inference Pipeline — Alur Lengkap dari Sensor ke Prediksi")

    steps = [
        (BLUE,       "1  fetch_sensor_data()",    "t_loggers\nN_INPUT_HOURS × 3 jam"),
        (CYAN,       "2  resample('h').mean()",   "Rata-rata per jam\n→ (72, 14)"),
        (TEAL,       "3  detect_anomalies()",     "Rolling z-score\n3σ → NaN → interp"),
        (GREEN,      "4  clean()",                "Linear interp +\nIQR clip [Q1-1.5IQR]"),
        (ORANGE,     "5  add_time_features()",    "sin/cos hour, dow\n→ (24, 18)"),
        (PURPLE_MID, "6  normalize(fit=False)",   "MinMaxScaler per-UID\n→ [0, 1]"),
        (RED,        "7  BiLSTM.predict()",       "_prepare_input → SSA\n→ (1, 6, 14) normed"),
        (RGBColor(0x7E,0x22,0xCE), "8  LightGBM.predict()",  "3 models: point,\nlower, upper"),
        (GRAY_MID,   "9  denormalize()",          "inverse_transform\n→ nilai asli"),
    ]

    for i, (color, name, note) in enumerate(steps):
        x = Inches(0.35 + i * 1.41)
        rect(sl, x, Inches(1.4), Inches(1.25), Inches(0.42), color)
        txt(sl, name, x + Inches(0.05), Inches(1.42),
            Inches(1.15), Inches(0.38), size=9, bold=True, color=WHITE)
        rect(sl, x, Inches(1.82), Inches(1.25), Inches(1.0), GRAY_LIGHT)
        txt(sl, note, x + Inches(0.05), Inches(1.88),
            Inches(1.15), Inches(0.9), size=9, color=GRAY_DARK)
        if i < len(steps) - 1:
            txt(sl, "→", x + Inches(1.25), Inches(1.53),
                Inches(0.2), Inches(0.3), size=14, bold=True,
                color=GRAY_MID, align=PP_ALIGN.CENTER)

    # Shape annotations
    rect(sl, Inches(0.35), Inches(3.1), SLIDE_W - Inches(0.7), Inches(0.04), PURPLE_LIGHT)

    shapes = [
        (Inches(0.35),  "fetch raw\n(N, 14)"),
        (Inches(1.76),  "hourly\n(72, 14)"),
        (Inches(3.17),  "anomaly\nfixed"),
        (Inches(4.58),  "clean\n(72, 14)"),
        (Inches(5.99),  "timed\n(24, 18)"),
        (Inches(7.40),  "normed\n(1,24,18)"),
        (Inches(8.81),  "bilstm out\n(1,6,14)"),
        (Inches(10.22), "lgbm in\n(6,17)"),
        (Inches(11.63), "final\n(6,14)×3"),
    ]
    for x, label in shapes:
        txt(sl, label, x, Inches(3.2), Inches(1.25), Inches(0.55),
            size=9, color=PURPLE_MID, align=PP_ALIGN.CENTER, italic=True)

    # Config constants
    rect(sl, Inches(0.35), Inches(4.0), SLIDE_W - Inches(0.7), Inches(3.1), CODE_BG)
    code_block(sl, [
        "# config.py — Konstanta model",
        "N_INPUT_HOURS      = 24        # panjang jendela input (jam)",
        "N_FORECAST_HOURS   = 6         # jumlah langkah prediksi ke depan",
        "N_FEATURES         = 14        # jumlah kolom sensor (FEATURE_COLS)",
        "N_TIME_FEATURES    = 4         # hour_sin, hour_cos, dow_sin, dow_cos",
        "SSA_WINDOW         = 12        # window decomposisi SSA",
        "BILSTM_UNITS       = 128       # unit per arah BiLSTM layer pertama",
        "TRAIN_HISTORY_HOURS = 24*365   # panjang data historis untuk training",
        "",
        "FEATURE_COLS = ['pm_25','pm_25_correction','pm_10','pm_10_correction',",
        "                'tsp','tsp_correction','noise','temp','mmhg','humidity',",
        "                'aqi_index_pm25','aqi_index_pm10','aqi_index_tsp','aqi_index']",
    ], Inches(0.35), Inches(4.0), SLIDE_W - Inches(0.7), Inches(3.1), size=11)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 3: Preprocessing Detail
# ─────────────────────────────────────────────────────────────────────────────
def slide_03_preprocessing(prs):
    sl = blank_slide(prs)
    fill_bg(sl, WHITE)
    slide_header(sl, "Preprocessing — 5 Tahap Sebelum Model Menerima Data", accent=BLUE)

    # Left column
    # 1. Resample
    rect(sl, Inches(0.3), Inches(1.2), Inches(6.0), Inches(0.38), BLUE)
    txt(sl, "① resample_hourly()  — Data sensor → rata-rata per jam",
        Inches(0.45), Inches(1.23), Inches(5.7), Inches(0.32),
        size=12, bold=True, color=WHITE)
    code_block(sl, [
        "df['timestamp'] = pd.to_datetime(df['datetime_unix'], unit='s')",
        "df.set_index('timestamp')[FEATURE_COLS].resample('h').mean()",
        "# Input bisa menit-an; output selalu per jam → shape (N, 14)",
    ], Inches(0.3), Inches(1.58), Inches(6.0), Inches(0.8), size=10)

    # 2. Anomaly
    rect(sl, Inches(0.3), Inches(2.48), Inches(6.0), Inches(0.38), TEAL)
    txt(sl, "② detect_and_remove_anomalies()  — Rolling z-score",
        Inches(0.45), Inches(2.51), Inches(5.7), Inches(0.32),
        size=12, bold=True, color=WHITE)
    code_block(sl, [
        "rolling_median = df[col].rolling(24, center=True).median()",
        "rolling_std    = df[col].rolling(24, center=True).std()",
        "anomaly_mask   = |df[col] - median| > 3 * std",
        "df[anomaly_mask] = NaN  →  interpolate(linear).ffill().bfill()",
    ], Inches(0.3), Inches(2.86), Inches(6.0), Inches(1.0), size=10)

    # 3. Clean
    rect(sl, Inches(0.3), Inches(3.96), Inches(6.0), Inches(0.38), GREEN)
    txt(sl, "③ clean()  — Interpolasi + IQR Clipping",
        Inches(0.45), Inches(3.99), Inches(5.7), Inches(0.32),
        size=12, bold=True, color=WHITE)
    code_block(sl, [
        "df = df.interpolate('linear').ffill().bfill()",
        "# IQR clipping per kolom",
        "lower = Q1 - 1.5 × IQR",
        "upper = Q3 + 1.5 × IQR",
        "df[col] = df[col].clip(lower, upper)",
    ], Inches(0.3), Inches(4.34), Inches(6.0), Inches(1.15), size=10)

    # Right column
    # 4. Time features
    rect(sl, Inches(6.7), Inches(1.2), Inches(6.3), Inches(0.38), ORANGE)
    txt(sl, "④ add_time_features()  — Cyclical Encoding",
        Inches(6.85), Inches(1.23), Inches(6.0), Inches(0.32),
        size=12, bold=True, color=WHITE)
    code_block(sl, [
        "# Encode hour & dayofweek sebagai sin/cos",
        "# supaya jam 23 dan jam 0 'berdekatan'",
        "hour_sin = sin(2π × hour / 24)",
        "hour_cos = cos(2π × hour / 24)",
        "dow_sin  = sin(2π × dayofweek / 7)",
        "dow_cos  = cos(2π × dayofweek / 7)",
        "# Hasil: (N, 14) → (N, 18)",
    ], Inches(6.7), Inches(1.58), Inches(6.3), Inches(1.35), size=10)

    # 5. Normalize
    rect(sl, Inches(6.7), Inches(3.03), Inches(6.3), Inches(0.38), PURPLE_MID)
    txt(sl, "⑤ normalize()  — MinMaxScaler per-UID",
        Inches(6.85), Inches(3.06), Inches(6.0), Inches(0.32),
        size=12, bold=True, color=WHITE)
    code_block(sl, [
        "# fit=True  saat training → simpan scaler",
        "# fit=False saat predict  → load scaler",
        "scaler = MinMaxScaler()",
        "scaled = scaler.fit_transform(df[FEATURE_COLS])",
        "# TIME_COLS (sin/cos) tidak di-scale → pass through",
        "# Disimpan: saved_models/{uid}/scaler.pkl",
        "# Denormalize: scaler.inverse_transform(arr)",
    ], Inches(6.7), Inches(3.41), Inches(6.3), Inches(1.35), size=10)

    # Bottom: Why per-UID scaler
    rect(sl, Inches(6.7), Inches(4.86), Inches(6.3), Inches(0.85), PURPLE_LIGHT)
    txt(sl, "⚠  Mengapa scaler per-UID?",
        Inches(6.85), Inches(4.9), Inches(5.9), Inches(0.32),
        size=11, bold=True, color=PURPLE_DARK)
    txt(sl, "Setiap sensor memiliki range nilai berbeda (sensor di jalan raya vs. hutan). "
            "Satu scaler global akan mendistorsi distribusi data sensor dengan range rendah.",
        Inches(6.85), Inches(5.22), Inches(5.9), Inches(0.45),
        size=10, color=GRAY_DARK)

    # Sequence creation
    rect(sl, Inches(0.3), Inches(5.59), Inches(6.0), Inches(0.38), GRAY_DARK)
    txt(sl, "create_sequences()  — Sliding Window untuk Training",
        Inches(0.45), Inches(5.62), Inches(5.7), Inches(0.32),
        size=12, bold=True, color=WHITE)
    code_block(sl, [
        "# X: (N, 24, 18)  — input sequences",
        "# y: (N,  6, 14)  — target (14 sensor, bukan time features)",
        "for i in range(len(data) - 24 - 6 + 1):",
        "    X[i] = data[i : i+24]",
        "    y[i] = data[i+24 : i+30, :14]  # n_target_cols=14",
    ], Inches(0.3), Inches(5.97), Inches(6.0), Inches(1.1), size=10)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 4: SSA
# ─────────────────────────────────────────────────────────────────────────────
def slide_04_ssa(prs):
    sl = blank_slide(prs)
    fill_bg(sl, WHITE)
    slide_header(sl, "SSA — Singular Spectrum Analysis  (ssa.py)", accent=TEAL)

    # Motivation
    rect(sl, Inches(0.3), Inches(1.15), Inches(12.73), Inches(0.65), PURPLE_LIGHT)
    txt(sl, "Tujuan: Dekomposisi setiap deret waktu sensor menjadi komponen trend dan osilasi "
            "sebelum dimasukkan ke BiLSTM, sehingga model belajar dua pola secara terpisah.",
        Inches(0.45), Inches(1.18), Inches(12.43), Inches(0.58),
        size=12, color=PURPLE_DARK)

    # Algorithm steps
    rect(sl, Inches(0.3), Inches(1.9), Inches(6.2), Inches(0.38), TEAL)
    txt(sl, "Algoritma SSA — Step by Step (SSA_WINDOW = 12)",
        Inches(0.45), Inches(1.93), Inches(5.9), Inches(0.32),
        size=12, bold=True, color=WHITE)
    code_block(sl, [
        "# 1. Embedding — series (n,) → trajectory matrix",
        "K = n - window + 1           # K = 24 - 12 + 1 = 13",
        "trajectory = [[s[i:i+12]] for i in range(K)]  # shape (13, 12)",
        "",
        "# 2. SVD — faktorisasi matriks trajectory",
        "U, sigma, Vt = np.linalg.svd(trajectory)      # full SVD",
        "",
        "# 3. Grouping — ambil komponen dominan pertama sebagai TREND",
        "rank1 = sigma[0] * outer(U[:,0], Vt[0,:])     # shape (13, 12)",
        "",
        "# 4. Diagonal Averaging — konversi matriks → series",
        "trend = diagonal_average(rank1, window=12, n=24)  # shape (24,)",
        "",
        "# 5. Osilasi = residu",
        "oscillation = series - trend                   # shape (24,)",
    ], Inches(0.3), Inches(2.28), Inches(6.2), Inches(3.4), size=10)

    # apply_ssa_to_dataframe
    rect(sl, Inches(0.3), Inches(5.78), Inches(6.2), Inches(0.38), TEAL)
    txt(sl, "apply_ssa_to_dataframe()  — Diterapkan ke 14 fitur sekaligus",
        Inches(0.45), Inches(5.81), Inches(5.9), Inches(0.32),
        size=12, bold=True, color=WHITE)
    code_block(sl, [
        "# Input:  (24, 14)   — 1 sample × 24 jam × 14 sensor",
        "# Output: (24, 28)   — trend[0..13] + oscillation[14..27]",
        "for i in range(14):",
        "    trend, osc = ssa_decompose(data[:, i])",
        "    output[:, i]    = trend",
        "    output[:, i+14] = osc",
    ], Inches(0.3), Inches(6.16), Inches(6.2), Inches(1.0), size=10)

    # Right — diagram & concat
    rect(sl, Inches(6.8), Inches(1.9), Inches(6.23), Inches(0.38), GRAY_DARK)
    txt(sl, "Transformasi Dimensi Sebelum BiLSTM",
        Inches(6.95), Inches(1.93), Inches(5.9), Inches(0.32),
        size=12, bold=True, color=WHITE)

    rows = [
        (BLUE,       "Raw sensor input",  "shape: (1, 24, 14)  — 14 fitur sensor ter-normalize"),
        (TEAL,       "Pisah sensor & time","X_sensor = X[:,:,:14]  →  (1,24,14)\nX_time   = X[:,:,14:]  →  (1,24,4)"),
        (TEAL,       "apply_ssa() per sample","X_ssa = [apply_ssa(x) for x in X_sensor]\n→  (1, 24, 28)   ← trend + osilasi"),
        (PURPLE_MID, "Concatenate",       "np.concatenate([X_ssa, X_time], axis=2)\n→  (1, 24, 32)   ← input final BiLSTM"),
        (GREEN,      "N_SSA_FEATURES",    "N_FEATURES*2 + N_TIME_FEATURES\n= 14×2 + 4 = 32"),
    ]
    for i, (color, title, desc) in enumerate(rows):
        y = Inches(2.38 + i * 0.96)
        rect(sl, Inches(6.8), y, Inches(2.1), Inches(0.82), color)
        txt(sl, title, Inches(6.9), y + Inches(0.05),
            Inches(1.9), Inches(0.72), size=10, bold=True, color=WHITE)
        rect(sl, Inches(8.9), y, Inches(4.13), Inches(0.82), GRAY_LIGHT)
        txt(sl, desc, Inches(9.0), y + Inches(0.05),
            Inches(3.93), Inches(0.72), size=10, color=GRAY_DARK,
            font="Consolas")
        if i < len(rows) - 1:
            txt(sl, "↓", Inches(6.8), y + Inches(0.82),
                Inches(2.1), Inches(0.18), size=11, bold=True,
                color=TEAL, align=PP_ALIGN.CENTER)

    # Key insight box
    rect(sl, Inches(6.8), Inches(7.0), Inches(6.23), Inches(0.42), RGBColor(0xEC, 0xFD, 0xF5))
    txt(sl, "💡  Komponen pertama SVD (σ₀) menangkap variansi terbesar deret waktu → trend dominan. "
            "Residu (osilasi) merepresentasikan siklus harian & noise terstruktur.",
        Inches(6.9), Inches(7.04), Inches(6.03), Inches(0.38),
        size=9, color=GREEN)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 5: BiLSTM Architecture
# ─────────────────────────────────────────────────────────────────────────────
def slide_05_bilstm(prs):
    sl = blank_slide(prs)
    fill_bg(sl, WHITE)
    slide_header(sl, "BiLSTM — Bidirectional LSTM Network  (bilstm.py)", accent=PURPLE_MID)

    # Layer diagram
    layers = [
        (BLUE,       "Input",                   "(batch, 24, 32)",  "24 timestep, 32 fitur (SSA+time)"),
        (PURPLE_MID, "Bidirectional LSTM(128)",  "(batch, 24, 256)", "128 unit × 2 arah, return_sequences=True"),
        (GRAY_MID,   "Dropout(0.2)",             "(batch, 24, 256)", "Regularisasi — matikan 20% unit"),
        (PURPLE_MID, "Bidirectional LSTM(64)",   "(batch, 24, 128)", "64 unit × 2 arah, return_sequences=True"),
        (GRAY_MID,   "Dropout(0.2)",             "(batch, 24, 128)", ""),
        (RGBColor(0x7E,0x22,0xCE), "Bidirectional LSTM(64)", "(batch, 256)", "return_sequences=False → collapse ke vector"),
        (GRAY_MID,   "Dropout(0.2)",             "(batch, 256)",     ""),
        (GREEN,      "Dense(84)",                "(batch, 84)",      "6 × 14 = 84 output unit"),
        (ORANGE,     "Reshape(6, 14)",           "(batch, 6, 14)",   "6 langkah × 14 fitur sensor"),
    ]

    for i, (color, name, shape, note) in enumerate(layers):
        y = Inches(1.2 + i * 0.67)
        rect(sl, Inches(0.4), y, Inches(3.5), Inches(0.52), color)
        txt(sl, name, Inches(0.5), y + Inches(0.06),
            Inches(3.3), Inches(0.4), size=12, bold=True, color=WHITE)
        rect(sl, Inches(3.9), y, Inches(2.2), Inches(0.52), PURPLE_LIGHT)
        txt(sl, shape, Inches(3.95), y + Inches(0.1),
            Inches(2.1), Inches(0.32), size=11, bold=True, color=PURPLE_DARK,
            align=PP_ALIGN.CENTER, font="Consolas")
        txt(sl, note, Inches(6.2), y + Inches(0.1),
            Inches(6.7), Inches(0.4), size=10, color=GRAY_MID)

    # Right panel: training config & code
    rect(sl, Inches(6.2), Inches(1.15), Inches(6.73), Inches(0.38), GRAY_DARK)
    txt(sl, "Konfigurasi Training",
        Inches(6.35), Inches(1.18), Inches(6.4), Inches(0.32),
        size=12, bold=True, color=WHITE)
    code_block(sl, [
        "model.compile(",
        "    optimizer = 'adam',",
        "    loss      = 'mse',",
        "    metrics   = ['mae'],",
        ")",
        "",
        "EarlyStopping(",
        "    monitor            = 'val_loss',",
        "    patience           = 20,",
        "    restore_best_weights = True,",
        ")",
        "",
        "model.fit(",
        "    X_prepared, y,",
        "    epochs           = 100,",
        "    batch_size       = 32,",
        "    validation_split = 0.1,",
        "    callbacks        = [early_stop],",
        ")",
        "",
        "# Simpan ke: saved_models/{uid}/bilstm.keras",
    ], Inches(6.2), Inches(1.53), Inches(6.73), Inches(4.5), size=10)

    # Bidirectional explanation
    rect(sl, Inches(6.2), Inches(6.13), Inches(6.73), Inches(0.38), PURPLE_MID)
    txt(sl, "Mengapa Bidirectional?",
        Inches(6.35), Inches(6.16), Inches(6.4), Inches(0.32),
        size=12, bold=True, color=WHITE)
    txt(sl, "Forward LSTM: t=0→23 (pola naik/turun normal)\n"
            "Backward LSTM: t=23→0 (konteks dari akhir jendela ke awal)\n"
            "Output digabung (concatenate) → representasi lebih kaya per timestep",
        Inches(6.2), Inches(6.55), Inches(6.73), Inches(0.85),
        size=11, color=GRAY_DARK)

    # Model cache note
    rect(sl, Inches(0.4), Inches(7.22), Inches(5.7), Inches(0.22), GRAY_LIGHT)
    txt(sl, "_model_cache: dict[str, Model] = {}  — load sekali per UID, reuse di memory",
        Inches(0.5), Inches(7.24), Inches(5.5), Inches(0.18),
        size=9, color=GRAY_MID, italic=True, font="Consolas")


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 6: LightGBM Ensemble
# ─────────────────────────────────────────────────────────────────────────────
def slide_06_lgbm(prs):
    sl = blank_slide(prs)
    fill_bg(sl, WHITE)
    slide_header(sl, "LightGBM Ensemble — Refinement + Quantile Regression  (lgbm.py)", accent=GREEN)

    # Why LightGBM after BiLSTM
    rect(sl, Inches(0.3), Inches(1.15), Inches(12.73), Inches(0.55), RGBColor(0xEC, 0xFD, 0xF5))
    txt(sl, "BiLSTM menangkap pola sekuensial (urutan waktu). LightGBM meng-refine output BiLSTM "
            "dengan menambahkan konteks kalender (jam, hari, bulan) yang tidak dipelajari LSTM.",
        Inches(0.45), Inches(1.18), Inches(12.43), Inches(0.48),
        size=12, color=GREEN)

    # Feature construction
    rect(sl, Inches(0.3), Inches(1.8), Inches(6.0), Inches(0.38), GREEN)
    txt(sl, "build_lgbm_features()  — Feature Engineering",
        Inches(0.45), Inches(1.83), Inches(5.7), Inches(0.32),
        size=12, bold=True, color=WHITE)
    code_block(sl, [
        "# Untuk setiap step s ∈ {1..6}:",
        "t = base_time + Timedelta(hours=s)",
        "",
        "features[s] = concatenate([",
        "    bilstm_output[s],    # 14 nilai dari BiLSTM (normed)",
        "    t.hour,              # jam target (0-23)",
        "    t.dayofweek,         # hari (0=Senin, 6=Minggu)",
        "    t.month,             # bulan (1-12)",
        "])  # → (17,) per step",
        "",
        "X = stack([features[s] for s in 1..6])",
        "# X.shape = (6, 17)  → 6 baris, 17 fitur",
    ], Inches(0.3), Inches(2.18), Inches(6.0), Inches(2.65), size=10)

    # 3 models
    rect(sl, Inches(0.3), Inches(4.93), Inches(6.0), Inches(0.38), GREEN)
    txt(sl, "3 Model LightGBM  — Point + Lower + Upper",
        Inches(0.45), Inches(4.96), Inches(5.7), Inches(0.32),
        size=12, bold=True, color=WHITE)
    code_block(sl, [
        "for suffix, extra in [",
        "    ('',       {}),                          # MSE loss → prediksi titik",
        "    ('_lower', {'objective':'quantile',",
        "                'alpha': 0.1}),              # P10 → batas bawah 80% CI",
        "    ('_upper', {'objective':'quantile',",
        "                'alpha': 0.9}),              # P90 → batas atas 80% CI",
        "]:",
        "    model = MultiOutputRegressor(LGBMRegressor(**params))",
        "    model.fit(X_all, y_all)   # y_all.shape = (N×6, 14)",
        "    joblib.dump(model, f'saved_models/{uid}/lgbm{suffix}.pkl')",
    ], Inches(0.3), Inches(5.31), Inches(6.0), Inches(2.05), size=10)

    # Right: hyperparams & MultiOutput
    rect(sl, Inches(6.5), Inches(1.8), Inches(6.53), Inches(0.38), GREEN)
    txt(sl, "Hyperparameter Default",
        Inches(6.65), Inches(1.83), Inches(6.2), Inches(0.32),
        size=12, bold=True, color=WHITE)
    code_block(sl, [
        "n_estimators    = 400",
        "learning_rate   = 0.03",
        "num_leaves      = 63",
        "min_child_samples = 10",
        "subsample       = 0.8   # row subsampling",
        "colsample_bytree = 0.8  # feature subsampling",
        "verbose         = -1",
        "",
        "# Override dengan hasil Optuna tuning:",
        "if lgbm_params:",
        "    base.update(lgbm_params)",
    ], Inches(6.5), Inches(2.18), Inches(6.53), Inches(2.3), size=10)

    # MultiOutputRegressor explanation
    rect(sl, Inches(6.5), Inches(4.58), Inches(6.53), Inches(0.38), ORANGE)
    txt(sl, "MultiOutputRegressor — Satu Model per Target",
        Inches(6.65), Inches(4.61), Inches(6.2), Inches(0.32),
        size=12, bold=True, color=WHITE)
    code_block(sl, [
        "# sklearn wrapper — 14 LGBMRegressor terpisah",
        "# Masing-masing memprediksi 1 fitur × 6 step",
        "# Output: predict(X) → (6, 14)",
        "",
        "# predict() saat inference:",
        "X = build_lgbm_features(bilstm_out, base_time)  # (6, 17)",
        "point = lgbm.predict(X)    # (6, 14)",
        "lower = lgbm_lower.predict(X)  # (6, 14)",
        "upper = lgbm_upper.predict(X)  # (6, 14)",
    ], Inches(6.5), Inches(4.96), Inches(6.53), Inches(2.1), size=10)

    rect(sl, Inches(6.5), Inches(7.11), Inches(6.53), Inches(0.3),
         RGBColor(0xFF, 0xF7, 0xED))
    txt(sl, "Confidence interval 80%  =  [ P10, P90 ]  dari quantile regression",
        Inches(6.6), Inches(7.14), Inches(6.3), Inches(0.25),
        size=10, bold=True, color=ORANGE)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 7: Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def slide_07_training(prs):
    sl = blank_slide(prs)
    fill_bg(sl, WHITE)
    slide_header(sl, "Training Pipeline  (pipeline.py — run_training())", accent=ORANGE)

    code_block(sl, [
        "def run_training(uid, bilstm_params=None, lgbm_params=None):",
        "    # ── 1. Preprocessing ──────────────────────────────────────────",
        "    X, y = preprocess_for_training(uid)   # X:(N,24,18)  y:(N,6,14)",
        "    # TRAIN_HISTORY_HOURS = 8760 jam (1 tahun)",
        "",
        "    # ── 2. Train/Val split 90/10 ──────────────────────────────────",
        "    split      = int(len(X) * 0.9)",
        "    X_train, X_val = X[:split], X[split:]",
        "    y_train, y_val = y[:split], y[split:]",
        "",
        "    # ── 3. Train BiLSTM ───────────────────────────────────────────",
        "    train_bilstm(X_train, y_train, uid, **bilstm_params)",
        "    # Simpan: saved_models/{uid}/bilstm.keras",
        "",
        "    # ── 4. Generate BiLSTM predictions → jadi input LightGBM ─────",
        "    bilstm_preds_train = [predict_bilstm(X_train[i:i+1], uid)[0]",
        "                          for i in range(len(X_train))]",
        "    # shape: (N_train, 6, 14)",
        "",
        "    # ── 5. Train LightGBM (3 models: point, lower, upper) ─────────",
        "    train_lgbm(bilstm_preds_train, y_train, base_times, uid, lgbm_params)",
        "    # Simpan: saved_models/{uid}/lgbm.pkl  lgbm_lower.pkl  lgbm_upper.pkl",
        "",
        "    # ── 6. Evaluasi di validation set → MAE ─────────────────────",
        "    val_bilstm = [predict_bilstm(X_val[i:i+1], uid)[0] ...]",
        "    val_lgbm   = [predict_lgbm(val_bilstm[i], t, uid)['point'] ...]",
        "    mae = float(mean(abs(val_lgbm - y_val)))",
        "    # MAE disimpan ke model_metadata sebagai BASELINE",
        "",
        "    return {'training_samples': len(X), 'mae_score': round(mae, 6)}",
    ], Inches(0.3), Inches(1.15), Inches(12.73), Inches(6.2), size=10.5)

    rect(sl, Inches(0.3), Inches(7.32), Inches(12.73), Inches(0.1), ORANGE)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 8: Hyperparameter Tuning
# ─────────────────────────────────────────────────────────────────────────────
def slide_08_tuning(prs):
    sl = blank_slide(prs)
    fill_bg(sl, WHITE)
    slide_header(sl, "Hyperparameter Tuning — Optuna  (tuning.py)", accent=CYAN)

    rect(sl, Inches(0.3), Inches(1.15), Inches(5.9), Inches(0.38), CYAN)
    txt(sl, "tune_bilstm()  — Search Space BiLSTM",
        Inches(0.45), Inches(1.18), Inches(5.6), Inches(0.32),
        size=12, bold=True, color=WHITE)
    code_block(sl, [
        "def objective(trial):",
        "    params = {",
        "        'units':      trial.suggest_int('units', 64, 256, step=32),",
        "        'dropout':    trial.suggest_float('dropout', 0.1, 0.4),",
        "        'batch_size': trial.suggest_categorical('batch_size', [16,32,64]),",
        "        'epochs':     trial.suggest_int('epochs', 50, 150, step=25),",
        "        'patience':   trial.suggest_int('patience', 10, 30, step=5),",
        "    }",
        "    result = run_training(uid, bilstm_params=params)",
        "    return result['mae_score']",
        "",
        "study = optuna.create_study(direction='minimize')",
        "study.optimize(objective, n_trials=N_TRIALS)",
        "best = study.best_params",
    ], Inches(0.3), Inches(1.53), Inches(5.9), Inches(3.1), size=10)

    rect(sl, Inches(0.3), Inches(4.73), Inches(5.9), Inches(0.38), CYAN)
    txt(sl, "tune_lgbm()  — Search Space LightGBM",
        Inches(0.45), Inches(4.76), Inches(5.6), Inches(0.32),
        size=12, bold=True, color=WHITE)
    code_block(sl, [
        "def objective(trial):",
        "    params = {",
        "        'n_estimators':     trial.suggest_int(100, 600, step=50),",
        "        'learning_rate':    trial.suggest_float(0.01, 0.1, log=True),",
        "        'num_leaves':       trial.suggest_int(15, 127),",
        "        'min_child_samples':trial.suggest_int(5, 50),",
        "        'subsample':        trial.suggest_float(0.6, 1.0),",
        "        'colsample_bytree': trial.suggest_float(0.6, 1.0),",
        "    }",
        "    result = run_training(uid, lgbm_params=params)",
        "    return result['mae_score']",
    ], Inches(0.3), Inches(5.11), Inches(5.9), Inches(2.35), size=10)

    # Right side
    rect(sl, Inches(6.5), Inches(1.15), Inches(6.53), Inches(0.38), GRAY_DARK)
    txt(sl, "Persist Best Params — save/load",
        Inches(6.65), Inches(1.18), Inches(6.2), Inches(0.32),
        size=12, bold=True, color=WHITE)
    code_block(sl, [
        "# Simpan ke JSON setelah tuning selesai",
        "def save_best_params(uid, bilstm, lgbm):",
        "    path = f'saved_models/{uid}/best_params.json'",
        "    json.dump({'bilstm': bilstm, 'lgbm': lgbm}, open(path))",
        "",
        "# Load saat retrain — skip tuning jika sudah ada",
        "def load_best_params(uid):",
        "    path = f'saved_models/{uid}/best_params.json'",
        "    if not exists(path): return None",
        "    return json.load(open(path))",
        "",
        "# Alur retrain_sensor():",
        "params = load_best_params(uid)",
        "if params:",
        "    run_training(uid,",
        "        bilstm_params=params['bilstm'],",
        "        lgbm_params=params['lgbm'])",
        "elif run_tuning:",
        "    tune_bilstm(uid, X, y)",
        "    tune_lgbm(uid, X, y)",
        "else:",
        "    run_training(uid)  # default params",
    ], Inches(6.5), Inches(1.53), Inches(6.53), Inches(4.5), size=10)

    rect(sl, Inches(6.5), Inches(6.13), Inches(6.53), Inches(0.38), CYAN)
    txt(sl, "Evict model cache setelah tuning",
        Inches(6.65), Inches(6.16), Inches(6.2), Inches(0.32),
        size=12, bold=True, color=WHITE)
    code_block(sl, [
        "# bilstm.py — _model_cache",
        "if uid in _model_cache:",
        "    del _model_cache[uid]",
        "# Paksa reload model baru dari disk",
        "# pada prediksi berikutnya",
    ], Inches(6.5), Inches(6.51), Inches(6.53), Inches(0.95), size=10)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 9: Model Files per UID
# ─────────────────────────────────────────────────────────────────────────────
def slide_09_artifacts(prs):
    sl = blank_slide(prs)
    fill_bg(sl, WHITE)
    slide_header(sl, "Model Artifacts — Struktur File per Sensor UID", accent=GRAY_MID)

    # File tree
    rect(sl, Inches(0.3), Inches(1.15), Inches(6.0), Inches(0.38), GRAY_DARK)
    txt(sl, "saved_models/  — Satu folder per UID",
        Inches(0.45), Inches(1.18), Inches(5.7), Inches(0.32),
        size=12, bold=True, color=WHITE)
    code_block(sl, [
        "saved_models/",
        "├── {uid_A}/",
        "│   ├── bilstm.keras        # Keras model (BiLSTM weights)",
        "│   ├── scaler.pkl          # MinMaxScaler (fitted pada data uid_A)",
        "│   ├── lgbm.pkl            # LGBMRegressor point prediction",
        "│   ├── lgbm_lower.pkl      # LGBMRegressor quantile α=0.10",
        "│   ├── lgbm_upper.pkl      # LGBMRegressor quantile α=0.90",
        "│   └── best_params.json    # Optuna best hyperparameters",
        "├── {uid_B}/",
        "│   ├── bilstm.keras",
        "│   ├── scaler.pkl",
        "│   └── ...",
        "└── {uid_C}/",
        "    └── ...",
    ], Inches(0.3), Inches(1.53), Inches(6.0), Inches(3.2), size=11)

    # File details
    files = [
        (PURPLE_MID, "bilstm.keras",      "TF SavedModel format. Dimuat dengan tf.keras.models.load_model(). Di-cache ke _model_cache[uid] setelah load pertama."),
        (BLUE,       "scaler.pkl",        "sklearn MinMaxScaler, di-fit hanya pada FEATURE_COLS saat training. Dipakai untuk normalize input dan denormalize output."),
        (GREEN,      "lgbm.pkl",          "MultiOutputRegressor wrapping 14 LGBMRegressor (satu per target). Objective: MSE regression."),
        (ORANGE,     "lgbm_lower.pkl",    "Sama dengan lgbm.pkl tapi objective='quantile', alpha=0.1 → P10."),
        (RED,        "lgbm_upper.pkl",    "Sama dengan lgbm.pkl tapi objective='quantile', alpha=0.9 → P90."),
        (TEAL,       "best_params.json",  "{\"bilstm\": {...}, \"lgbm\": {...}} — hasil Optuna. Jika ada, digunakan saat retrain. Jika tidak ada, gunakan default."),
    ]
    for i, (color, name, desc) in enumerate(files):
        y = Inches(4.83 + i * 0.45)
        rect(sl, Inches(0.3), y, Inches(2.1), Inches(0.38), color)
        txt(sl, name, Inches(0.38), y + Inches(0.06),
            Inches(1.94), Inches(0.28), size=10, bold=True,
            color=WHITE, font="Consolas")
        rect(sl, Inches(2.4), y, Inches(3.9), Inches(0.38), GRAY_LIGHT)
        txt(sl, desc, Inches(2.5), y + Inches(0.04),
            Inches(3.8), Inches(0.34), size=9, color=GRAY_DARK)

    # Right: predict_lgbm load logic
    rect(sl, Inches(6.5), Inches(1.15), Inches(6.53), Inches(0.38), GREEN)
    txt(sl, "predict_lgbm()  — Lazy Load dari Disk",
        Inches(6.65), Inches(1.18), Inches(6.2), Inches(0.32),
        size=12, bold=True, color=WHITE)
    code_block(sl, [
        "def predict_lgbm(bilstm_output, base_time, uid):",
        "    X = build_lgbm_features(bilstm_output, base_time)",
        "    # X.shape = (6, 17)",
        "",
        "    result = {}",
        "    for key, suffix in [",
        "        ('point', ''),",
        "        ('lower', '_lower'),",
        "        ('upper', '_upper'),",
        "    ]:",
        "        path = f'saved_models/{uid}/lgbm{suffix}.pkl'",
        "        if os.path.exists(path):",
        "            result[key] = joblib.load(path).predict(X)",
        "            # shape: (6, 14)",
        "        else:",
        "            result[key] = None",
        "            # Model belum dilatih → hasilkan None",
        "",
        "    return result  # {'point':(6,14), 'lower':(6,14), 'upper':(6,14)}",
    ], Inches(6.5), Inches(1.53), Inches(6.53), Inches(3.6), size=10)

    rect(sl, Inches(6.5), Inches(5.23), Inches(6.53), Inches(0.38), PURPLE_DARK)
    txt(sl, "Isolasi antar sensor — tidak ada shared state",
        Inches(6.65), Inches(5.26), Inches(6.2), Inches(0.32),
        size=12, bold=True, color=WHITE)
    txt(sl, "Setiap UID memiliki model + scaler sendiri. "
            "Training satu sensor tidak memengaruhi sensor lain. "
            "Memungkinkan sensor baru ditambahkan kapan saja tanpa retrain global.",
        Inches(6.5), Inches(5.68), Inches(6.53), Inches(0.65),
        size=11, color=GRAY_DARK)

    rect(sl, Inches(6.5), Inches(6.43), Inches(6.53), Inches(0.38), ORANGE)
    txt(sl, "Model Cache — BiLSTM saja, bukan LightGBM",
        Inches(6.65), Inches(6.46), Inches(6.2), Inches(0.32),
        size=12, bold=True, color=WHITE)
    txt(sl, "BiLSTM di-cache karena load Keras model mahal (~0.5 detik). "
            "LightGBM di-load tiap prediksi via joblib karena cepat dan tidak makan RAM signifikan.",
        Inches(6.5), Inches(6.89), Inches(6.53), Inches(0.55),
        size=11, color=GRAY_DARK)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    prs = new_prs()

    slide_01_cover(prs)
    slide_02_pipeline(prs)
    slide_03_preprocessing(prs)
    slide_04_ssa(prs)
    slide_05_bilstm(prs)
    slide_06_lgbm(prs)
    slide_07_training(prs)
    slide_08_tuning(prs)
    slide_09_artifacts(prs)

    out = r"C:\Users\nurch\OneDrive\Documents\project\air_quality_forecast\Air_Quality_Model_Technical.pptx"
    prs.save(out)
    print(f"Saved: {out}")
    print(f"Slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
