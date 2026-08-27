"""
Presentasi gaya jurnal/akademik — struktur formal,
konten disesuaikan dengan implementasi nyata.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Palette (formal, bersih) ─────────────────────────────────────────────────
NAVY     = RGBColor(0x0F, 0x23, 0x4E)
BLUE     = RGBColor(0x1D, 0x4E, 0xD8)
BLUE_L   = RGBColor(0xDB, 0xEA, 0xFE)
BLUE_LL  = RGBColor(0xEF, 0xF6, 0xFF)
TEAL     = RGBColor(0x0F, 0x76, 0x6E)
TEAL_L   = RGBColor(0xCC, 0xFB, 0xF1)
GREEN    = RGBColor(0x05, 0x78, 0x52)
GREEN_L  = RGBColor(0xD1, 0xFA, 0xE5)
ORANGE   = RGBColor(0xB4, 0x5A, 0x09)
ORANGE_L = RGBColor(0xFF, 0xED, 0xD5)
RED      = RGBColor(0x99, 0x1B, 0x1B)
RED_L    = RGBColor(0xFE, 0xE2, 0xE2)
PURPLE   = RGBColor(0x5B, 0x21, 0xB6)
PURPLE_L = RGBColor(0xED, 0xE9, 0xFE)
GOLD     = RGBColor(0x92, 0x40, 0x00)
GOLD_L   = RGBColor(0xFF, 0xF7, 0xED)
GRAY_D   = RGBColor(0x1F, 0x29, 0x37)
GRAY     = RGBColor(0x6B, 0x72, 0x80)
GRAY_L   = RGBColor(0xF1, 0xF5, 0xF9)
GRAY_LL  = RGBColor(0xF8, 0xFA, 0xFC)
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
BLACK    = RGBColor(0x00, 0x00, 0x00)

W = Inches(13.33)
H = Inches(7.5)
FONT = "Calibri"


# ── Primitives ───────────────────────────────────────────────────────────────
def new_prs():
    p = Presentation()
    p.slide_width, p.slide_height = W, H
    return p

def sl(prs): return prs.slides.add_slide(prs.slide_layouts[6])

def bg(slide, c):
    f = slide.background.fill; f.solid(); f.fore_color.rgb = c

def box(slide, l, t, w, h, fill, line_c=None, lw=1.0):
    s = slide.shapes.add_shape(1, l, t, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line_c: s.line.color.rgb = line_c; s.line.width = Pt(lw)
    else: s.line.fill.background()
    return s

def T(slide, text, l, t, w, h, size=12, bold=False, color=GRAY_D,
      align=PP_ALIGN.LEFT, italic=False):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold
    r.font.italic = italic; r.font.color.rgb = color
    r.font.name = FONT
    return tb

def Tm(slide, items, l, t, w, h, default_align=PP_ALIGN.LEFT):
    """Multi-run / multi-para textbox. items = list of (text, size, bold, color, space_after)"""
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    for i, item in enumerate(items):
        text  = item[0]
        size  = item[1] if len(item) > 1 else 12
        bold  = item[2] if len(item) > 2 else False
        color = item[3] if len(item) > 3 else GRAY_D
        space = item[4] if len(item) > 4 else 3
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = default_align; p.space_after = Pt(space)
        r = p.add_run(); r.text = text
        r.font.size = Pt(size); r.font.bold = bold
        r.font.color.rgb = color; r.font.name = FONT
    return tb

# ── Layout helpers ───────────────────────────────────────────────────────────
def header_band(slide, title, subtitle=None, accent=BLUE):
    """Navy top band with blue accent strip."""
    box(slide, 0, 0, W, Inches(1.05), NAVY)
    box(slide, 0, Inches(1.05), W, Inches(0.055), accent)
    T(slide, title, Inches(0.5), Inches(0.1), Inches(12.33), Inches(0.62),
      size=26, bold=True, color=WHITE)
    if subtitle:
        T(slide, subtitle, Inches(0.5), Inches(0.68), Inches(12.33), Inches(0.35),
          size=11, color=RGBColor(0xBA, 0xC8, 0xFF), italic=True)

def section_label(slide, label, l, t, w, color=BLUE):
    """Small coloured label pill above a block."""
    box(slide, l, t, w, Inches(0.28), color)
    T(slide, label, l + Inches(0.1), t + Inches(0.02), w - Inches(0.2), Inches(0.24),
      size=9, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

def card(slide, l, t, w, h, title, body_items,
         accent=BLUE, title_size=12, body_size=11):
    box(slide, l, t, w, Inches(0.38), accent)
    T(slide, title, l + Inches(0.12), t + Inches(0.05),
      w - Inches(0.24), Inches(0.28), size=title_size, bold=True, color=WHITE)
    box(slide, l, t + Inches(0.38), w, h - Inches(0.38), GRAY_LL,
        line_c=accent, lw=1.0)
    y_off = t + Inches(0.46)
    for item in body_items:
        T(slide, f"  •  {item}", l + Inches(0.1), y_off,
          w - Inches(0.2), Inches(0.32), size=body_size, color=GRAY_D)
        y_off += Inches(0.33)

def formula_box(slide, l, t, w, h, formula, label=None, accent=BLUE):
    box(slide, l, t, w, h, GRAY_LL, line_c=accent, lw=1.5)
    if label:
        T(slide, label, l + Inches(0.12), t + Inches(0.06),
          w - Inches(0.24), Inches(0.25), size=9, color=GRAY, italic=True)
    T(slide, formula,
      l + Inches(0.12), t + (Inches(0.28) if label else Inches(0.1)),
      w - Inches(0.24), h - (Inches(0.32) if label else Inches(0.18)),
      size=13, bold=True, color=accent, align=PP_ALIGN.CENTER)

def step_row(slide, steps, y, h=Inches(1.1)):
    """Horizontal flow: list of (color, number, title, desc)."""
    n = len(steps)
    step_w = (W - Inches(0.6)) / n
    for i, (c, num, title, desc) in enumerate(steps):
        x = Inches(0.3) + int(i * step_w)
        w_ = step_w - Inches(0.12)
        box(slide, x, y, w_, Inches(0.42), c)
        T(slide, f"{num}  {title}", x + Inches(0.1), y + Inches(0.06),
          w_ - Inches(0.2), Inches(0.32), size=11, bold=True, color=WHITE)
        box(slide, x, y + Inches(0.42), w_, h - Inches(0.42),
            WHITE, line_c=c, lw=1.0)
        T(slide, desc, x + Inches(0.1), y + Inches(0.48),
          w_ - Inches(0.2), h - Inches(0.54), size=10, color=GRAY_D)
        if i < n - 1:
            T(slide, "›", x + w_, y + Inches(0.22),
              Inches(0.15), Inches(0.3), size=14, bold=True,
              color=c, align=PP_ALIGN.CENTER)

def page_num(slide, n, total):
    T(slide, f"{n} / {total}", W - Inches(0.9), H - Inches(0.32),
      Inches(0.8), Inches(0.28), size=9, color=GRAY, align=PP_ALIGN.RIGHT)


# ═══════════════════════════════════════════════════════════════════════════
# S1 — COVER
# ═══════════════════════════════════════════════════════════════════════════
def s01(prs):
    s = sl(prs); bg(s, NAVY)
    box(s, 0, Inches(4.55), W, Inches(0.07), BLUE)
    box(s, 0, H - Inches(0.07), W, Inches(0.07), BLUE)
    box(s, 0, 0, Inches(0.1), H, BLUE)

    T(s, "Prediksi Kualitas Udara Berbasis",
      Inches(0.5), Inches(0.6), Inches(9.5), Inches(0.75),
      size=34, bold=True, color=WHITE)
    T(s, "Hybrid BiLSTM–LightGBM dengan Dekomposisi SSA",
      Inches(0.5), Inches(1.35), Inches(9.5), Inches(0.65),
      size=28, bold=False, color=RGBColor(0x93, 0xC5, 0xFD))
    T(s, "Multi-Step, Multi-Output Forecasting dengan Interval Kepercayaan",
      Inches(0.5), Inches(2.1), Inches(9.5), Inches(0.4),
      size=15, color=RGBColor(0x60, 0x82, 0xBC), italic=True)

    box(s, Inches(0.5), Inches(2.65), Inches(5.5), Inches(0.04), BLUE)

    info = [
        "Sistem pemantauan kualitas udara real-time",
        "Studi kasus: Jaringan sensor AQMS multi-titik",
    ]
    for i, line in enumerate(info):
        T(s, f"  •  {line}", Inches(0.5), Inches(2.8 + i * 0.4),
          Inches(9), Inches(0.36), size=13, color=RGBColor(0xBA, 0xC8, 0xFF))

    # Right panel — key specs
    box(s, Inches(10.1), Inches(0.5), Inches(3.0), Inches(6.5),
        RGBColor(0x1A, 0x2E, 0x6B))
    T(s, "Spesifikasi", Inches(10.25), Inches(0.65),
      Inches(2.7), Inches(0.38), size=13, bold=True, color=BLUE,
      align=PP_ALIGN.CENTER)
    specs = [
        ("Input window",   "24 jam"),
        ("Output horizon", "6 langkah"),
        ("Target fitur",   "14 parameter"),
        ("Confidence",     "80% CI"),
        ("Frekuensi",      "Setiap jam"),
        ("Per-sensor",     "Model unik"),
    ]
    for i, (k, v) in enumerate(specs):
        y = Inches(1.15 + i * 0.85)
        box(s, Inches(10.25), y, Inches(2.7), Inches(0.72),
            RGBColor(0x1E, 0x3A, 0x80))
        T(s, k, Inches(10.35), y + Inches(0.04),
          Inches(2.5), Inches(0.28), size=10, color=RGBColor(0x93, 0xC5, 0xFD))
        T(s, v, Inches(10.35), y + Inches(0.34),
          Inches(2.5), Inches(0.32), size=16, bold=True, color=WHITE)

    T(s, "Python 3.11  ·  TensorFlow/Keras  ·  LightGBM  ·  FastAPI",
      Inches(0.5), Inches(4.8), Inches(9.3), Inches(0.35),
      size=12, color=RGBColor(0x64, 0x74, 0x8B))


# ═══════════════════════════════════════════════════════════════════════════
# S2 — LATAR BELAKANG
# ═══════════════════════════════════════════════════════════════════════════
def s02(prs):
    s = sl(prs); bg(s, GRAY_LL)
    header_band(s, "Latar Belakang",
                "Mengapa diperlukan model prediksi kualitas udara?", BLUE)

    # Problem statement
    box(s, Inches(0.3), Inches(1.2), W - Inches(0.6), Inches(0.8), BLUE_L,
        line_c=BLUE, lw=1.5)
    T(s, "Permasalahan",
      Inches(0.45), Inches(1.24), Inches(3), Inches(0.3),
      size=11, bold=True, color=BLUE)
    T(s, "Kualitas udara bersifat dinamis dan dipengaruhi oleh banyak variabel yang saling berinteraksi "
         "(polutan, meteorologi, pola aktivitas). Pemantauan reaktif tidak cukup — diperlukan sistem "
         "prediktif yang dapat memberikan peringatan dini sebelum kondisi memburuk.",
      Inches(0.45), Inches(1.54), W - Inches(0.9), Inches(0.44),
      size=11, color=GRAY_D)

    # Three challenges
    challs = [
        (BLUE,   "Tantangan 1\nNon-Linearitas",
         "Hubungan antar variabel polutan bersifat non-linear dan "
         "bergantung pada konteks waktu (jam, hari, musim)."),
        (TEAL,   "Tantangan 2\nKetergantungan Temporal",
         "Nilai saat ini dipengaruhi oleh urutan nilai masa lalu — "
         "model tabular konvensional tidak mampu menangkap ini."),
        (ORANGE, "Tantangan 3\nKetidakpastian Prediksi",
         "Prediksi titik saja tidak cukup. Pengambil keputusan "
         "membutuhkan interval kepercayaan untuk manajemen risiko."),
    ]
    for i, (c, title, desc) in enumerate(challs):
        x = Inches(0.3 + i * 4.3)
        box(s, x, Inches(2.15), Inches(4.0), Inches(0.46), c)
        T(s, title, x + Inches(0.12), Inches(2.18),
          Inches(3.76), Inches(0.4), size=12, bold=True, color=WHITE)
        box(s, x, Inches(2.61), Inches(4.0), Inches(1.4), WHITE,
            line_c=c, lw=1.0)
        T(s, desc, x + Inches(0.12), Inches(2.68),
          Inches(3.76), Inches(1.28), size=11, color=GRAY_D)

    # Solution
    box(s, Inches(0.3), Inches(4.15), W - Inches(0.6), Inches(0.42), NAVY)
    T(s, "Solusi yang Diusulkan",
      Inches(0.45), Inches(4.18), Inches(4), Inches(0.35),
      size=12, bold=True, color=WHITE)

    solutions = [
        (BLUE,   "BiLSTM",           "Menangkap ketergantungan temporal jangka pendek dan panjang secara bidireksional"),
        (TEAL,   "SSA",              "Dekomposisi sinyal untuk memisahkan komponen tren dan osilasi sebelum learning"),
        (GREEN,  "LightGBM Ensemble","Memperbaiki prediksi BiLSTM dan menghasilkan interval kepercayaan via quantile regression"),
        (ORANGE, "Per-UID Model",    "Model terpisah per sensor untuk menangkap karakteristik lokasi yang unik"),
    ]
    for i, (c, title, desc) in enumerate(solutions):
        x = Inches(0.3 + i * 3.25)
        y = Inches(4.67)
        box(s, x, y, Inches(3.0), Inches(0.38), c)
        T(s, title, x + Inches(0.1), y + Inches(0.05),
          Inches(2.8), Inches(0.3), size=11, bold=True, color=WHITE)
        box(s, x, y + Inches(0.38), Inches(3.0), Inches(0.92),
            WHITE, line_c=c, lw=1.0)
        T(s, desc, x + Inches(0.1), y + Inches(0.44),
          Inches(2.8), Inches(0.82), size=10, color=GRAY_D)

    page_num(s, 2, 12)


# ═══════════════════════════════════════════════════════════════════════════
# S3 — TINJAUAN PUSTAKA
# ═══════════════════════════════════════════════════════════════════════════
def s03(prs):
    s = sl(prs); bg(s, GRAY_LL)
    header_band(s, "Tinjauan Pustaka",
                "Metode-metode yang menjadi landasan implementasi", TEAL)

    methods = [
        (BLUE, "LSTM & BiLSTM",
         [
             "Long Short-Term Memory (LSTM) — Hochreiter & Schmidhuber, 1997",
             "Dirancang untuk menangkap dependensi jangka panjang dalam deret waktu",
             "BiLSTM memproses sekuens dua arah (maju & mundur) secara simultan",
             "Terbukti unggul untuk prediksi deret waktu multivariat (air quality, traffic, finance)",
             "Implementasi: 3 lapis BiLSTM, units={128, 64, 64}, dropout=0.2",
         ]),
        (TEAL, "Singular Spectrum Analysis (SSA)",
         [
             "Metode dekomposisi non-parametrik berbasis SVD (Broomhead & King, 1986)",
             "Memisahkan sinyal menjadi komponen trend, osilasi, dan noise",
             "Digunakan sebagai feature engineering sebelum input ke neural network",
             "Window length L=12 — menangkap siklus harian setengah-hari",
             "Implementasi: setiap fitur sensor diurai menjadi tren + osilasi",
         ]),
        (GREEN, "LightGBM",
         [
             "Gradient Boosting framework berbasis decision tree (Ke et al., 2017)",
             "Histogram-based — efisien untuk data tabular berdimensi tinggi",
             "Mendukung quantile regression untuk estimasi interval kepercayaan",
             "Digunakan sebagai residual corrector atas output BiLSTM",
             "Implementasi: 3 model — point (MSE), lower (α=0.1), upper (α=0.9)",
         ]),
        (ORANGE, "Quantile Regression",
         [
             "Memodelkan distribusi kondisional output, bukan hanya ekspektasi",
             "Pinball loss: Lα(y,ŷ) = α·(y−ŷ) jika y≥ŷ, else (α−1)·(y−ŷ)",
             "α=0.1 → P10, α=0.9 → P90 menghasilkan interval 80% CI",
             "Tidak bergantung asumsi distribusi (distribution-free)",
             "Implementasi: LGBMRegressor(objective='quantile', alpha=0.1/0.9)",
         ]),
    ]

    for i, (c, title, items) in enumerate(methods):
        col = i % 2; row = i // 2
        x = Inches(0.3 + col * 6.5)
        y = Inches(1.18 + row * 3.05)
        box(s, x, y, Inches(6.18), Inches(0.42), c)
        T(s, title, x + Inches(0.12), y + Inches(0.06),
          Inches(5.94), Inches(0.3), size=13, bold=True, color=WHITE)
        box(s, x, y + Inches(0.42), Inches(6.18), Inches(2.55),
            WHITE, line_c=c, lw=1.0)
        for j, item in enumerate(items):
            marker = "›  " if j > 0 else "►  "
            clr = c if j == 0 else GRAY_D
            sz = 11 if j > 0 else 11
            bd = (j == 0)
            T(s, marker + item,
              x + Inches(0.12), y + Inches(0.48 + j * 0.41),
              Inches(5.94), Inches(0.38), size=sz, bold=bd, color=clr)

    page_num(s, 3, 12)


# ═══════════════════════════════════════════════════════════════════════════
# S4 — METODOLOGI: GAMBARAN SISTEM
# ═══════════════════════════════════════════════════════════════════════════
def s04(prs):
    s = sl(prs); bg(s, GRAY_LL)
    header_band(s, "Metodologi  —  Arsitektur Sistem Keseluruhan",
                "Alur data dari sensor hingga hasil prediksi", NAVY)

    # Layers
    layers = [
        (BLUE,   "Layer 1: Akuisisi Data",
         "Sensor IoT (PM2.5, PM10, TSP, Noise, Temp, Humidity, Pressure, AQI)",
         "Database: t_loggers (SENSOR_DB) — data masuk per menit, resolusi bervariasi"),
        (TEAL,   "Layer 2: Pra-Pemrosesan",
         "Resample hourly → Deteksi anomali (rolling z-score 3σ) → IQR clipping → Normalisasi MinMax",
         "Output: sekuens ternormalisasi shape (N, 24, 18) — 14 sensor + 4 fitur waktu siklus"),
        (PURPLE, "Layer 3: Feature Engineering",
         "SSA Decomposition — setiap fitur sensor dipecah menjadi tren dan osilasi via SVD",
         "Output: shape (N, 24, 32) — 14 tren + 14 osilasi + 4 fitur waktu"),
        (BLUE,   "Layer 4: BiLSTM Backbone",
         "3 lapis Bidirectional LSTM — mempelajari pola temporal dari kedua arah (maju & mundur)",
         "Output: shape (N, 6, 14) — prediksi awal 6 langkah × 14 fitur (masih ternormalisasi)"),
        (GREEN,  "Layer 5: LightGBM Corrector",
         "3 model LightGBM paralel: point (MSE) + lower (P10) + upper (P90) quantile regression",
         "Input: output BiLSTM + fitur kalender (jam, hari, bulan) → Output: (6, 14) × 3"),
        (ORANGE, "Layer 6: Post-Processing & Storage",
         "Denormalisasi ke satuan asli (µg/m³, °C, dBA, dll.) via inverse MinMax transform",
         "Simpan ke predictions (RESULT_DB) — step, target_time, nilai + lower + upper bounds"),
    ]

    for i, (c, title, desc1, desc2) in enumerate(layers):
        y = Inches(1.15 + i * 1.04)
        box(s, Inches(0.3), y, Inches(2.7), Inches(0.9), c)
        T(s, title, Inches(0.4), y + Inches(0.18),
          Inches(2.5), Inches(0.55), size=10, bold=True, color=WHITE)
        box(s, Inches(3.0), y, Inches(9.8), Inches(0.42), GRAY_L)
        T(s, desc1, Inches(3.12), y + Inches(0.05),
          Inches(9.56), Inches(0.34), size=10, bold=True, color=GRAY_D)
        box(s, Inches(3.0), y + Inches(0.42), Inches(9.8), Inches(0.48), WHITE,
            line_c=c, lw=0.8)
        T(s, desc2, Inches(3.12), y + Inches(0.47),
          Inches(9.56), Inches(0.4), size=10, color=GRAY, italic=True)
        if i < len(layers) - 1:
            T(s, "↓", Inches(1.5), y + Inches(0.9),
              Inches(0.5), Inches(0.18), size=10, bold=True,
              color=c, align=PP_ALIGN.CENTER)

    page_num(s, 4, 12)


# ═══════════════════════════════════════════════════════════════════════════
# S5 — PRA-PEMROSESAN
# ═══════════════════════════════════════════════════════════════════════════
def s05(prs):
    s = sl(prs); bg(s, GRAY_LL)
    header_band(s, "Pra-Pemrosesan Data",
                "Pipeline pembersihan dan transformasi sebelum model", TEAL)

    steps = [
        (BLUE,   "①  Resampling",
         "df[FEATURE_COLS].resample('h').mean()",
         "Data sub-jam diratakan ke 1 data per jam menggunakan aritmetika rata-rata. "
         "Menghasilkan deret waktu reguler yang diperlukan oleh LSTM."),
        (RED,    "②  Deteksi Anomali",
         "|xₜ − median_roll| > 3σ_roll  →  NaN  →  interpolasi linear",
         "Spike terdeteksi via rolling window 24 jam. Nilai yang menyimpang >3σ "
         "dari median digantikan dengan interpolasi linear. Mencegah outlier sensor "
         "mendistorsi training."),
        (TEAL,   "③  Pengisian & Pemotongan",
         "interpolate(linear) → ffill/bfill → clip[Q1−1.5IQR, Q3+1.5IQR]",
         "Gap sisa diisi interpolasi, diikuti forward/backward fill. "
         "IQR fencing mencegah distribusi didominasi kejadian ekstrem langka."),
        (GREEN,  "④  Fitur Waktu Siklus",
         "sin(2π·h/24), cos(2π·h/24), sin(2π·d/7), cos(2π·d/7)",
         "Encoding siklus menghindari diskontinuitas. "
         "Representasi sin/cos memastikan jam 23 ≈ jam 0 secara geometris "
         "— properti yang tidak dimiliki encoding ordinal."),
        (ORANGE, "⑤  Normalisasi",
         "MinMaxScaler — fit pada FEATURE_COLS saat training, load saat inference",
         "Skala nilai ke [0, 1]. Scaler disimpan per-UID (saved_models/{uid}/scaler.pkl) "
         "karena sensor yang berbeda memiliki rentang nilai yang sangat berbeda."),
        (PURPLE, "⑥  Pembentukan Sekuens",
         "X: (N, 24, 18)  ←  sliding window N_INPUT=24, stride=1",
         "Target y: (N, 6, 14) — 6 langkah ke depan, 14 kolom sensor saja "
         "(fitur waktu bukan target prediksi). Data training mencakup ±8.760 jam (1 tahun)."),
    ]

    for i, (c, title, formula, desc) in enumerate(steps):
        col = i % 2; row = i // 2
        x = Inches(0.3 + col * 6.5)
        y = Inches(1.15 + row * 2.05)
        box(s, x, y, Inches(6.18), Inches(0.38), c)
        T(s, title, x + Inches(0.12), y + Inches(0.05),
          Inches(5.94), Inches(0.28), size=12, bold=True, color=WHITE)
        box(s, x, y + Inches(0.38), Inches(6.18), Inches(0.42), GRAY_L)
        T(s, formula, x + Inches(0.12), y + Inches(0.4),
          Inches(5.94), Inches(0.36), size=10, color=c,
          bold=True, italic=True)
        box(s, x, y + Inches(0.8), Inches(6.18), Inches(1.2),
            WHITE, line_c=c, lw=1.0)
        T(s, desc, x + Inches(0.12), y + Inches(0.87),
          Inches(5.94), Inches(1.08), size=10, color=GRAY_D)

    page_num(s, 5, 12)


# ═══════════════════════════════════════════════════════════════════════════
# S6 — SSA
# ═══════════════════════════════════════════════════════════════════════════
def s06(prs):
    s = sl(prs); bg(s, GRAY_LL)
    header_band(s, "Dekomposisi SSA (Singular Spectrum Analysis)",
                "Feature engineering berbasis SVD untuk memisahkan tren dan osilasi", TEAL)

    # Left: algorithm
    box(s, Inches(0.3), Inches(1.15), Inches(6.0), Inches(0.38), TEAL)
    T(s, "Prosedur Matematika  (window L = 12, series length n = 24)",
      Inches(0.42), Inches(1.18), Inches(5.76), Inches(0.3),
      size=12, bold=True, color=WHITE)

    algo_steps = [
        (BLUE, "Langkah 1: Embedding",
         "Konstruksi trajectory matrix 𝒳 ∈ ℝᴷˣᴸ\n"
         "K = n − L + 1 = 24 − 12 + 1 = 13\n"
         "𝒳ᵢⱼ = xᵢ₊ⱼ₋₁,    i=1..K,  j=1..L"),
        (TEAL, "Langkah 2: Dekomposisi SVD",
         "𝒳 = UΣVᵀ\n"
         "Komponen dominan ke-1:  X̃₁ = σ₁·u₁vᵀ₁\n"
         "σ₁ = nilai singular terbesar → menangkap variansi maksimum"),
        (GREEN, "Langkah 3: Diagonal Averaging",
         "Hankelisasi: ubah matriks rank-1 X̃₁ kembali ke series\n"
         "x̂ₖ = rata-rata elemen anti-diagonal ke-k dari X̃₁\n"
         "→ trend x̂ ∈ ℝⁿ"),
        (ORANGE, "Langkah 4: Residual",
         "oscillation = x − x̂\n"
         "Merepresentasikan siklus harian, variasi musiman, noise terstruktur"),
    ]

    for i, (c, title, body) in enumerate(algo_steps):
        y = Inches(1.63 + i * 1.38)
        box(s, Inches(0.3), y, Inches(0.06), Inches(1.18), c)
        box(s, Inches(0.36), y, Inches(5.94), Inches(0.38), GRAY_L)
        T(s, title, Inches(0.48), y + Inches(0.05),
          Inches(5.7), Inches(0.28), size=11, bold=True, color=c)
        box(s, Inches(0.36), y + Inches(0.38), Inches(5.94), Inches(0.8),
            WHITE, line_c=c, lw=0.8)
        T(s, body, Inches(0.48), y + Inches(0.43),
          Inches(5.7), Inches(0.72), size=10, color=GRAY_D)

    # Right: transformation
    box(s, Inches(6.6), Inches(1.15), Inches(6.43), Inches(0.38), NAVY)
    T(s, "Transformasi Dimensi Fitur",
      Inches(6.72), Inches(1.18), Inches(6.19), Inches(0.3),
      size=12, bold=True, color=WHITE)

    transforms = [
        (BLUE,   "Input sensor ternnormalisasi",  "(N, 24, 14)"),
        (TEAL,   "SSA per fitur → tren + osilasi","(N, 24, 28)"),
        (GREEN,  "Concat dengan fitur waktu",     "(N, 24, 32)"),
        (ORANGE, "Masuk ke BiLSTM",               "(N, 24, 32) ✓"),
    ]
    for i, (c, label, shape) in enumerate(transforms):
        y = Inches(1.65 + i * 1.0)
        box(s, Inches(6.6), y, Inches(3.9), Inches(0.78), WHITE, line_c=c, lw=1.5)
        T(s, label, Inches(6.72), y + Inches(0.08),
          Inches(3.66), Inches(0.3), size=11, color=GRAY_D)
        box(s, Inches(9.9), y, Inches(1.13), Inches(0.78), c)
        T(s, shape, Inches(9.92), y + Inches(0.15),
          Inches(1.09), Inches(0.48), size=10, bold=True, color=WHITE,
          align=PP_ALIGN.CENTER)
        if i < len(transforms) - 1:
            T(s, "↓", Inches(7.3), y + Inches(0.78),
              Inches(0.5), Inches(0.25), size=12, color=c,
              align=PP_ALIGN.CENTER)

    # Key insight
    box(s, Inches(6.6), Inches(5.75), Inches(6.43), Inches(1.65),
        TEAL_L, line_c=TEAL, lw=1.5)
    T(s, "Rasionalisasi", Inches(6.72), Inches(5.82),
      Inches(6.19), Inches(0.3), size=11, bold=True, color=TEAL)
    T(s, "Prediksi tren dan osilasi merupakan dua masalah yang berbeda secara karakteristik. "
         "Tren bersifat smooth dan persisten, sedangkan osilasi bersifat siklus dan lokal. "
         "Dengan memisahkan keduanya sebelum masuk ke LSTM, model dapat belajar "
         "representasi yang lebih terstruktur dan generalisasi yang lebih baik.",
      Inches(6.72), Inches(6.16), Inches(6.19), Inches(1.18),
      size=10, color=GRAY_D)

    page_num(s, 6, 12)


# ═══════════════════════════════════════════════════════════════════════════
# S7 — BiLSTM
# ═══════════════════════════════════════════════════════════════════════════
def s07(prs):
    s = sl(prs); bg(s, GRAY_LL)
    header_band(s, "Arsitektur BiLSTM",
                "Backbone deep learning untuk pemodelan dependensi temporal", PURPLE)

    # Architecture table
    box(s, Inches(0.3), Inches(1.15), Inches(6.2), Inches(0.42), NAVY)
    T(s, "Struktur Lapisan", Inches(0.42), Inches(1.18),
      Inches(5.96), Inches(0.34), size=12, bold=True, color=WHITE)

    headers = ["Layer", "Tipe", "Parameter", "Output Shape", "Keterangan"]
    col_ws  = [Inches(1.6), Inches(1.5), Inches(1.2), Inches(1.5), Inches(3.0)]
    hx = [Inches(0.3), Inches(1.9), Inches(3.4), Inches(4.6), Inches(6.1)]

    box(s, Inches(0.3), Inches(1.57), Inches(6.2), Inches(0.35), GRAY_D)
    for j, (h, w, x) in enumerate(zip(headers, col_ws, hx)):
        T(s, h, x + Inches(0.06), Inches(1.6),
          w - Inches(0.12), Inches(0.3),
          size=10, bold=True, color=WHITE)

    rows = [
        ("Input",      "—",              "—",             "(B, 24, 32)", "24 timestep, 32 fitur SSA+time"),
        ("BiLSTM-1",   "Bidirectional",  "128 unit ×2",   "(B, 24, 256)","return_sequences=True"),
        ("Dropout-1",  "Regularisasi",   "p = 0.20",      "(B, 24, 256)","Mencegah overfitting"),
        ("BiLSTM-2",   "Bidirectional",  "64 unit ×2",    "(B, 24, 128)","return_sequences=True"),
        ("Dropout-2",  "Regularisasi",   "p = 0.20",      "(B, 24, 128)",""),
        ("BiLSTM-3",   "Bidirectional",  "64 unit ×2",    "(B, 256)",    "return_sequences=False"),
        ("Dropout-3",  "Regularisasi",   "p = 0.20",      "(B, 256)",    ""),
        ("Dense",      "Fully Connected","84 unit",        "(B, 84)",     "6 × 14 = 84 output"),
        ("Reshape",    "—",              "—",             "(B, 6, 14)",  "6 langkah, 14 fitur"),
    ]
    layer_colors = [BLUE, PURPLE, GRAY, PURPLE, GRAY,
                    RGBColor(0x6D,0x28,0xD9), GRAY, GREEN, ORANGE]
    for i, (row, rc) in enumerate(zip(rows, layer_colors)):
        y = Inches(1.92 + i * 0.44)
        bg_c = WHITE if i % 2 == 0 else GRAY_LL
        box(s, Inches(0.3), y, Inches(6.2), Inches(0.42), bg_c)
        box(s, Inches(0.3), y, Inches(0.06), Inches(0.42), rc)
        for j, (val, w, x) in enumerate(zip(row, col_ws, hx)):
            bd = (j == 0)
            T(s, val, x + Inches(0.06), y + Inches(0.07),
              w - Inches(0.12), Inches(0.3), size=9.5,
              bold=bd, color=rc if j == 0 else GRAY_D)

    # Right: training & objective
    box(s, Inches(6.7), Inches(1.15), Inches(6.33), Inches(0.42), NAVY)
    T(s, "Fungsi Objektif & Konfigurasi Training",
      Inches(6.82), Inches(1.18), Inches(6.09), Inches(0.34),
      size=12, bold=True, color=WHITE)

    formula_box(s, Inches(6.7), Inches(1.67), Inches(6.33), Inches(0.72),
                "L_MSE(y, ŷ) = (1/N) Σᵢ (yᵢ − ŷᵢ)²",
                "Loss Function: Mean Squared Error", PURPLE)

    training_params = [
        ("Optimizer",          "Adam  (lr = adaptif)"),
        ("Batch size",         "32"),
        ("Max epochs",         "100"),
        ("Early stopping",     "patience = 20, monitor = val_loss"),
        ("Validation split",   "10% dari data training"),
        ("restore_best_weights","True — simpan bobot terbaik"),
        ("Aktivasi",           "tanh (LSTM default)"),
    ]
    box(s, Inches(6.7), Inches(2.5), Inches(6.33), Inches(0.38), GRAY_D)
    T(s, "Parameter Training", Inches(6.82), Inches(2.53),
      Inches(6.09), Inches(0.3), size=11, bold=True, color=WHITE)

    for i, (k, v) in enumerate(training_params):
        y = Inches(2.98 + i * 0.48)
        box(s, Inches(6.7), y, Inches(2.3), Inches(0.42), GRAY_L)
        T(s, k, Inches(6.8), y + Inches(0.06),
          Inches(2.1), Inches(0.3), size=10, bold=True, color=GRAY_D)
        T(s, v, Inches(9.05), y + Inches(0.06),
          Inches(3.9), Inches(0.3), size=10, color=GRAY_D)

    box(s, Inches(6.7), Inches(6.42), Inches(6.33), Inches(1.02),
        PURPLE_L, line_c=PURPLE, lw=1.5)
    T(s, "Keunggulan BiLSTM", Inches(6.82), Inches(6.48),
      Inches(6.09), Inches(0.3), size=11, bold=True, color=PURPLE)
    T(s, "LSTM forward menangkap pola kausalitas (t-n → t). "
         "LSTM backward menangkap konteks reversibel (periodisitas harian). "
         "Concatenation menghasilkan representasi yang lebih informatif per timestep.",
      Inches(6.82), Inches(6.82), Inches(6.09), Inches(0.58),
      size=10, color=GRAY_D)

    page_num(s, 7, 12)


# ═══════════════════════════════════════════════════════════════════════════
# S8 — LightGBM ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════
def s08(prs):
    s = sl(prs); bg(s, GRAY_LL)
    header_band(s, "LightGBM Ensemble  —  Koreksi Residual & Quantile Regression",
                "Memperbaiki output BiLSTM dan menghasilkan interval kepercayaan", GREEN)

    # Rationale
    box(s, Inches(0.3), Inches(1.15), W - Inches(0.6), Inches(0.62), GREEN_L,
        line_c=GREEN, lw=1.5)
    T(s, "Rasionalisasi Ensemble:", Inches(0.42), Inches(1.18),
      Inches(3), Inches(0.28), size=11, bold=True, color=GREEN)
    T(s, "BiLSTM menghasilkan prediksi dalam ruang fitur ternormalisasi berdasarkan urutan temporal. "
         "LightGBM memperbaikinya dengan menambahkan konteks kalender eksplisit dan menghasilkan "
         "distribusi kondisional output (bukan hanya ekspektasi), memungkinkan estimasi ketidakpastian.",
      Inches(0.42), Inches(1.46), W - Inches(0.84), Inches(0.28),
      size=11, color=GRAY_D)

    # Feature construction
    box(s, Inches(0.3), Inches(1.88), Inches(5.9), Inches(0.38), GREEN)
    T(s, "Konstruksi Fitur Input LightGBM",
      Inches(0.42), Inches(1.91), Inches(5.66), Inches(0.3),
      size=12, bold=True, color=WHITE)

    feats = [
        (BLUE,   "BiLSTM output\n[14 nilai]",   "Prediksi awal untuk langkah ke-s\n(ternormalisasi [0,1])"),
        (ORANGE, "Jam target\n[1 nilai]",        "Hour(base_time + s·1h)\nKonteks waktu dalam hari"),
        (TEAL,   "Hari dalam minggu\n[1 nilai]", "DayOfWeek(target)\nPola weekday vs. weekend"),
        (PURPLE, "Bulan\n[1 nilai]",             "Month(target)\nPola musiman / cuaca"),
    ]
    for i, (c, title, desc) in enumerate(feats):
        x = Inches(0.3 + i * 1.43)
        box(s, x, Inches(2.36), Inches(1.3), Inches(0.56), c)
        T(s, title, x + Inches(0.08), Inches(2.38),
          Inches(1.14), Inches(0.52), size=9, bold=True, color=WHITE,
          align=PP_ALIGN.CENTER)
        box(s, x, Inches(2.92), Inches(1.3), Inches(0.9), WHITE, line_c=c, lw=0.8)
        T(s, desc, x + Inches(0.07), Inches(2.95),
          Inches(1.16), Inches(0.84), size=8.5, color=GRAY_D, italic=True)

    T(s, "→ Input per baris: 17 fitur  ×  6 baris (langkah)  =  matriks (6, 17)",
      Inches(0.3), Inches(3.95), Inches(5.9), Inches(0.3),
      size=11, bold=True, color=GREEN)

    # 3 models
    box(s, Inches(0.3), Inches(4.38), Inches(5.9), Inches(0.38), NAVY)
    T(s, "Tiga Model LightGBM Terlatih Paralel",
      Inches(0.42), Inches(4.41), Inches(5.66), Inches(0.3),
      size=12, bold=True, color=WHITE)

    formula_box(s, Inches(0.3), Inches(4.88), Inches(5.9), Inches(0.62),
                "lgbm.pkl        →  ŷ = E[Y|X]                (MSE objective)", accent=GREEN)
    formula_box(s, Inches(0.3), Inches(5.58), Inches(5.9), Inches(0.62),
                "lgbm_lower.pkl  →  ŷ₀.₁ = Q₀.₁[Y|X]         (Pinball α=0.10)", accent=BLUE)
    formula_box(s, Inches(0.3), Inches(6.28), Inches(5.9), Inches(0.62),
                "lgbm_upper.pkl  →  ŷ₀.₉ = Q₀.₉[Y|X]         (Pinball α=0.90)", accent=ORANGE)

    T(s, "80% Prediction Interval  =  [ ŷ₀.₁,  ŷ₀.₉ ]",
      Inches(0.3), Inches(7.0), Inches(5.9), Inches(0.35),
      size=11, bold=True, color=NAVY)

    # Right: hyperparams + MultiOutputRegressor
    box(s, Inches(6.5), Inches(1.88), Inches(6.53), Inches(0.38), GREEN)
    T(s, "Hyperparameter & Implementasi",
      Inches(6.62), Inches(1.91), Inches(6.29), Inches(0.3),
      size=12, bold=True, color=WHITE)

    params = [
        ("n_estimators",     "400",   "Jumlah pohon"),
        ("learning_rate",    "0.03",  "Shrinkage per pohon"),
        ("num_leaves",       "63",    "Kompleksitas pohon"),
        ("min_child_samples","10",    "Regularisasi leaf"),
        ("subsample",        "0.80",  "Row subsampling"),
        ("colsample_bytree", "0.80",  "Feature subsampling"),
    ]
    box(s, Inches(6.5), Inches(2.36), Inches(6.53), Inches(0.35), GRAY_D)
    for j, lbl in enumerate(["Parameter", "Nilai", "Keterangan"]):
        xs = [Inches(6.62), Inches(8.5), Inches(9.6)]
        T(s, lbl, xs[j], Inches(2.38), Inches(1.7), Inches(0.28),
          size=10, bold=True, color=WHITE)

    for i, (param, val, note) in enumerate(params):
        y = Inches(2.78 + i * 0.4)
        bg_c = WHITE if i % 2 == 0 else GRAY_LL
        box(s, Inches(6.5), y, Inches(6.53), Inches(0.38), bg_c)
        for j, (txt_, xp) in enumerate([(param, Inches(6.62)),
                                         (val,   Inches(8.5)),
                                         (note,  Inches(9.6))]):
            T(s, txt_, xp, y + Inches(0.06),
              Inches(1.7), Inches(0.28), size=10,
              bold=(j == 0), color=GREEN if j == 1 else GRAY_D)

    box(s, Inches(6.5), Inches(5.22), Inches(6.53), Inches(0.38), TEAL)
    T(s, "MultiOutputRegressor (sklearn)",
      Inches(6.62), Inches(5.25), Inches(6.29), Inches(0.3),
      size=12, bold=True, color=WHITE)
    T(s, "Wrapper yang melatih satu LGBMRegressor terpisah untuk setiap kolom target. "
         "Menghasilkan 14 estimator per model (14 fitur) — "
         "total 14 × 3 = 42 LightGBM regressor dalam sistem.",
      Inches(6.62), Inches(5.68), Inches(6.29), Inches(0.7),
      size=10, color=GRAY_D)

    box(s, Inches(6.5), Inches(6.45), Inches(6.53), Inches(0.38), ORANGE)
    T(s, "Optuna Hyperparameter Tuning",
      Inches(6.62), Inches(6.48), Inches(6.29), Inches(0.3),
      size=12, bold=True, color=WHITE)
    T(s, "Parameter di atas merupakan default. Sistem mendukung Optuna-based tuning "
         "(tree-structured Parzen estimator) yang mengoptimasi MAE pada validation set. "
         "Best params disimpan di saved_models/{uid}/best_params.json.",
      Inches(6.62), Inches(6.9), Inches(6.29), Inches(0.5),
      size=10, color=GRAY_D)

    page_num(s, 8, 12)


# ═══════════════════════════════════════════════════════════════════════════
# S9 — EVALUASI MODEL
# ═══════════════════════════════════════════════════════════════════════════
def s09(prs):
    s = sl(prs); bg(s, GRAY_LL)
    header_band(s, "Evaluasi & Metrik",
                "Strategi validasi dan pengukuran performa model", NAVY)

    # MAE formula
    section_label(s, "Metrik Utama: MAE", Inches(0.3), Inches(1.18), Inches(3.5))
    formula_box(s, Inches(0.3), Inches(1.52), Inches(5.9), Inches(0.72),
                "MAE = (1/N) Σᵢ | yᵢ − ŷᵢ |",
                "Mean Absolute Error — dalam ruang ternormalisasi [0,1]", NAVY)

    T(s, "Nilai MAE disimpan ke model_metadata.mae_score sebagai baseline perbandingan "
         "drift detection. Semakin kecil MAE, semakin akurat model.",
      Inches(0.3), Inches(2.34), Inches(5.9), Inches(0.5),
      size=11, color=GRAY_D)

    # Train-val split
    section_label(s, "Strategi Validasi", Inches(0.3), Inches(2.95), Inches(3.5))
    splits = [
        (BLUE,   "Training Set  (90%)",
         "Digunakan untuk melatih BiLSTM dan LightGBM.\n"
         "Urutan waktu dijaga — tidak ada random shuffle.\n"
         "≈ 7.900 sekuens dari total ±8.750 jam data."),
        (ORANGE, "Validation Set  (10%)",
         "10% akhir dari data — mencerminkan kondisi terkini.\n"
         "BiLSTM: validation_split=0.1 internal.\n"
         "MAE final dihitung pada val set LightGBM."),
    ]
    for i, (c, title, desc) in enumerate(splits):
        x = Inches(0.3 + i * 3.1)
        box(s, x, Inches(3.3), Inches(2.85), Inches(0.38), c)
        T(s, title, x + Inches(0.1), Inches(3.33),
          Inches(2.65), Inches(0.3), size=11, bold=True, color=WHITE)
        box(s, x, Inches(3.68), Inches(2.85), Inches(1.42),
            WHITE, line_c=c, lw=1.0)
        T(s, desc, x + Inches(0.1), Inches(3.74),
          Inches(2.65), Inches(1.32), size=10, color=GRAY_D)

    # Drift monitoring
    section_label(s, "Monitoring Drift — Evaluasi Berkelanjutan",
                  Inches(0.3), Inches(5.23), Inches(5.5))
    box(s, Inches(0.3), Inches(5.57), Inches(5.9), Inches(1.88),
        WHITE, line_c=RED, lw=1.5)

    formula_box(s, Inches(0.42), Inches(5.68), Inches(5.66), Inches(0.65),
                "drift_score  =  MAE_recent  /  MAE_baseline",
                "Drift Score Formula", RED)
    T(s, "Window: 24 prediksi terakhir  ·  Threshold: 1.5×  ·  Update: setiap jam\n"
         "Jika drift_score > 1.5  →  status 'drifted'  →  auto-retrain di background thread",
      Inches(0.42), Inches(6.4), Inches(5.66), Inches(0.95),
      size=10, color=GRAY_D)

    # Right: per-step accuracy breakdown
    box(s, Inches(6.5), Inches(1.18), Inches(6.53), Inches(0.38), NAVY)
    T(s, "Akurasi Per Langkah Prediksi",
      Inches(6.62), Inches(1.21), Inches(6.29), Inches(0.3),
      size=12, bold=True, color=WHITE)

    T(s, "Umumnya akurasi menurun seiring bertambahnya horizon prediksi. "
         "Sistem melaporkan MAE per-step dari compute_accuracy():",
      Inches(6.62), Inches(1.66), Inches(6.29), Inches(0.52),
      size=11, color=GRAY_D)

    steps_data = [
        ("+1 jam", GREEN,  "Paling akurat — hanya 1 langkah ke depan"),
        ("+2 jam", TEAL,   "Masih sangat akurat"),
        ("+3 jam", BLUE,   "Error mulai terakumulasi"),
        ("+4 jam", ORANGE, "Uncertainty meningkat"),
        ("+5 jam", RED,    "Confidence band melebar"),
        ("+6 jam", RED,    "Horizon terjauh — akurasi terendah"),
    ]
    for i, (label, c, note) in enumerate(steps_data):
        y = Inches(2.28 + i * 0.55)
        box(s, Inches(6.5), y, Inches(1.0), Inches(0.45), c)
        T(s, label, Inches(6.5), y + Inches(0.05),
          Inches(1.0), Inches(0.35), size=11, bold=True, color=WHITE,
          align=PP_ALIGN.CENTER)
        # accuracy bar (representative)
        bar_fill = int(100 - i * 7)
        bar_w = Inches(0.045 * bar_fill)
        box(s, Inches(7.55), y + Inches(0.1), bar_w, Inches(0.25), c)
        T(s, note, Inches(7.6 + bar_w / Inches(1)), y + Inches(0.1),
          Inches(5.3), Inches(0.3), size=9, color=GRAY)

    box(s, Inches(6.5), Inches(5.65), Inches(6.53), Inches(1.8),
        BLUE_LL, line_c=BLUE, lw=1.5)
    T(s, "Catatan Implementasi", Inches(6.62), Inches(5.72),
      Inches(6.29), Inches(0.3), size=11, bold=True, color=BLUE)
    notes = [
        "MAE dihitung di ruang ternormalisasi [0,1] — nilai kecil lebih penting dari angka absolut",
        "actual_values diambil dari t_loggers dalam window ±30 menit dari target_time",
        "Jika tidak ada data aktual dalam window → baris tidak diselesaikan (NULL)",
        "compute_accuracy() menghasilkan breakdown per-step per-fitur untuk analisis mendalam",
    ]
    for i, note in enumerate(notes):
        T(s, f"  ›  {note}", Inches(6.62), Inches(6.1 + i * 0.32),
          Inches(6.29), Inches(0.3), size=10, color=GRAY_D)

    page_num(s, 9, 12)


# ═══════════════════════════════════════════════════════════════════════════
# S10 — HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════════════════
def s10(prs):
    s = sl(prs); bg(s, GRAY_LL)
    header_band(s, "Optimasi Hyperparameter  —  Optuna TPE",
                "Tree-structured Parzen Estimator untuk pencarian parameter optimal", GOLD)

    box(s, Inches(0.3), Inches(1.15), W - Inches(0.6), Inches(0.55), GOLD_L,
        line_c=GOLD, lw=1.5)
    T(s, "Pendekatan:", Inches(0.42), Inches(1.18),
      Inches(2), Inches(0.25), size=11, bold=True, color=GOLD)
    T(s, "Optuna menggunakan Tree-structured Parzen Estimator (TPE) — "
         "pendekatan Bayesian yang memodelkan distribusi prior dan posterior parameter "
         "untuk menemukan konfigurasi optimal secara efisien (lebih baik dari grid/random search).",
      Inches(0.42), Inches(1.43), W - Inches(0.84), Inches(0.25),
      size=11, color=GRAY_D)

    # Two search spaces side by side
    sides = [
        (PURPLE, "BiLSTM — Search Space", [
            ("units",       "suggest_int",         "64 – 256  (step=32)",   "Kapasitas representasi per arah"),
            ("dropout",     "suggest_float",        "0.10 – 0.40",          "Kekuatan regularisasi dropout"),
            ("batch_size",  "suggest_categorical",  "{ 16, 32, 64 }",       "Ukuran mini-batch SGD"),
            ("epochs",      "suggest_int",          "50 – 150  (step=25)",  "Batas iterasi maksimum"),
            ("patience",    "suggest_int",          "10 – 30   (step=5)",   "Early stopping window"),
        ]),
        (GREEN, "LightGBM — Search Space", [
            ("n_estimators",      "suggest_int",   "100 – 600  (step=50)", "Jumlah pohon dalam ensemble"),
            ("learning_rate",     "suggest_float", "0.01 – 0.10  (log)",  "Shrinkage rate, log scale"),
            ("num_leaves",        "suggest_int",   "15 – 127",            "Kompleksitas per pohon"),
            ("min_child_samples", "suggest_int",   "5 – 50",              "Regularisasi ukuran leaf"),
            ("subsample",         "suggest_float", "0.60 – 1.00",         "Proporsi baris per pohon"),
            ("colsample_bytree",  "suggest_float", "0.60 – 1.00",         "Proporsi fitur per pohon"),
        ]),
    ]

    for si, (c, title, params) in enumerate(sides):
        x = Inches(0.3 + si * 6.5)
        box(s, x, Inches(1.83), Inches(6.18), Inches(0.38), c)
        T(s, title, x + Inches(0.12), Inches(1.86),
          Inches(5.94), Inches(0.3), size=12, bold=True, color=WHITE)
        box(s, x, Inches(2.21), Inches(6.18), Inches(0.32), GRAY_D)
        for j, lbl in enumerate(["Parameter", "Metode Sampling", "Rentang", "Keterangan"]):
            xs = [x+Inches(0.1), x+Inches(1.35), x+Inches(2.8), x+Inches(4.1)]
            T(s, lbl, xs[j], Inches(2.24), Inches(1.2), Inches(0.26),
              size=9, bold=True, color=WHITE)
        for i, (param, method, rng, note) in enumerate(params):
            y = Inches(2.6 + i * 0.5)
            bg_c = WHITE if i % 2 == 0 else GRAY_LL
            box(s, x, y, Inches(6.18), Inches(0.48), bg_c)
            vals_and_xs = [
                (param, x + Inches(0.1), 9.5, True, c),
                (method, x + Inches(1.35), 9, False, GRAY),
                (rng, x + Inches(2.8), 9.5, True, GRAY_D),
                (note, x + Inches(4.1), 9, False, GRAY),
            ]
            for v, xp, sz, bd, clr in vals_and_xs:
                T(s, v, xp, y + Inches(0.1), Inches(1.2), Inches(0.3),
                  size=sz, bold=bd, color=clr)

    # Bottom: persist
    box(s, Inches(0.3), Inches(6.72), W - Inches(0.6), Inches(0.68),
        GOLD_L, line_c=GOLD, lw=1.5)
    T(s, "Persistensi Best Params:",
      Inches(0.42), Inches(6.75), Inches(3), Inches(0.28),
      size=11, bold=True, color=GOLD)
    T(s, "Setelah tuning, best_params disimpan ke saved_models/{uid}/best_params.json. "
         "Pada retrain berikutnya, parameter ini diload otomatis — "
         "menghindari tuning ulang yang mahal kecuali diminta eksplisit (run_tuning=True). "
         "Cache BiLSTM di-evict setelah tuning untuk memaksa load model baru.",
      Inches(0.42), Inches(7.03), W - Inches(0.84), Inches(0.3),
      size=10, color=GRAY_D)

    page_num(s, 10, 12)


# ═══════════════════════════════════════════════════════════════════════════
# S11 — IMPLEMENTASI SISTEM
# ═══════════════════════════════════════════════════════════════════════════
def s11(prs):
    s = sl(prs); bg(s, GRAY_LL)
    header_band(s, "Implementasi Sistem",
                "Arsitektur perangkat lunak dan integrasi komponen", NAVY)

    # Stack
    layers_sys = [
        (RED,    "Frontend",     "TypeScript + Tailwind CSS",
         "Highcharts (spline + arearange)  ·  Google Maps JS API  ·  Tab switcher  ·  Auto-refresh polling"),
        (ORANGE, "Web App",      "Laravel 10 / PHP 8.2",
         "ForecastService (HTTP proxy)  ·  PlatformAirQualityController  ·  CSV streaming download"),
        (BLUE,   "API Backend",  "FastAPI (Python 3.11)",
         "REST endpoints: /predictions, /accuracy, /status, /retrain, /tune  ·  JWT auth  ·  Background threads"),
        (TEAL,   "ML Engine",    "BiLSTM + LightGBM",
         "Keras/TF SavedModel + joblib pkl  ·  Per-UID saved_models/  ·  MinMax scaler  ·  Optuna tuning"),
        (NAVY,   "Database",     "MySQL 8.0 (dual DB)",
         "SENSOR_DB: t_loggers (data sensor real-time)  ·  RESULT_DB: predictions + model_metadata"),
    ]
    for i, (c, layer, tech, desc) in enumerate(layers_sys):
        y = Inches(1.18 + i * 1.2)
        box(s, Inches(0.3), y, Inches(1.4), Inches(1.0), c)
        T(s, layer, Inches(0.32), y + Inches(0.28),
          Inches(1.36), Inches(0.44), size=12, bold=True, color=WHITE,
          align=PP_ALIGN.CENTER)
        box(s, Inches(1.7), y, Inches(2.2), Inches(1.0), GRAY_L)
        T(s, tech, Inches(1.8), y + Inches(0.1),
          Inches(2.0), Inches(0.8), size=10, bold=True, color=c)
        box(s, Inches(3.9), y, Inches(9.13), Inches(1.0), WHITE, line_c=c, lw=0.8)
        T(s, desc, Inches(4.0), y + Inches(0.1),
          Inches(8.93), Inches(0.8), size=10, color=GRAY_D)
        if i < len(layers_sys) - 1:
            T(s, "↕", Inches(0.7), y + Inches(1.0),
              Inches(0.4), Inches(0.24), size=11, color=c,
              align=PP_ALIGN.CENTER)

    # Key design decisions
    box(s, Inches(0.3), Inches(7.12), W - Inches(0.6), Inches(0.34), NAVY)
    decisions = [
        "Per-UID model isolation  →  sensor baru tanpa retrain global",
        "BiLSTM cache in-memory  →  latency prediksi minimal",
        "Quantile regression  →  confidence band tanpa distributional assumption",
        "Hourly worker + drift check  →  kualitas model terjaga otomatis",
    ]
    for i, d in enumerate(decisions):
        T(s, f"  ✓  {d}", Inches(0.35 + i * 3.24), Inches(7.15),
          Inches(3.15), Inches(0.28), size=9, bold=True, color=WHITE)

    page_num(s, 11, 12)


# ═══════════════════════════════════════════════════════════════════════════
# S12 — KESIMPULAN
# ═══════════════════════════════════════════════════════════════════════════
def s12(prs):
    s = sl(prs); bg(s, NAVY)
    box(s, 0, 0, Inches(0.1), H, BLUE)
    box(s, 0, H - Inches(0.07), W, Inches(0.07), BLUE)

    T(s, "Kesimpulan", Inches(0.5), Inches(0.2), Inches(12.3), Inches(0.65),
      size=30, bold=True, color=WHITE)
    box(s, Inches(0.5), Inches(0.85), Inches(5), Inches(0.055), BLUE)

    contribs = [
        (BLUE,   "Hybrid Architecture",
         "Kombinasi BiLSTM (temporal learning) dan LightGBM (residual correction + quantile) "
         "menghasilkan prediksi yang lebih akurat dan dilengkapi interval kepercayaan 80%."),
        (TEAL,   "SSA Feature Engineering",
         "Dekomposisi SSA memisahkan tren dan osilasi sebelum learning, "
         "memungkinkan model belajar representasi yang lebih terstruktur dari sinyal sensor."),
        (GREEN,  "Per-UID Specialization",
         "Setiap sensor memiliki model, scaler, dan hyperparameter sendiri, "
         "menangkap karakteristik lokasi yang unik tanpa interferensi antar sensor."),
        (ORANGE, "Autonomous Drift Monitoring",
         "Sistem memantau akurasi prediksi setiap jam dan melakukan retrain otomatis "
         "ketika MAE melebihi 1.5× baseline — menjaga kualitas model tanpa intervensi manual."),
        (PURPLE, "Production-Ready Pipeline",
         "FastAPI backend dengan hourly worker, dual-database architecture, "
         "dashboard real-time (tabel + peta + histori + CSV export), dan 73 automated tests."),
    ]

    for i, (c, title, desc) in enumerate(contribs):
        col = i % 2 if i < 4 else None
        if i < 4:
            x = Inches(0.5 + col * 6.2)
            y = Inches(1.05 + (i // 2) * 2.0)
            w_ = Inches(5.9)
        else:
            x = Inches(0.5); y = Inches(5.08); w_ = W - Inches(1.0)

        box(s, x, y, Inches(0.06), Inches(1.7) if i < 4 else Inches(1.3), c)
        box(s, x + Inches(0.06), y, w_ - Inches(0.06), Inches(0.42),
            RGBColor(0x1E, 0x3A, 0x80))
        T(s, title, x + Inches(0.18), y + Inches(0.07),
          w_ - Inches(0.3), Inches(0.3), size=13, bold=True, color=WHITE)
        box(s, x + Inches(0.06), y + Inches(0.42),
            w_ - Inches(0.06), Inches(1.28) if i < 4 else Inches(0.88),
            RGBColor(0x16, 0x2A, 0x5E))
        T(s, desc, x + Inches(0.18), y + Inches(0.5),
          w_ - Inches(0.3), Inches(1.15) if i < 4 else Inches(0.78),
          size=11, color=RGBColor(0xBA, 0xC8, 0xFF))

    page_num(s, 12, 12)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    prs = new_prs()
    s01(prs)  # Cover
    s02(prs)  # Latar Belakang
    s03(prs)  # Tinjauan Pustaka
    s04(prs)  # Metodologi: Gambaran Sistem
    s05(prs)  # Pra-Pemrosesan
    s06(prs)  # SSA
    s07(prs)  # BiLSTM
    s08(prs)  # LightGBM
    s09(prs)  # Evaluasi & Metrik
    s10(prs)  # Hyperparameter Tuning
    s11(prs)  # Implementasi Sistem
    s12(prs)  # Kesimpulan

    out = r"C:\Users\nurch\OneDrive\Documents\project\air_quality_forecast\Air_Quality_Journal_Presentation.pptx"
    prs.save(out)
    print(f"Saved : {out}")
    print(f"Slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
