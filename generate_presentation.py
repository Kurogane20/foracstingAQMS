"""Generate Air Quality Forecast ML System presentation."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ── Colour palette ──────────────────────────────────────────────────────────
PURPLE_DARK  = RGBColor(0x4C, 0x1D, 0x95)   # purple-900
PURPLE_MID   = RGBColor(0x7C, 0x3A, 0xED)   # purple-600
PURPLE_LIGHT = RGBColor(0xED, 0xE9, 0xFE)   # purple-100
WHITE        = RGBColor(0xFF, 0xFF, 0xFF)
GRAY_DARK    = RGBColor(0x1F, 0x29, 0x37)   # gray-800
GRAY_MID     = RGBColor(0x6B, 0x72, 0x80)   # gray-500
GRAY_LIGHT   = RGBColor(0xF3, 0xF4, 0xF6)   # gray-100
GREEN        = RGBColor(0x05, 0x96, 0x69)   # emerald-600
ORANGE       = RGBColor(0xEA, 0x58, 0x0C)   # orange-600
RED          = RGBColor(0xDC, 0x26, 0x26)   # red-600
BLUE         = RGBColor(0x25, 0x63, 0xEB)   # blue-600

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)


def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def blank_slide(prs: Presentation):
    blank_layout = prs.slide_layouts[6]
    return prs.slides.add_slide(blank_layout)


def fill_bg(slide, color: RGBColor):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, left, top, width, height, color: RGBColor, radius=0):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def add_text(slide, text, left, top, width, height,
             font_size=18, bold=False, color=GRAY_DARK,
             align=PP_ALIGN.LEFT, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font_name
    return txBox


def add_bullet_box(slide, items, left, top, width, height,
                   font_size=16, color=GRAY_DARK, bullet="•"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run()
        run.text = f"{bullet}  {item}"
        run.font.size = Pt(font_size)
        run.font.color.rgb = color
        run.font.name = "Calibri"
        p.space_after = Pt(6)
    return txBox


def add_card(slide, left, top, width, height, bg_color, title, body_items,
             title_size=15, body_size=13, title_color=WHITE, body_color=GRAY_DARK):
    # Card background
    add_rect(slide, left, top, width, Inches(0.42), bg_color)
    # Title
    add_text(slide, title,
             left + Inches(0.18), top + Inches(0.05),
             width - Inches(0.36), Inches(0.35),
             font_size=title_size, bold=True, color=title_color)
    # Body background
    add_rect(slide, left, top + Inches(0.42), width, height - Inches(0.42), GRAY_LIGHT)
    # Body text
    add_bullet_box(slide, body_items,
                   left + Inches(0.18), top + Inches(0.50),
                   width - Inches(0.36), height - Inches(0.60),
                   font_size=body_size, color=GRAY_DARK)


# ── Slide helpers ────────────────────────────────────────────────────────────

def slide_01_cover(prs):
    """Cover slide"""
    sl = blank_slide(prs)
    fill_bg(sl, PURPLE_DARK)

    # Accent bar
    add_rect(sl, 0, Inches(5.8), SLIDE_W, Inches(0.06), PURPLE_MID)

    # Main title
    add_text(sl, "Air Quality Forecast",
             Inches(1.2), Inches(1.6), Inches(10.9), Inches(1.2),
             font_size=52, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    add_text(sl, "ML Prediction System",
             Inches(1.2), Inches(2.75), Inches(10.9), Inches(0.8),
             font_size=32, bold=False, color=RGBColor(0xC4, 0xB5, 0xFD),
             align=PP_ALIGN.CENTER)

    # Subtitle
    add_text(sl, "BiLSTM + LightGBM  •  6-Step Multi-Output Forecasting  •  Real-Time Dashboard",
             Inches(1.2), Inches(3.6), Inches(10.9), Inches(0.5),
             font_size=16, color=RGBColor(0xA7, 0x8B, 0xFA),
             align=PP_ALIGN.CENTER)

    # Bottom tag
    add_text(sl, "FastAPI  ·  Laravel  ·  TypeScript  ·  Highcharts  ·  Google Maps",
             Inches(1.2), Inches(6.2), Inches(10.9), Inches(0.4),
             font_size=13, color=RGBColor(0x8B, 0x5C, 0xF6),
             align=PP_ALIGN.CENTER)


def slide_02_overview(prs):
    """Project Overview"""
    sl = blank_slide(prs)
    fill_bg(sl, WHITE)
    add_rect(sl, 0, 0, Inches(0.06), SLIDE_H, PURPLE_MID)

    add_text(sl, "Project Overview",
             Inches(0.5), Inches(0.3), Inches(12), Inches(0.6),
             font_size=30, bold=True, color=PURPLE_DARK)
    add_rect(sl, Inches(0.5), Inches(0.95), Inches(4), Inches(0.04), PURPLE_MID)

    add_text(sl, "Sistem prediksi kualitas udara berbasis Machine Learning yang menganalisis "
                 "data sensor secara real-time, menghasilkan prakiraan 6 langkah ke depan (6 jam), "
                 "dan mendeteksi degradasi model secara otomatis.",
             Inches(0.5), Inches(1.1), Inches(12.5), Inches(0.9),
             font_size=15, color=GRAY_MID)

    # Three pillars
    cols = [
        (PURPLE_MID,  "Prediksi",          ["6 langkah ke depan (1-6 jam)", "14 parameter kualitas udara", "Confidence interval (band)", "Auto-refresh setiap jam"]),
        (GREEN,       "Akurasi & Drift",   ["Resolusi aktual vs prediksi", "Hitung MAE per langkah", "Deteksi drift otomatis", "Auto-retrain jika drift >1.5×"]),
        (BLUE,        "Dashboard UI",      ["Tabel prakiraan interaktif", "Grafik histori 7 hari", "Export CSV prakiraan", "Peta Google Maps sensor"]),
    ]
    for i, (color, title, items) in enumerate(cols):
        x = Inches(0.5 + i * 4.2)
        add_card(sl, x, Inches(2.15), Inches(3.9), Inches(4.5),
                 color, title, items, title_size=16, body_size=13)


def slide_03_architecture(prs):
    """System Architecture"""
    sl = blank_slide(prs)
    fill_bg(sl, GRAY_LIGHT)
    add_rect(sl, 0, 0, SLIDE_W, Inches(1.1), PURPLE_DARK)

    add_text(sl, "Arsitektur Sistem",
             Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
             font_size=28, bold=True, color=WHITE)

    # Layer boxes
    layers = [
        (BLUE,       "Sensor Hardware",   "t_loggers (SENSOR_DB)\nData sensor per jam — PM2.5, PM10, TSP, AQI, Noise, Temp, Humidity, Pressure"),
        (PURPLE_MID, "FastAPI Backend",   "BiLSTM + LightGBM · SSA Decomposition · Hyperparameter Tuning\n/predictions · /accuracy · /status · /retrain"),
        (GREEN,      "Result Database",   "predictions (RESULT_DB)\nmodel_metadata · actual_values · drift_score · mae_score"),
        (ORANGE,     "Laravel Web App",   "ForecastService · PlatformAirQualityController\nAuth · CSV Export · Proxy ke FastAPI"),
        (RED,        "Frontend (TS)",     "Highcharts · Google Maps · History Modal · Tab Switcher\nReal-time polling · Auto-refresh 10s saat training"),
    ]

    arrow = "▼"
    for i, (color, title, desc) in enumerate(layers):
        y = Inches(1.25 + i * 1.14)
        add_rect(sl, Inches(1.0), y, Inches(11.3), Inches(0.95), color)
        add_text(sl, title,
                 Inches(1.2), y + Inches(0.06), Inches(3.5), Inches(0.38),
                 font_size=14, bold=True, color=WHITE)
        add_text(sl, desc,
                 Inches(4.8), y + Inches(0.05), Inches(7.3), Inches(0.85),
                 font_size=11, color=WHITE)
        if i < len(layers) - 1:
            add_text(sl, arrow,
                     Inches(6.3), y + Inches(0.9), Inches(0.5), Inches(0.3),
                     font_size=12, color=GRAY_MID, align=PP_ALIGN.CENTER)


def slide_04_ml_model(prs):
    """ML Model"""
    sl = blank_slide(prs)
    fill_bg(sl, WHITE)
    add_rect(sl, 0, 0, SLIDE_W, Inches(0.06), PURPLE_MID)

    add_text(sl, "Model Machine Learning",
             Inches(0.5), Inches(0.2), Inches(12), Inches(0.6),
             font_size=28, bold=True, color=PURPLE_DARK)

    # Left: BiLSTM
    add_rect(sl, Inches(0.4), Inches(1.0), Inches(5.9), Inches(5.7), PURPLE_LIGHT)
    add_rect(sl, Inches(0.4), Inches(1.0), Inches(5.9), Inches(0.45), PURPLE_MID)
    add_text(sl, "BiLSTM (Deep Learning)",
             Inches(0.6), Inches(1.03), Inches(5.5), Inches(0.38),
             font_size=15, bold=True, color=WHITE)
    add_bullet_box(sl, [
        "Input: sekuens 24 jam terakhir",
        "SSA Decomposition (trend + osilasi)",
        "Bidirectional LSTM layers",
        "Output: 6 langkah × 14 fitur",
        "Confidence band (lower/upper bounds)",
        "Hyperparameter tuning via Optuna",
    ], Inches(0.6), Inches(1.55), Inches(5.5), Inches(4.8),
       font_size=13, color=GRAY_DARK)

    # Right: LightGBM
    add_rect(sl, Inches(6.8), Inches(1.0), Inches(5.9), Inches(5.7), RGBColor(0xEC, 0xFD, 0xF5))
    add_rect(sl, Inches(6.8), Inches(1.0), Inches(5.9), Inches(0.45), GREEN)
    add_text(sl, "LightGBM (Gradient Boosting)",
             Inches(7.0), Inches(1.03), Inches(5.5), Inches(0.38),
             font_size=15, bold=True, color=WHITE)
    add_bullet_box(sl, [
        "Input: fitur tabular + time features",
        "Hour-of-day, day-of-week encoding",
        "Lag features (1h, 6h, 24h)",
        "3 model: trend, seasonal, residual",
        "Ensemble dengan BiLSTM output",
        "Per-step model untuk akurasi optimal",
    ], Inches(7.0), Inches(1.55), Inches(5.5), Inches(4.8),
       font_size=13, color=GRAY_DARK)

    # Center arrow
    add_text(sl, "+", Inches(6.2), Inches(3.5), Inches(0.7), Inches(0.5),
             font_size=28, bold=True, color=PURPLE_MID, align=PP_ALIGN.CENTER)


def slide_05_features(prs):
    """14 Predicted Parameters"""
    sl = blank_slide(prs)
    fill_bg(sl, PURPLE_DARK)

    add_text(sl, "14 Parameter yang Diprediksi",
             Inches(0.5), Inches(0.2), Inches(12.3), Inches(0.65),
             font_size=28, bold=True, color=WHITE)
    add_text(sl, "Setiap prediksi menghasilkan 6 langkah waktu ke depan untuk semua parameter berikut:",
             Inches(0.5), Inches(0.85), Inches(12.3), Inches(0.4),
             font_size=14, color=RGBColor(0xC4, 0xB5, 0xFD))

    params = [
        ("PM2.5",           "µg/m³",  "Partikel halus < 2.5 µm"),
        ("PM2.5 Correction","µg/m³",  "PM2.5 terkoreksi"),
        ("PM10",            "µg/m³",  "Partikel < 10 µm"),
        ("PM10 Correction", "µg/m³",  "PM10 terkoreksi"),
        ("TSP",             "µg/m³",  "Total Suspended Particles"),
        ("TSP Correction",  "µg/m³",  "TSP terkoreksi"),
        ("Noise",           "dBA",    "Tingkat kebisingan"),
        ("Temperature",     "°C",     "Suhu udara"),
        ("Pressure",        "mmHg",   "Tekanan udara"),
        ("Humidity",        "%",      "Kelembaban relatif"),
        ("AQI PM2.5",       "index",  "Indeks kualitas udara PM2.5"),
        ("AQI PM10",        "index",  "Indeks kualitas udara PM10"),
        ("AQI TSP",         "index",  "Indeks kualitas udara TSP"),
        ("AQI Index",       "index",  "Indeks kualitas udara gabungan"),
    ]

    cols = 2
    rows = 7
    for i, (name, unit, desc) in enumerate(params):
        col = i % cols
        row = i // cols
        x = Inches(0.5 + col * 6.35)
        y = Inches(1.4 + row * 0.85)
        add_rect(sl, x, y, Inches(6.0), Inches(0.72), RGBColor(0x5B, 0x21, 0xB6))
        add_text(sl, f"{name}  ({unit})",
                 x + Inches(0.15), y + Inches(0.05), Inches(3.5), Inches(0.35),
                 font_size=13, bold=True, color=WHITE)
        add_text(sl, desc,
                 x + Inches(0.15), y + Inches(0.36), Inches(5.7), Inches(0.3),
                 font_size=11, color=RGBColor(0xC4, 0xB5, 0xFD))


def slide_06_accuracy(prs):
    """Accuracy Tracking & Drift Detection"""
    sl = blank_slide(prs)
    fill_bg(sl, WHITE)
    add_rect(sl, 0, 0, SLIDE_W, Inches(0.06), GREEN)

    add_text(sl, "Accuracy Tracking & Drift Detection",
             Inches(0.5), Inches(0.2), Inches(12.3), Inches(0.6),
             font_size=28, bold=True, color=GRAY_DARK)

    # Flow diagram
    steps = [
        (GREEN,       "Predict",        "Setiap jam, model\nmenghasilkan 6\nprediksi ke depan"),
        (BLUE,        "Resolve",        "target_time tiba →\nambil aktual dari\nt_loggers (±30 min)"),
        (ORANGE,      "Compute MAE",    "Hitung MAE per step\nper parameter dari\n24 prediksi terakhir"),
        (RED,         "Drift Check",    "Bandingkan dengan\nbaseline MAE\n× 1.5 threshold"),
        (PURPLE_MID,  "Auto-Retrain",   "Jika drift > 1.5×\nauto-retrain di\nbackground thread"),
    ]

    for i, (color, title, desc) in enumerate(steps):
        x = Inches(0.4 + i * 2.5)
        add_rect(sl, x, Inches(1.2), Inches(2.1), Inches(2.0), color)
        add_text(sl, title,
                 x + Inches(0.1), Inches(1.25), Inches(1.9), Inches(0.4),
                 font_size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text(sl, desc,
                 x + Inches(0.1), Inches(1.65), Inches(1.9), Inches(1.4),
                 font_size=11, color=WHITE, align=PP_ALIGN.CENTER)
        if i < len(steps) - 1:
            add_text(sl, "→",
                     x + Inches(2.1), Inches(1.9), Inches(0.4), Inches(0.4),
                     font_size=20, bold=True, color=GRAY_MID, align=PP_ALIGN.CENTER)

    # Status badges
    add_text(sl, "Status Model",
             Inches(0.5), Inches(3.55), Inches(5), Inches(0.4),
             font_size=16, bold=True, color=GRAY_DARK)

    badges = [
        (GREEN,       "ready",     "Model normal, prediksi aktif"),
        (BLUE,        "training",  "Sedang melatih model baru"),
        (ORANGE,      "drifted",   "Akurasi turun, retrain dijadwalkan"),
        (RED,         "error",     "Training/prediksi gagal"),
        (GRAY_MID,    "untrained", "Belum pernah dilatih"),
    ]
    for i, (color, status, desc) in enumerate(badges):
        x = Inches(0.5 + i * 2.55)
        add_rect(sl, x, Inches(4.1), Inches(2.3), Inches(0.38), color)
        add_text(sl, status,
                 x + Inches(0.1), Inches(4.11), Inches(2.1), Inches(0.34),
                 font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text(sl, desc,
                 x, Inches(4.55), Inches(2.3), Inches(0.5),
                 font_size=10, color=GRAY_MID, align=PP_ALIGN.CENTER)

    # Drift formula
    add_rect(sl, Inches(0.5), Inches(5.3), Inches(12.3), Inches(1.5), GRAY_LIGHT)
    add_text(sl, "Rumus Drift Score",
             Inches(0.7), Inches(5.4), Inches(5), Inches(0.35),
             font_size=13, bold=True, color=GRAY_DARK)
    add_text(sl, "drift_score  =  recent_MAE  ÷  baseline_MAE",
             Inches(0.7), Inches(5.75), Inches(7), Inches(0.4),
             font_size=15, bold=True, color=PURPLE_MID)
    add_text(sl, "Window: 24 prediksi terakhir   ·   Threshold: 1.5×   ·   Update: setiap jam",
             Inches(0.7), Inches(6.2), Inches(11), Inches(0.35),
             font_size=12, color=GRAY_MID)


def slide_07_ui_features(prs):
    """UI Features"""
    sl = blank_slide(prs)
    fill_bg(sl, GRAY_LIGHT)
    add_rect(sl, 0, 0, SLIDE_W, Inches(1.05), PURPLE_DARK)

    add_text(sl, "Fitur Dashboard UI",
             Inches(0.5), Inches(0.2), Inches(12.3), Inches(0.65),
             font_size=28, bold=True, color=WHITE)

    cards = [
        (PURPLE_MID, "Tabel Prakiraan", [
            "Tampilkan 6 langkah (1-6 jam)",
            "14 parameter + pilih parameter",
            "Color coding AQI (Good/Moderate/Unhealthy)",
            "Confidence band (lower–upper bounds)",
            "Trend arrow ↑↓→ per sensor",
            "Staleness warning jika data > 2 jam",
        ]),
        (GREEN, "Grafik Histori 7 Hari", [
            "Tombol history (🕐) per sensor",
            "Fetch /ml-forecast/history/{uid}",
            "Filter step=1 (prediksi 1 jam ke depan)",
            "Highcharts spline chart interaktif",
            "X-axis: waktu, Y-axis: nilai parameter",
            "Pilih parameter dari dropdown",
        ]),
        (BLUE, "Export CSV", [
            "Tombol Export CSV di modal histori",
            "Download 7 hari riwayat prediksi",
            "14 kolom: pm_25, pm_10, aqi_index...",
            "Filename: forecast_history_{uid}_{date}",
            "Streaming download via Laravel",
            "Sanitasi uid (mencegah path traversal)",
        ]),
        (ORANGE, "Peta Google Maps", [
            "Tab switcher: Table / Map",
            "Marker berwarna per status model",
            "🟢 ready  🟠 drifted  🔴 error  ⚫ lainnya",
            "InfoWindow: alias, site, status, AQI",
            "Auto fitBounds semua sensor",
            "Init sekali, lazy load saat tab aktif",
        ]),
    ]

    for i, (color, title, items) in enumerate(cards):
        col = i % 2
        row = i // 2
        x = Inches(0.4 + col * 6.45)
        y = Inches(1.15 + row * 3.05)
        add_card(sl, x, y, Inches(6.1), Inches(2.85),
                 color, title, items, title_size=15, body_size=12)


def slide_08_api(prs):
    """API Endpoints"""
    sl = blank_slide(prs)
    fill_bg(sl, WHITE)
    add_rect(sl, 0, 0, Inches(0.06), SLIDE_H, BLUE)

    add_text(sl, "API Endpoints",
             Inches(0.5), Inches(0.2), Inches(12.3), Inches(0.6),
             font_size=28, bold=True, color=GRAY_DARK)
    add_rect(sl, Inches(0.5), Inches(0.85), Inches(5), Inches(0.04), BLUE)

    endpoints = [
        # (method_color, method, path, description)
        (GREEN,  "GET",  "/predictions/{uid}",           "Prediksi terbaru untuk satu sensor"),
        (GREEN,  "GET",  "/predictions/{uid}/history",   "Histori 7 hari prediksi (days=1-90)"),
        (GREEN,  "GET",  "/predictions/all",             "Prediksi terbaru semua sensor"),
        (BLUE,   "GET",  "/status",                      "Status & metadata semua model"),
        (GREEN,  "GET",  "/accuracy/{uid}",              "Hitung MAE & resolve aktual"),
        (GREEN,  "GET",  "/accuracy/all",                "Accuracy check semua sensor"),
        (ORANGE, "POST", "/retrain/{uid}",               "Trigger retrain satu sensor"),
        (ORANGE, "POST", "/retrain-all",                 "Trigger retrain semua sensor"),
        (ORANGE, "POST", "/predict/{uid}",               "Trigger prediksi on-demand"),
        (ORANGE, "POST", "/tune/{uid}",                  "Trigger hyperparameter tuning"),
    ]

    for i, (color, method, path, desc) in enumerate(endpoints):
        y = Inches(1.1 + i * 0.6)
        add_rect(sl, Inches(0.5), y, Inches(0.75), Inches(0.42), color)
        add_text(sl, method,
                 Inches(0.5), y + Inches(0.04), Inches(0.75), Inches(0.34),
                 font_size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text(sl, path,
                 Inches(1.35), y + Inches(0.05), Inches(4.5), Inches(0.38),
                 font_size=12, bold=True, color=PURPLE_DARK)
        add_text(sl, desc,
                 Inches(5.9), y + Inches(0.05), Inches(7.0), Inches(0.38),
                 font_size=12, color=GRAY_MID)
        add_rect(sl, Inches(0.5), y + Inches(0.42), Inches(12.3), Inches(0.01), GRAY_LIGHT)


def slide_09_tech_stack(prs):
    """Tech Stack"""
    sl = blank_slide(prs)
    fill_bg(sl, PURPLE_DARK)

    add_text(sl, "Tech Stack",
             Inches(0.5), Inches(0.2), Inches(12.3), Inches(0.6),
             font_size=28, bold=True, color=WHITE)

    tech = [
        ("Backend ML",    PURPLE_MID, ["Python 3.11", "FastAPI", "BiLSTM (Keras/TF)", "LightGBM", "Optuna (tuning)", "mysql-connector-python"]),
        ("Database",      BLUE,       ["MySQL 8.0", "SENSOR_DB — t_loggers", "RESULT_DB — predictions", "model_metadata", "actual_values (JSON)", "drift_score / mae_score"]),
        ("Web App",       GREEN,      ["Laravel 10 / PHP 8.2", "ForecastService (HTTP proxy)", "PlatformAirQualityController", "Eloquent ORM", "Sanctum Auth", "Route middleware"]),
        ("Frontend",      ORANGE,     ["TypeScript", "Highcharts (spline + arearange)", "Google Maps JS API", "@googlemaps/js-api-loader", "MapsHelper (custom wrapper)", "Tailwind CSS"]),
    ]

    for i, (title, color, items) in enumerate(tech):
        x = Inches(0.4 + i * 3.15)
        add_rect(sl, x, Inches(1.1), Inches(2.9), Inches(5.7), RGBColor(0x4C, 0x1D, 0x95))
        add_rect(sl, x, Inches(1.1), Inches(2.9), Inches(0.45), color)
        add_text(sl, title,
                 x + Inches(0.12), Inches(1.13), Inches(2.7), Inches(0.38),
                 font_size=14, bold=True, color=WHITE)
        add_bullet_box(sl, items,
                       x + Inches(0.12), Inches(1.65), Inches(2.7), Inches(5.0),
                       font_size=12, color=RGBColor(0xE0, 0xD7, 0xFF))


def slide_10_closing(prs):
    """Closing"""
    sl = blank_slide(prs)
    fill_bg(sl, PURPLE_DARK)

    add_rect(sl, 0, Inches(3.3), SLIDE_W, Inches(0.08), PURPLE_MID)

    add_text(sl, "Terima Kasih",
             Inches(1.2), Inches(1.2), Inches(10.9), Inches(1.0),
             font_size=52, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    add_text(sl, "Air Quality Forecast ML System",
             Inches(1.2), Inches(2.3), Inches(10.9), Inches(0.6),
             font_size=22, color=RGBColor(0xC4, 0xB5, 0xFD), align=PP_ALIGN.CENTER)

    summary = [
        "✅  73 automated tests — semua passing",
        "✅  FastAPI + BiLSTM + LightGBM pipeline",
        "✅  Accuracy tracking + drift detection + auto-retrain",
        "✅  Dashboard: tabel, histori, CSV export, Google Maps",
    ]
    for i, line in enumerate(summary):
        add_text(sl, line,
                 Inches(2.5), Inches(3.7 + i * 0.72), Inches(8.3), Inches(0.55),
                 font_size=16, color=WHITE, align=PP_ALIGN.CENTER)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    prs = new_prs()

    slide_01_cover(prs)
    slide_02_overview(prs)
    slide_03_architecture(prs)
    slide_04_ml_model(prs)
    slide_05_features(prs)
    slide_06_accuracy(prs)
    slide_07_ui_features(prs)
    slide_08_api(prs)
    slide_09_tech_stack(prs)
    slide_10_closing(prs)

    out = r"C:\Users\nurch\OneDrive\Documents\project\air_quality_forecast\Air_Quality_Forecast_Presentation.pptx"
    prs.save(out)
    print(f"Saved: {out}")
    print(f"Slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
