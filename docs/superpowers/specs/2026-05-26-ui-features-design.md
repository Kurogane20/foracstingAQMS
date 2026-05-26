# UI Features (Sub-project C) — Design Spec

## Goal

Add three UI features to the ML Forecast page: 7-day prediction history chart, CSV export, and a Google Maps spatial view of sensor locations.

## Tech Stack

Python 3.11, FastAPI, MySQL (RESULT_DB `predictions`), Laravel/PHP, TypeScript, Highcharts, Google Maps (`@googlemaps/js-api-loader`)

---

## Section 1: FastAPI — Prediction History Endpoint

### db.py

New function `get_prediction_history(uid, days=7)`:

```python
def get_prediction_history(uid: str, days: int = 7) -> list[dict]:
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            f"""
            SELECT step, target_time, predicted_at, {', '.join(FEATURE_COLS)}
            FROM predictions
            WHERE uid = %s
              AND predicted_at >= NOW() - INTERVAL %s DAY
            ORDER BY predicted_at ASC, step ASC
            """,
            (uid, days),
        )
        rows = []
        for r in cursor.fetchall():
            row = dict(r)
            for k, v in row.items():
                if isinstance(v, datetime):
                    row[k] = v.isoformat()
            rows.append(row)
        return rows
    finally:
        conn.close()
```

### routers/prediction.py

New endpoint added **before** `/predictions/{uid}` to prevent routing conflicts:

```python
@router.get("/predictions/{uid}/history")
def get_history(uid: str, days: int = 7):
    from app.db import get_prediction_history
    return get_prediction_history(uid, days=days)
```

No auth required (read-only, consistent with existing `/predictions/{uid}`).

---

## Section 2: Laravel Changes

### ForecastService.php

New method:

```php
public function getPredictionHistory(string $uid, int $days = 7): array {
    try {
        $response = Http::withHeaders($this->headers)->timeout(10)
            ->get("{$this->baseUrl}/predictions/{$uid}/history", ['days' => $days]);
        return $response->successful() ? $response->json([]) : [];
    } catch (\Exception $e) {
        Log::warning("ForecastService::getPredictionHistory failed for {$uid}: {$e->getMessage()}");
        return [];
    }
}
```

### PlatformAirQualityController.php

Three changes:

**1. `getAllMlForecast()`** — add `lat` and `lng` to each result row:
```php
$result[] = [
    // ... existing fields ...
    'lat'             => $platform->lat,
    'lng'             => $platform->lng,
];
```

**2. `getMlForecastHistory(Request $request, string $uid)`** — new method:
- Check platform exists + user access (same pattern as `getMlForecast`)
- Call `$this->forecast->getPredictionHistory($uid)`
- Return `response()->json($history)`

**3. `exportMlForecastCsv(Request $request, string $uid)`** — new method:
- Check platform exists + user access
- Call `$this->forecast->getPredictionHistory($uid, days: 7)`
- Stream CSV with `response()->streamDownload()`
- Filename: `forecast_history_{$uid}_{date}.csv`
- CSV columns: `predicted_at, target_time, step, pm_25, pm_10, tsp, noise, temp, humidity, mmhg, aqi_index, aqi_index_pm25, aqi_index_pm10, aqi_index_tsp`

### routes/web.php

Add inside the `dashboard` prefix group (after the existing `tune/{uid}` route):

```php
Route::get('ml-forecast/history/{uid}', [PlatformAirQualityController::class, 'getMlForecastHistory']);
Route::get('ml-forecast/history/{uid}/export', [PlatformAirQualityController::class, 'exportMlForecastCsv']);
```

---

## Section 3: Frontend Changes

### index.blade.php

Four additions:

**1. Tab switcher** — above the summary bar:
```html
<div class="flex items-center gap-2 mb-4">
    <button class="btnTabTable ...active styles...">Table</button>
    <button class="btnTabMap ...inactive styles...">Map</button>
</div>
```

**2. Map container** — below the summary bar, hidden initially:
```html
<div id="mlForecastMapContainer" class="hidden h-[480px] rounded-lg overflow-hidden border border-gray-200"></div>
```

**3. History modal** — similar structure to the existing forecast chart modal:
```html
<div class="modal hidden modalHistoryChart">
    <div class="modal-main !w-[720px]">
        <div class="modal-head">
            <div class="flex justify-between items-center">
                <div class="modal-title historyChartTitle">...</div>
                <div class="cursor-pointer closeHistoryModal"><i class="fas fa-close"></i></div>
            </div>
        </div>
        <div class="modal-body">
            <div class="historyChartBody h-[320px]"></div>
        </div>
        <div class="modal-footer flex justify-end">
            <a class="btnExportCsv inline-flex items-center gap-1 px-3 py-1.5 text-[12px] bg-green-50 hover:bg-green-100 text-green-700 border border-green-200 rounded-lg">
                <i class="fas fa-download"></i> Export CSV
            </a>
        </div>
    </div>
</div>
```

### index.tsx

**Interface additions to `SensorForecast`:**
```ts
lat: number | null
lng: number | null
```

**History button** per row in the Actions `<div>` (after `chartBtn`):
```ts
const historyBtn = `<button class="btnShowHistory ..." data-uid="${sensor.uid}" data-alias="${sensor.uid_alias}">
    <i class="fas fa-clock-rotate-left"></i>
</button>`
```

**History modal logic:**
- On click: fetch `/aqms/dashboard/ml-forecast/history/{uid}` → filter rows where `step === 1` → sort by `target_time` → render Highcharts spline chart
- Chart: X = target_time (datetime), Y = selected param value
- Export CSV `href`: `/aqms/dashboard/ml-forecast/history/{uid}/export`

**Map tab logic:**
- Tab switcher: clicking "Map" hides `.card` table wrapper + shows `#mlForecastMapContainer`; clicking "Table" reverses
- Map initializes once (`mapInitialized` flag); on first Map tab click, call `new MapsHelper().mapsConfig(mapContainer)` then place markers
- Marker fill color by `model_status`:
  - `ready` → `#22c55e`
  - `drifted` → `#f97316`
  - `error` → `#ef4444`
  - `untrained` / default → `#9ca3af`
- Sensors without lat/lng are skipped
- Infowindow content: uid_alias, siteName, status badge HTML, current AQI prediction (step=1 `aqi_index` from predictions array, or `—` if none)

---

## Files to Create / Modify

| Action | File |
|---|---|
| Modify | `app/db.py` — add `get_prediction_history()` |
| Modify | `app/routers/prediction.py` — add history endpoint |
| Modify | `tests/test_db.py` — test for `get_prediction_history` |
| Modify | `bc-enviro-web/app/Services/ForecastService.php` — add `getPredictionHistory()` |
| Modify | `bc-enviro-web/app/Http/Controllers/BeAqms/Dashboard/PlatformAirQualityController.php` — 3 changes |
| Modify | `bc-enviro-web/routes/web.php` — 2 new routes |
| Modify | `bc-enviro-web/resources/views/main/be-aqms/ml-forecast/index.blade.php` — tab + map + history modal |
| Modify | `bc-enviro-web/resources/js/main/be-aqms/ml-forecast/index.tsx` — history modal + map tab logic |
