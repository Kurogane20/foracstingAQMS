# UI Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add prediction history chart, CSV export, and spatial map to the ML Forecast page.

**Architecture:** FastAPI gets a new `/predictions/{uid}/history` endpoint. Laravel proxies it via new controller methods and streams CSV. The frontend adds: a History modal per sensor row (Highcharts line chart, step=1 over 7 days); an Export CSV button; and a Table/Map tab that renders all sensor locations on a Google Map.

**Tech Stack:** Python 3.11, FastAPI, MySQL (RESULT_DB), Laravel/PHP, TypeScript, Highcharts, Google Maps (`@googlemaps/js-api-loader`)

---

## Files

| Action | File |
|---|---|
| Modify | `air_quality_forecast/app/db.py` |
| Modify | `air_quality_forecast/app/routers/prediction.py` |
| Modify | `air_quality_forecast/tests/test_db.py` |
| Modify | `bc-enviro-web/app/Services/ForecastService.php` |
| Modify | `bc-enviro-web/app/Http/Controllers/BeAqms/Dashboard/PlatformAirQualityController.php` |
| Modify | `bc-enviro-web/routes/web.php` |
| Modify | `bc-enviro-web/resources/views/main/be-aqms/ml-forecast/index.blade.php` |
| Modify | `bc-enviro-web/resources/js/main/be-aqms/ml-forecast/index.tsx` |

---

### Task 1: FastAPI — prediction history endpoint

**Files:**
- Modify: `air_quality_forecast/app/db.py`
- Modify: `air_quality_forecast/app/routers/prediction.py`
- Modify: `air_quality_forecast/tests/test_db.py` (append)

**Context:** RESULT_DB_CONFIG and FEATURE_COLS are already imported in db.py. The existing pattern is `mysql.connector.connect(**RESULT_DB_CONFIG)` with a `try/finally conn.close()`. The prediction router already has 4 endpoints. Tests use `unittest.mock.patch("app.db.mysql.connector.connect", ...)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_db.py`:

```python
def test_get_prediction_history_returns_list():
    from app.db import get_prediction_history
    ts = datetime(2024, 1, 1, 12, 0, 0)
    row = {
        "step": 1,
        "target_time": ts,
        "predicted_at": ts,
        **{col: 50.0 for col in FEATURE_COLS},
    }
    conn_mock = MagicMock()
    cursor_mock = MagicMock()
    conn_mock.cursor.return_value = cursor_mock
    cursor_mock.fetchall.return_value = [row]
    with patch("app.db.mysql.connector.connect", return_value=conn_mock):
        result = get_prediction_history("uid_001", days=7)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["step"] == 1
    assert isinstance(result[0]["target_time"], str)  # serialized to ISO string


def test_get_prediction_history_empty_returns_list():
    from app.db import get_prediction_history
    conn_mock = MagicMock()
    cursor_mock = MagicMock()
    conn_mock.cursor.return_value = cursor_mock
    cursor_mock.fetchall.return_value = []
    with patch("app.db.mysql.connector.connect", return_value=conn_mock):
        result = get_prediction_history("uid_001", days=7)
    assert result == []
```

Also add needed imports at top of test file (check existing imports — add `from datetime import datetime` and `from app.config import FEATURE_COLS` if not already present).

- [ ] **Step 2: Run test to verify it fails**

```
cd C:\Users\nurch\OneDrive\Documents\project\air_quality_forecast
python -m pytest tests/test_db.py::test_get_prediction_history_returns_list -v
```
Expected: FAIL with `ImportError` or `AttributeError` (function not defined yet).

- [ ] **Step 3: Add `get_prediction_history` to `app/db.py`**

Add after `get_resolved_predictions` (around line 220):

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

- [ ] **Step 4: Add history endpoint to `app/routers/prediction.py`**

Add **before** the existing `@router.get("/predictions/{uid}")` line (order matters — FastAPI matches `/predictions/{uid}/history` only if declared first):

```python
@router.get("/predictions/{uid}/history")
def get_history(uid: str, days: int = 7):
    return get_prediction_history(uid, days=days)
```

Also add `get_prediction_history` to the import from `app.db` at the top of the file.

- [ ] **Step 5: Run all db tests**

```
python -m pytest tests/test_db.py -v
```
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
cd C:\Users\nurch\OneDrive\Documents\project\air_quality_forecast
git add app/db.py app/routers/prediction.py tests/test_db.py
git commit -m "feat: add prediction history endpoint GET /predictions/{uid}/history"
```

---

### Task 2: Laravel — history proxy + CSV export + lat/lng in status

**Files:**
- Modify: `bc-enviro-web/app/Services/ForecastService.php`
- Modify: `bc-enviro-web/app/Http/Controllers/BeAqms/Dashboard/PlatformAirQualityController.php`
- Modify: `bc-enviro-web/routes/web.php`

**Context:** `ForecastService` uses `Http::withHeaders($this->headers)->timeout(N)->get(...)`. The controller's `getMlForecast()` method shows the exact user-access check pattern. `routes/web.php` adds new routes inside the `dashboard` prefix group ending around line 265. CSV streaming uses `response()->streamDownload(function() { ... }, 'filename.csv')`.

- [ ] **Step 1: Add `getPredictionHistory` to ForecastService.php**

Add after `getLatestPredictions`:

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

- [ ] **Step 2: Add `lat` + `lng` to `getAllMlForecast()` response**

In `PlatformAirQualityController::getAllMlForecast()`, find the `$result[] = [...]` array (around line 669). Add two fields:

```php
'lat'             => $platform->lat,
'lng'             => $platform->lng,
```

- [ ] **Step 3: Add `getMlForecastHistory` to the controller**

Add after `getMlForecast()`:

```php
public function getMlForecastHistory(Request $request, string $uid): \Illuminate\Http\JsonResponse {
    try {
        $platform = Platforms::with(['sitesLocation'])->where('uid', $uid)->first();
        if (!$platform) {
            return response()->json(['message' => 'Platform not found'], 404);
        }
        if ($request->user()->user_level != 'super_admin') {
            $hasAccess = UserPlatforms::where('user_id', $request->user()->id)
                ->where('platform_id', $platform->id)->exists();
            if (!$hasAccess) {
                return response()->json(['message' => 'Access denied'], 403);
            }
        }
        $history = $this->forecast->getPredictionHistory($uid);
        return response()->json($history);
    } catch (Exception $e) {
        return response()->json([
            'message' => 'Failed to load history',
            'error'   => config('app.debug') ? $e->getMessage() . ' on line ' . $e->getLine() : 'Internal server error',
        ], 500);
    }
}
```

- [ ] **Step 4: Add `exportMlForecastCsv` to the controller**

Add after `getMlForecastHistory`:

```php
public function exportMlForecastCsv(Request $request, string $uid): \Symfony\Component\HttpFoundation\StreamedResponse|\Illuminate\Http\JsonResponse {
    try {
        $platform = Platforms::where('uid', $uid)->first();
        if (!$platform) {
            return response()->json(['message' => 'Platform not found'], 404);
        }
        if ($request->user()->user_level != 'super_admin') {
            $hasAccess = UserPlatforms::where('user_id', $request->user()->id)
                ->where('platform_id', $platform->id)->exists();
            if (!$hasAccess) {
                return response()->json(['message' => 'Access denied'], 403);
            }
        }
        $history = $this->forecast->getPredictionHistory($uid, days: 7);
        $filename = "forecast_history_{$uid}_" . now()->format('Y-m-d') . '.csv';
        $headers = ['Content-Type' => 'text/csv', 'Content-Disposition' => "attachment; filename=\"{$filename}\""];
        $columns = ['predicted_at', 'target_time', 'step', 'pm_25', 'pm_10', 'tsp', 'noise', 'temp', 'humidity', 'mmhg', 'aqi_index', 'aqi_index_pm25', 'aqi_index_pm10', 'aqi_index_tsp'];
        return response()->streamDownload(function() use ($history, $columns) {
            $out = fopen('php://output', 'w');
            fputcsv($out, $columns);
            foreach ($history as $row) {
                $line = [];
                foreach ($columns as $col) {
                    $line[] = $row[$col] ?? '';
                }
                fputcsv($out, $line);
            }
            fclose($out);
        }, $filename, $headers);
    } catch (Exception $e) {
        return response()->json([
            'message' => 'Failed to export CSV',
            'error'   => config('app.debug') ? $e->getMessage() . ' on line ' . $e->getLine() : 'Internal server error',
        ], 500);
    }
}
```

- [ ] **Step 5: Add routes to `routes/web.php`**

Inside the `dashboard` prefix group, after the `tune/{uid}` route (line 265):

```php
Route::get('ml-forecast/history/{uid}', [PlatformAirQualityController::class, 'getMlForecastHistory']);
Route::get('ml-forecast/history/{uid}/export', [PlatformAirQualityController::class, 'exportMlForecastCsv']);
```

- [ ] **Step 6: Verify PHP syntax**

```
cd C:\Users\nurch\OneDrive\Documents\project\bc-enviro-web
php artisan route:list --path=ml-forecast 2>&1 | head -20
```
Expected: new routes appear without errors.

- [ ] **Step 7: Commit**

```bash
cd C:\Users\nurch\OneDrive\Documents\project\bc-enviro-web
git add app/Services/ForecastService.php app/Http/Controllers/BeAqms/Dashboard/PlatformAirQualityController.php routes/web.php
git commit -m "feat: add prediction history proxy + CSV export + lat/lng in forecast status"
```

---

### Task 3: Frontend — prediction history modal + export button

**Files:**
- Modify: `bc-enviro-web/resources/views/main/be-aqms/ml-forecast/index.blade.php`
- Modify: `bc-enviro-web/resources/js/main/be-aqms/ml-forecast/index.tsx`

**Context:** The page already has a `modalMlForecastChart` modal for per-sensor forecast charts. The history modal follows the same structure (`modal`, `modal-main`, `modal-head`, `modal-body`). `showModalDialog` / `closeModalDialog` from `@/js/plugins/modal` are already imported. The Actions column `<td>` builds HTML buttons: `trainBtn`, `tuneBtn`, `chartBtn` joined and injected into a `<div class="flex items-center justify-center gap-1">`. The wire-up loops use `tableBody.querySelectorAll<HTMLButtonElement>('.btnX')`.

- [ ] **Step 1: Add history modal to `index.blade.php`**

After the existing forecast chart modal (closing `</div>` of `.modalMlForecastChart`, around line 155), add:

```html
{{-- History modal --}}
<div class="modal hidden modalHistoryChart">
    <div class="modal-main !w-[720px]">
        <div class="modal-head">
            <div class="flex justify-between items-center">
                <div class="modal-title historyChartTitle">
                    <i class="fas fa-clock-rotate-left mr-2"></i> Forecast History
                </div>
                <div class="cursor-pointer closeHistoryModal">
                    <i class="fas fa-close"></i>
                </div>
            </div>
        </div>
        <div class="modal-body">
            <div class="historyChartBody h-[320px]"></div>
        </div>
        <div class="modal-footer flex justify-end p-3 border-t">
            <a class="btnExportCsv inline-flex items-center gap-1 px-3 py-1.5 text-[12px] bg-green-50 hover:bg-green-100 text-green-700 border border-green-200 rounded-lg transition-colors" href="#" target="_blank">
                <i class="fas fa-download"></i> Export CSV
            </a>
        </div>
    </div>
</div>
```

- [ ] **Step 2: Add `lat`, `lng` to `SensorForecast` interface in `index.tsx`**

Find the `SensorForecast` interface. After `drift_score: number | null` and `last_accuracy_check_at: string | null`, add:

```ts
lat: number | null
lng: number | null
```

- [ ] **Step 3: Add history modal DOM references**

After the existing modal-related `querySelector` calls (around line 137-139 in `index.tsx`):

```ts
const modalHistory = document.querySelector<HTMLElement>('.modalHistoryChart')!
const historyChartBody = modalHistory.querySelector<HTMLElement>('.historyChartBody')!
const historyChartTitle = modalHistory.querySelector<HTMLElement>('.historyChartTitle')!
const btnExportCsv = modalHistory.querySelector<HTMLAnchorElement>('.btnExportCsv')!
```

- [ ] **Step 4: Wire close button for history modal**

Add after the existing `closeModalForm` wire-up:

```ts
modalHistory.querySelectorAll('.closeHistoryModal').forEach(btn => {
    btn.addEventListener('click', () => closeModalDialog(modalHistory))
})
```

- [ ] **Step 5: Add `historyBtn` to the Actions cell in `renderTable`**

In the `renderTable` function, add after `chartBtn`:

```ts
const historyBtn = `<button class="btnShowHistory inline-flex items-center gap-1 text-[11px] px-2 py-0.5 bg-gray-50 hover:bg-gray-100 text-gray-600 border border-gray-200 rounded transition-colors" data-uid="${sensor.uid}" data-alias="${sensor.uid_alias}">
    <i class="fas fa-clock-rotate-left"></i>
</button>`
```

Add `${historyBtn}` to the `<div class="flex items-center justify-center gap-1">` alongside the other buttons.

- [ ] **Step 6: Add `showHistoryChart` function**

Add a new function after `showForecastChart`:

```ts
async function showHistoryChart(uid: string, alias: string) {
    const param = paramSelect.value
    const label = PARAM_LABELS[param] || param

    historyChartTitle.innerHTML = `<i class="fas fa-clock-rotate-left mr-2"></i> ${alias} — 7-day history`
    btnExportCsv.href = `/aqms/dashboard/ml-forecast/history/${uid}/export`

    showModal(modalHistory, null, async () => {
        historyChartBody.innerHTML = '<div class="flex items-center justify-center h-full text-gray-400"><i class="fas fa-spinner fa-pulse mr-2"></i> Loading...</div>'

        const response = await fetch(`/aqms/dashboard/ml-forecast/history/${uid}`, {
            headers: { 'X-CSRF-TOKEN': csrfToken }
        })

        if (!response.ok) {
            historyChartBody.innerHTML = '<div class="flex items-center justify-center h-full text-red-400">Failed to load history.</div>'
            return
        }

        const data: Array<Record<string, any>> = await response.json()
        const step1 = data
            .filter(r => r.step === 1)
            .sort((a, b) => a.target_time < b.target_time ? -1 : 1)

        if (!step1.length) {
            historyChartBody.innerHTML = '<div class="flex items-center justify-center h-full text-gray-400">No history data available.</div>'
            return
        }

        const seriesData: [number, number][] = step1.map(r => [
            toUtcMs(r.target_time),
            +(r[param] ?? 0)
        ])

        historyChartBody.innerHTML = ''
        Highcharts.chart({
            chart: { renderTo: historyChartBody, type: 'spline', height: 300, style: { fontFamily: 'Arial, sans-serif' } },
            title: { text: null },
            credits: { enabled: false },
            xAxis: { type: 'datetime', labels: { style: { fontSize: '10px', color: '#999' } } },
            yAxis: { title: { text: label }, labels: { style: { fontSize: '10px', color: '#666' } }, gridLineColor: '#eee', gridLineDashStyle: 'Dash' },
            legend: { enabled: false },
            tooltip: {
                backgroundColor: 'white', borderWidth: 0, borderRadius: 8, shadow: true,
                formatter: function () {
                    return `<b>${Highcharts.dateFormat('%d %b %H:%M', this.x as number)}</b><br/>${label}: <b>${(this.y as number).toFixed(2)}</b>`
                }
            },
            plotOptions: { spline: { lineWidth: 2, marker: { enabled: false } } },
            series: [{ type: 'spline', name: label, data: seriesData, color: '#7c3aed' }]
        })
    })
}
```

- [ ] **Step 7: Wire up history buttons**

Add after the existing `btnShowChart` wire-up loop:

```ts
tableBody.querySelectorAll<HTMLButtonElement>('.btnShowHistory').forEach(btn => {
    btn.addEventListener('click', () => {
        const uid = btn.dataset.uid!
        const alias = btn.dataset.alias!
        showHistoryChart(uid, alias)
    })
})
```

- [ ] **Step 8: TypeScript check**

```
cd C:\Users\nurch\OneDrive\Documents\project\bc-enviro-web
npx tsc --noEmit 2>&1 | grep "ml-forecast/index.tsx"
```
Expected: no output (no errors in this file).

- [ ] **Step 9: Commit**

```bash
cd C:\Users\nurch\OneDrive\Documents\project\bc-enviro-web
git add resources/views/main/be-aqms/ml-forecast/index.blade.php resources/js/main/be-aqms/ml-forecast/index.tsx
git commit -m "feat: add prediction history modal and export CSV button"
```

---

### Task 4: Frontend — spatial map tab

**Files:**
- Modify: `bc-enviro-web/resources/views/main/be-aqms/ml-forecast/index.blade.php`
- Modify: `bc-enviro-web/resources/js/main/be-aqms/ml-forecast/index.tsx`

**Context:** `MapsHelper` is used in the AQMS dashboard (`index.tsx`) via `import MapsHelper from "@/js/plugins/mapsHelper"`. It returns `{map, google}` in a Promise. The marker color is set via `icon.fillColor`. Sensor data (with lat/lng) is in the `allData` array available in the closure. The table is in a `.card` element. The blade already has a `summaryBar` above the table card.

- [ ] **Step 1: Add tab switcher and map container to `index.blade.php`**

Find the `{{-- Summary Bar --}}` section (around line 27). **Before** it, add:

```html
{{-- Tab Switcher --}}
<div class="flex items-center gap-2 mb-4">
    <button class="btnTabTable inline-flex items-center gap-1.5 px-4 py-2 text-[13px] font-semibold rounded-lg bg-purple-600 text-white transition-colors">
        <i class="fas fa-table"></i> Table
    </button>
    <button class="btnTabMap inline-flex items-center gap-1.5 px-4 py-2 text-[13px] font-semibold rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-600 transition-colors">
        <i class="fas fa-map-location-dot"></i> Map
    </button>
</div>
```

After the `{{-- AQI Legend --}}` section and before the `{{-- Table --}}` card (around line 102), add:

```html
{{-- Map Container --}}
<div id="mlForecastMapContainer" class="hidden h-[480px] rounded-lg overflow-hidden border border-gray-200 mb-4"></div>
```

- [ ] **Step 2: Import MapsHelper in `index.tsx`**

Add at the top of `index.tsx` (with the existing imports):

```ts
import MapsHelper from '@/js/plugins/mapsHelper'
```

- [ ] **Step 3: Add tab + map DOM references**

After the existing `querySelector` calls for modals, add:

```ts
const btnTabTable = document.querySelector<HTMLButtonElement>('.btnTabTable')!
const btnTabMap = document.querySelector<HTMLButtonElement>('.btnTabMap')!
const mapContainer = document.querySelector<HTMLDivElement>('#mlForecastMapContainer')!
const tableCard = document.querySelector<HTMLElement>('.forecastTableBody')!.closest<HTMLElement>('.card')!
```

- [ ] **Step 4: Add map init state**

After the `allData` / `activeChart` declarations:

```ts
let mapInstance: google.maps.Map | null = null
let mapMarkers: google.maps.Marker[] = []
```

- [ ] **Step 5: Add `initOrUpdateMap` function**

Add after `updateCountdown`:

```ts
function markerColor(status: string): string {
    if (status === 'ready') return '#22c55e'
    if (status === 'drifted') return '#f97316'
    if (status === 'error') return '#ef4444'
    return '#9ca3af'
}

async function initOrUpdateMap(data: SensorForecast[]) {
    const sensorsWithCoords = data.filter(s => s.lat != null && s.lng != null)
    if (!sensorsWithCoords.length) {
        mapContainer.innerHTML = '<div class="flex items-center justify-center h-full text-gray-400">No sensor coordinates available.</div>'
        return
    }

    if (!mapInstance) {
        const mapsHelper = new MapsHelper()
        const { map, google: gApi } = await mapsHelper.mapsConfig(mapContainer)
        mapInstance = map

        const bounds = new gApi.maps.LatLngBounds()
        for (const sensor of sensorsWithCoords) {
            const step1 = sensor.predictions.find(p => p.step === 1)
            const aqi = step1 ? step1.aqi_index.toFixed(1) : '—'
            const pos = { lat: sensor.lat as number, lng: sensor.lng as number }
            const color = markerColor(sensor.model_status)

            const marker = new gApi.maps.Marker({
                position: pos,
                map,
                title: sensor.uid_alias,
                icon: {
                    path: 'M215.7 499.2C267 435 384 279.4 384 192C384 86 298 0 192 0S0 86 0 192c0 87.4 117 243 168.3 307.2c12.3 15.3 35.1 15.3 47.4 0zM192 128a64 64 0 1 1 0 128 64 64 0 1 1 0-128z',
                    scale: 0.055,
                    strokeWeight: 0.2,
                    strokeColor: color,
                    strokeOpacity: 1,
                    fillColor: color,
                    fillOpacity: 0.9,
                    anchor: new gApi.maps.Point(384 / 2, 512),
                },
            })

            const infowindow = new gApi.maps.InfoWindow({
                content: `<div style="font-size:12px;min-width:160px">
                    <div style="font-weight:700;margin-bottom:4px">${sensor.uid_alias}</div>
                    <div style="color:#6b7280;margin-bottom:4px">${sensor.siteName}</div>
                    <div style="margin-bottom:4px">${statusBadge(sensor.model_status)}</div>
                    <div>AQI +1h: <b>${aqi}</b></div>
                </div>`
            })
            marker.addListener('click', () => infowindow.open(map, marker))
            mapMarkers.push(marker)
            bounds.extend(pos)
        }
        map.fitBounds(bounds)
    }
}
```

- [ ] **Step 6: Wire tab buttons**

Add after the `btnPredictAll` event listener:

```ts
btnTabTable.addEventListener('click', () => {
    tableCard.classList.remove('hidden')
    summaryBar.classList.remove('hidden')
    mapContainer.classList.add('hidden')
    btnTabTable.className = btnTabTable.className.replace('bg-gray-100 hover:bg-gray-200 text-gray-600', 'bg-purple-600 text-white')
    btnTabMap.className = btnTabMap.className.replace('bg-purple-600 text-white', 'bg-gray-100 hover:bg-gray-200 text-gray-600')
})

btnTabMap.addEventListener('click', () => {
    tableCard.classList.add('hidden')
    summaryBar.classList.add('hidden')
    mapContainer.classList.remove('hidden')
    btnTabMap.className = btnTabMap.className.replace('bg-gray-100 hover:bg-gray-200 text-gray-600', 'bg-purple-600 text-white')
    btnTabTable.className = btnTabTable.className.replace('bg-purple-600 text-white', 'bg-gray-100 hover:bg-gray-200 text-gray-600')
    initOrUpdateMap(allData)
})
```

- [ ] **Step 7: TypeScript check**

```
cd C:\Users\nurch\OneDrive\Documents\project\bc-enviro-web
npx tsc --noEmit 2>&1 | grep "ml-forecast/index.tsx"
```
Expected: no output.

- [ ] **Step 8: Commit**

```bash
cd C:\Users\nurch\OneDrive\Documents\project\bc-enviro-web
git add resources/views/main/be-aqms/ml-forecast/index.blade.php resources/js/main/be-aqms/ml-forecast/index.tsx
git commit -m "feat: add spatial map tab with Google Maps sensor markers"
```

---

## Self-Review

**Spec coverage:**
- ✅ `get_prediction_history(uid, days)` in db.py (Task 1)
- ✅ `GET /predictions/{uid}/history` endpoint (Task 1)
- ✅ `ForecastService::getPredictionHistory()` (Task 2)
- ✅ `getAllMlForecast` returns lat/lng (Task 2)
- ✅ `getMlForecastHistory` controller method (Task 2)
- ✅ `exportMlForecastCsv` controller method + CSV columns (Task 2)
- ✅ 2 new Laravel routes (Task 2)
- ✅ History modal in blade (Task 3)
- ✅ History chart (step=1, 7-day, selected param) (Task 3)
- ✅ Export CSV button linked to `/history/{uid}/export` (Task 3)
- ✅ Tab switcher in blade (Task 4)
- ✅ Map container in blade (Task 4)
- ✅ Google Maps markers colored by model_status (Task 4)
- ✅ Marker infowindow with uid_alias, siteName, status, AQI (Task 4)

**Type consistency:**
- `get_prediction_history` imported in `prediction.py` and in test ✅
- `SensorForecast.lat/lng` used in `initOrUpdateMap` via `s.lat` / `s.lng` ✅
- `sensor.lat as number` cast only after null check `s.lat != null` ✅
- `statusBadge(sensor.model_status)` called in infowindow — function already defined ✅
