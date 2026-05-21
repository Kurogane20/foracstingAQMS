@extends('layouts.app')

@section('content')
<div class="container-fluid">
    <div class="d-flex justify-content-between align-items-center mb-4">
        <h2>Prediksi 6 Jam — <span class="font-monospace text-primary">{{ $uid }}</span></h2>
        <div>
            <form action="{{ route('forecast.predict', $uid) }}" method="POST" class="d-inline">
                @csrf
                <button class="btn btn-outline-primary btn-sm">Perbarui Prediksi</button>
            </form>
            <a href="{{ route('forecast.index') }}" class="btn btn-outline-secondary btn-sm ms-2">
                &larr; Kembali
            </a>
        </div>
    </div>

    @if(session('success'))
        <div class="alert alert-success">{{ session('success') }}</div>
    @endif

    @if(empty($predictions))
        <div class="alert alert-warning">
            Belum ada prediksi untuk sensor ini. Klik "Perbarui Prediksi" atau tunggu cron berikutnya.
        </div>
    @else

    {{-- Parameter Selector --}}
    <div class="mb-3">
        <label class="form-label fw-bold">Tampilkan Parameter:</label>
        <select id="paramSelector" class="form-select w-auto d-inline-block ms-2" onchange="updateChart()">
            <option value="aqi_index">AQI Index</option>
            <option value="pm_25">PM2.5</option>
            <option value="pm_10">PM10</option>
            <option value="tsp">TSP</option>
            <option value="temp">Suhu (°C)</option>
            <option value="humidity">Kelembaban (%)</option>
            <option value="noise">Kebisingan (dB)</option>
            <option value="mmhg">Tekanan (mmHg)</option>
        </select>
    </div>

    {{-- Chart --}}
    <div class="card shadow-sm mb-4">
        <div class="card-body">
            <div id="forecastChart" style="height: 320px;"></div>
        </div>
    </div>

    {{-- Prediction Table --}}
    <div class="card shadow-sm">
        <div class="card-header fw-bold">Tabel Prediksi</div>
        <div class="table-responsive">
            <table class="table table-sm table-striped table-hover mb-0">
                <thead class="table-dark">
                    <tr>
                        <th>Waktu</th>
                        <th>PM2.5</th>
                        <th>PM10</th>
                        <th>TSP</th>
                        <th>Suhu</th>
                        <th>Kelembaban</th>
                        <th>Kebisingan</th>
                        <th>Tekanan</th>
                        <th>AQI</th>
                    </tr>
                </thead>
                <tbody>
                    @foreach($predictions as $pred)
                    <tr>
                        <td class="font-monospace">
                            {{ \Carbon\Carbon::parse($pred['target_time'])->format('d M H:i') }}
                        </td>
                        <td>{{ number_format($pred['pm_25'] ?? 0, 2) }}</td>
                        <td>{{ number_format($pred['pm_10'] ?? 0, 2) }}</td>
                        <td>{{ number_format($pred['tsp'] ?? 0, 2) }}</td>
                        <td>{{ number_format($pred['temp'] ?? 0, 1) }}</td>
                        <td>{{ number_format($pred['humidity'] ?? 0, 1) }}%</td>
                        <td>{{ number_format($pred['noise'] ?? 0, 1) }}</td>
                        <td>{{ number_format($pred['mmhg'] ?? 0, 1) }}</td>
                        <td><strong>{{ number_format($pred['aqi_index'] ?? 0, 1) }}</strong></td>
                    </tr>
                    @endforeach
                </tbody>
            </table>
        </div>
    </div>

    @endif
</div>
@endsection

@push('scripts')
<script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
<script>
const predictions = @json($predictions);
let chart = null;

function buildSeries(param) {
    return predictions.map(p => ({
        x: new Date(p.target_time).getTime(),
        y: parseFloat(p[param] ?? 0).toFixed(2)
    }));
}

function updateChart() {
    const param = document.getElementById('paramSelector').value;
    const label = document.getElementById('paramSelector').selectedOptions[0].text;

    if (chart) { chart.destroy(); }

    chart = new ApexCharts(document.getElementById('forecastChart'), {
        chart: { type: 'line', height: 320, toolbar: { show: false } },
        series: [{ name: 'Prediksi ' + label, data: buildSeries(param) }],
        stroke: { curve: 'smooth', dashArray: 5, width: 2 },
        xaxis: { type: 'datetime', labels: { format: 'HH:mm' } },
        yaxis: { title: { text: label }, decimalsInFloat: 2 },
        colors: ['#3b82f6'],
        tooltip: { x: { format: 'dd MMM HH:mm' } },
        markers: { size: 5 },
        annotations: {
            xaxis: [{
                x: predictions[0] ? new Date(predictions[0].target_time).getTime() : 0,
                borderColor: '#ef4444',
                label: { text: 'Sekarang +1h', style: { color: '#fff', background: '#ef4444' } }
            }]
        }
    });
    chart.render();
}

document.addEventListener('DOMContentLoaded', function() {
    if (predictions.length > 0) { updateChart(); }
});
</script>
@endpush
