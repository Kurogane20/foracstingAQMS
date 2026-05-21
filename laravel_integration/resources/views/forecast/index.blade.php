@extends('layouts.app')

@section('content')
<div class="container-fluid">
    <h2 class="mb-4">Forecast Status — Semua Sensor</h2>

    @if(session('success'))
        <div class="alert alert-success">{{ session('success') }}</div>
    @endif

    <div class="row">
        @forelse($statuses as $sensor)
        <div class="col-md-4 mb-3">
            <div class="card shadow-sm">
                <div class="card-body">
                    <h5 class="card-title font-monospace">{{ $sensor['uid'] }}</h5>
                    <span class="badge
                        @if($sensor['status'] === 'ready') bg-success
                        @elseif($sensor['status'] === 'training') bg-warning text-dark
                        @elseif($sensor['status'] === 'error') bg-danger
                        @else bg-secondary @endif">
                        {{ strtoupper($sensor['status']) }}
                    </span>
                    @if($sensor['mae_score'] ?? null)
                        <p class="mt-2 mb-1 small text-muted">MAE: {{ number_format($sensor['mae_score'], 4) }}</p>
                    @endif
                    @if($sensor['last_predicted_at'] ?? null)
                        <p class="mb-1 small text-muted">
                            Last predicted: {{ $sensor['last_predicted_at'] }}
                        </p>
                    @endif
                    <a href="{{ route('forecast.show', $sensor['uid']) }}"
                       class="btn btn-sm btn-primary mt-2">Lihat Prediksi</a>
                </div>
            </div>
        </div>
        @empty
            <div class="col-12">
                <p class="text-muted">Belum ada data. Jalankan training terlebih dahulu.</p>
            </div>
        @endforelse
    </div>
</div>
@endsection
