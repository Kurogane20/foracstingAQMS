<?php

namespace App\Http\Controllers;

use App\Services\ForecastService;
use Illuminate\Http\Request;

class ForecastController extends Controller
{
    public function __construct(private ForecastService $forecast) {}

    public function index()
    {
        $statuses = $this->forecast->getModelStatus();
        return view('forecast.index', compact('statuses'));
    }

    public function show(string $uid)
    {
        $predictions = $this->forecast->getLatestPredictions($uid);
        return view('forecast.show', compact('uid', 'predictions'));
    }

    public function predict(string $uid)
    {
        $this->forecast->triggerPredict($uid);
        return redirect()->route('forecast.show', $uid)
                         ->with('success', 'Prediksi berhasil diperbarui');
    }
}
