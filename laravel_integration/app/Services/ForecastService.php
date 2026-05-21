<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

class ForecastService
{
    private string $baseUrl;

    public function __construct()
    {
        $this->baseUrl = config('services.forecast.url');
    }

    public function getModelStatus(): array
    {
        try {
            $response = Http::timeout(5)->get("{$this->baseUrl}/status");
            return $response->successful() ? $response->json('sensors', []) : [];
        } catch (\Exception $e) {
            Log::warning("ForecastService::getModelStatus failed: {$e->getMessage()}");
            return [];
        }
    }

    public function getLatestPredictions(string $uid): array
    {
        try {
            $response = Http::timeout(5)->get("{$this->baseUrl}/predictions/{$uid}");
            return $response->successful() ? $response->json('predictions', []) : [];
        } catch (\Exception $e) {
            Log::warning("ForecastService::getLatestPredictions failed for {$uid}: {$e->getMessage()}");
            return [];
        }
    }

    public function triggerPredict(string $uid): bool
    {
        try {
            $response = Http::timeout(30)->post("{$this->baseUrl}/predict/{$uid}");
            return $response->successful();
        } catch (\Exception $e) {
            Log::error("ForecastService::triggerPredict failed for {$uid}: {$e->getMessage()}");
            return false;
        }
    }

    public function triggerPredictAll(): bool
    {
        try {
            $response = Http::timeout(120)->post("{$this->baseUrl}/predict/all");
            return $response->successful();
        } catch (\Exception $e) {
            Log::error("ForecastService::triggerPredictAll failed: {$e->getMessage()}");
            return false;
        }
    }

    public function triggerRetrainAll(): bool
    {
        try {
            $response = Http::timeout(3600)->post("{$this->baseUrl}/retrain/all");
            return $response->successful();
        } catch (\Exception $e) {
            Log::error("ForecastService::triggerRetrainAll failed: {$e->getMessage()}");
            return false;
        }
    }
}
