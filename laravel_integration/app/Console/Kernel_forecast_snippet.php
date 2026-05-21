<?php
// Add these scheduled tasks to the schedule() method in app/Console/Kernel.php:
//
// use App\Services\ForecastService;
//
// protected function schedule(Schedule $schedule): void
// {
//     // ... existing schedules ...
//
//     // Trigger prediction for all sensors every hour
//     $schedule->call(function () {
//         app(\App\Services\ForecastService::class)->triggerPredictAll();
//     })->hourly()->name('forecast:predict-all')->withoutOverlapping();
//
//     // Retrain all models every night at 02:00
//     $schedule->call(function () {
//         app(\App\Services\ForecastService::class)->triggerRetrainAll();
//     })->dailyAt('02:00')->name('forecast:retrain-all')->withoutOverlapping();
// }
