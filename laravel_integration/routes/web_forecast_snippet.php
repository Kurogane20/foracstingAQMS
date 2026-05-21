<?php
// Add these routes to routes/web.php:

use App\Http\Controllers\ForecastController;

Route::prefix('forecast')->name('forecast.')->group(function () {
    Route::get('/', [ForecastController::class, 'index'])->name('index');
    Route::get('/{uid}', [ForecastController::class, 'show'])->name('show');
    Route::post('/{uid}/predict', [ForecastController::class, 'predict'])->name('predict');
});
