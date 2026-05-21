# Laravel Integration Installation Guide

## 1. Copy Files

Copy the following files into your Laravel project:

| Source | Destination |
|--------|-------------|
| `app/Services/ForecastService.php` | `app/Services/ForecastService.php` |
| `app/Http/Controllers/ForecastController.php` | `app/Http/Controllers/ForecastController.php` |
| `resources/views/forecast/index.blade.php` | `resources/views/forecast/index.blade.php` |
| `resources/views/forecast/show.blade.php` | `resources/views/forecast/show.blade.php` |

## 2. Update config/services.php

Add to the array in `config/services.php`:
```php
'forecast' => [
    'url' => env('FORECAST_API_URL', 'http://localhost:8001'),
],
```

## 3. Update .env

Add to your `.env` file:
```
FORECAST_API_URL=http://localhost:8001
```

## 4. Update routes/web.php

Add to `routes/web.php`:
```php
use App\Http\Controllers\ForecastController;

Route::prefix('forecast')->name('forecast.')->group(function () {
    Route::get('/', [ForecastController::class, 'index'])->name('index');
    Route::get('/{uid}', [ForecastController::class, 'show'])->name('show');
    Route::post('/{uid}/predict', [ForecastController::class, 'predict'])->name('predict');
});
```

## 5. Update app/Console/Kernel.php

Add to the `schedule()` method:
```php
$schedule->call(function () {
    app(\App\Services\ForecastService::class)->triggerPredictAll();
})->hourly()->name('forecast:predict-all')->withoutOverlapping();

$schedule->call(function () {
    app(\App\Services\ForecastService::class)->triggerRetrainAll();
})->dailyAt('02:00')->name('forecast:retrain-all')->withoutOverlapping();
```

## 6. Make sure Laravel scheduler is running

Add this to your server's crontab:
```
* * * * * cd /path-to-your-laravel-project && php artisan schedule:run >> /dev/null 2>&1
```
