# Migration Guide from 0.15.0 to {{currentVersion}}

## Initialization

The `builtin-scheduler` feature has been removed. The functionality has been moved to `esp_radio_preempt_baremetal`.
`esp_radio_preempt_baremetal` needs to be initialized before calling `esp_radio::init`. Failure to do so will result in an error.

Depending on your chosen OS, you may need to use other `esp_radio_preempt_driver` implementations.

Furthermore, `esp_wifi::init` no longer requires `RNG` or a timer.

```diff
-let esp_wifi_ctrl = esp_wifi::init(timg0.timer0, Rng::new()).unwrap();
+esp_radio_preempt_baremetal::init(timg0.timer0);
+let esp_wifi_ctrl = esp_radio::init().unwrap();
```

## Importing

`esp_wifi` crate has been renamed to `esp_radio`

```diff 
- esp-wifi = "0.15.0"
+ esp-radio = "{{currentVersion}}"
``` 

## `EspWifi` prefix has been removed

```diff
- use esp_wifi::EspWifiController;
+ use esp_radio::Controller;
```
