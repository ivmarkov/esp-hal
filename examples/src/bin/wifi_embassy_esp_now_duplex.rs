//! Embassy ESP-NOW Example (Duplex)
//!
//! Asynchronously broadcasts, receives and sends messages via esp-now in
//! multiple embassy tasks
//!
//! Because of the huge task-arena size configured this won't work on ESP32-S2

//% FEATURES: embassy esp-radio esp-radio/esp-now esp-hal/unstable
//% CHIPS: esp32 esp32s2 esp32s3 esp32c2 esp32c3 esp32c6

#![no_std]
#![no_main]

use embassy_executor::Spawner;
use embassy_sync::{blocking_mutex::raw::NoopRawMutex, mutex::Mutex};
use embassy_time::{Duration, Ticker};
use esp_alloc as _;
use esp_backtrace as _;
use esp_hal::{clock::CpuClock, timer::timg::TimerGroup};
use esp_println::println;
use esp_radio::{
    Controller,
    esp_now::{BROADCAST_ADDRESS, EspNowManager, EspNowReceiver, EspNowSender, PeerInfo},
};

esp_bootloader_esp_idf::esp_app_desc!();

// When you are okay with using a nightly compiler it's better to use https://docs.rs/static_cell/2.1.0/static_cell/macro.make_static.html
macro_rules! mk_static {
    ($t:ty,$val:expr) => {{
        static STATIC_CELL: static_cell::StaticCell<$t> = static_cell::StaticCell::new();
        #[deny(unused_attributes)]
        let x = STATIC_CELL.uninit().write(($val));
        x
    }};
}

#[esp_hal_embassy::main]
async fn main(spawner: Spawner) -> ! {
    esp_println::logger::init_logger_from_env();
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);

    esp_alloc::heap_allocator!(size: 72 * 1024);

    let timg0 = TimerGroup::new(peripherals.TIMG0);
    esp_radio_preempt_baremetal::init(timg0.timer0);

    let esp_wifi_ctrl = &*mk_static!(Controller<'static>, esp_radio::init().unwrap());

    let wifi = peripherals.WIFI;
    let (mut controller, interfaces) = esp_radio::wifi::new(&esp_wifi_ctrl, wifi).unwrap();
    controller.set_mode(esp_radio::wifi::WifiMode::Sta).unwrap();
    controller.start().unwrap();

    let esp_now = interfaces.esp_now;
    esp_now.set_channel(11).unwrap();

    println!("esp-now version {}", esp_now.version().unwrap());

    cfg_if::cfg_if! {
        if #[cfg(feature = "esp32")] {
            let timg1 = TimerGroup::new(peripherals.TIMG1);
            esp_hal_embassy::init(timg1.timer0);
        } else {
            use esp_hal::timer::systimer::SystemTimer;
            let systimer = SystemTimer::new(peripherals.SYSTIMER);
            esp_hal_embassy::init(systimer.alarm0);
        }
    }

    let (manager, sender, receiver) = esp_now.split();
    let manager = mk_static!(EspNowManager<'static>, manager);
    let sender = mk_static!(
        Mutex::<NoopRawMutex, EspNowSender<'static>>,
        Mutex::<NoopRawMutex, _>::new(sender)
    );

    spawner.spawn(listener(manager, receiver)).ok();
    spawner.spawn(broadcaster(sender)).ok();

    let mut ticker = Ticker::every(Duration::from_millis(500));
    loop {
        ticker.next().await;
        let peer = match manager.fetch_peer(false) {
            Ok(peer) => peer,
            Err(_) => {
                if let Ok(peer) = manager.fetch_peer(true) {
                    peer
                } else {
                    continue;
                }
            }
        };

        println!("Send hello to peer {:?}", peer.peer_address);
        let mut sender = sender.lock().await;
        let status = sender.send_async(&peer.peer_address, b"Hello Peer.").await;
        println!("Send hello status: {:?}", status);
    }
}

#[embassy_executor::task]
async fn broadcaster(sender: &'static Mutex<NoopRawMutex, EspNowSender<'static>>) {
    let mut ticker = Ticker::every(Duration::from_secs(1));
    loop {
        ticker.next().await;

        println!("Send Broadcast...");
        let mut sender = sender.lock().await;
        let status = sender.send_async(&BROADCAST_ADDRESS, b"Hello.").await;
        println!("Send broadcast status: {:?}", status);
    }
}

#[embassy_executor::task]
async fn listener(manager: &'static EspNowManager<'static>, mut receiver: EspNowReceiver<'static>) {
    loop {
        let r = receiver.receive_async().await;
        println!("Received {:?}", r.data());
        if r.info.dst_address == BROADCAST_ADDRESS {
            if !manager.peer_exists(&r.info.src_address) {
                manager
                    .add_peer(PeerInfo {
                        interface: esp_radio::esp_now::EspNowWifiInterface::Sta,
                        peer_address: r.info.src_address,
                        lmk: None,
                        channel: None,
                        encrypt: false,
                    })
                    .unwrap();
                println!("Added peer {:?}", r.info.src_address);
            }
        }
    }
}
