//! CPU core utilities - getting the ID of the current code, putting the current
//! code to sleep, waking up a core.

use portable_atomic::{AtomicBool, Ordering};

use crate::interrupt::software::SoftwareInterrupt;

/// Available CPU cores
///
/// The actual number of available cores depends on the target.
#[derive(Debug, Copy, Clone, PartialEq, Eq, strum::FromRepr)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[repr(C)]
pub enum Cpu {
    /// The first core
    ProCpu = 0,
    /// The second core
    #[cfg(multi_core)]
    AppCpu = 1,
}

impl Cpu {
    /// Convert the numeric representation of a CPU to a `Cpu` enum.
    ///
    /// Return `None` if the numeric representation does not correspond to a
    /// valid CPU for the target.
    pub const fn from_primitive(cpu: usize) -> Option<Self> {
        match cpu {
            0 => Some(Self::ProCpu),
            #[cfg(multi_core)]
            1 => Some(Self::AppCpu),
            _ => None,
        }
    }
}

/// Which core the application is currently executing on
#[inline(always)]
pub fn get_core() -> Cpu {
    // This works for both RISCV and Xtensa because both
    // get_raw_core functions return zero, _or_ something
    // greater than zero; 1 in the case of RISCV and 0x2000
    // in the case of Xtensa.
    match get_raw_core() {
        0 => Cpu::ProCpu,
        #[cfg(all(multi_core, riscv))]
        1 => Cpu::AppCpu,
        #[cfg(all(multi_core, xtensa))]
        0x2000 => Cpu::AppCpu,
        _ => unreachable!(),
    }
}

/// Returns the raw value of the mhartid register.
///
/// Safety: This method should never return UNUSED_THREAD_ID_VALUE
#[cfg(riscv)]
#[inline(always)]
fn get_raw_core() -> usize {
    #[cfg(multi_core)]
    {
        riscv::register::mhartid::read()
    }

    #[cfg(not(multi_core))]
    0
}

/// Returns the result of reading the PRID register logically ANDed with 0x2000,
/// the 13th bit in the register. Espressif Xtensa chips use this bit to
/// determine the core id.
///
/// Returns either 0 or 0x2000
///
/// Safety: This method should never return UNUSED_THREAD_ID_VALUE
#[cfg(xtensa)]
#[inline(always)]
pub(crate) fn get_raw_core() -> usize {
    (xtensa_lx::get_processor_id() & 0x2000) as usize
}

/// Wake up a core sleeping in `wait`.
///
/// This function is safe to call from interrupts.
/// This function is a no-op if the core is not sleeping.
///
/// Arguments:
/// - `core`: The core to wake up.
pub fn wakeup(core: Cpu) {
    // Signal that there is work to be done.
    SIGNAL_WAKEUP[core as usize].store(true, Ordering::SeqCst);

    // If we are waking up the current core from the current core, we're done.
    // Otherwise, we need to make sure the other core wakes up.
    #[cfg(multi_core)]
    if core != get_core() {
        // We need to clear the interrupt from software. We don't actually
        // need it to trigger and run the interrupt handler, we just need to
        // kick waiti to return.
        unsafe { SoftwareInterrupt::<3>::steal().raise() };
    }
}

/// Put the current core in a low power state until woken up by `wakeup`.
///
/// Note that this function is subject to spurious wakeups, in that it will
/// return immediately if `wakeup` on the same core was called when the core was
/// not asleep in `wait`.
///
/// Therefore, it is recommended to use this function in a loop.
#[cfg(xtensa)]
pub fn wait() {
    let cpu = get_core() as usize;

    // Manual critical section implementation that only masks interrupts handlers.
    // We must not acquire the cross-core on dual-core systems because that would
    // prevent the other core from doing useful work while this core is sleeping.
    let token: critical_section::RawRestoreState;
    unsafe { core::arch::asm!("rsil {0}, 5", out(reg) token) };

    // we do not care about race conditions between the load and store operations,
    // interrupts will only set this value to true.
    if SIGNAL_WAKEUP[cpu].load(Ordering::SeqCst) {
        SIGNAL_WAKEUP[cpu].store(false, Ordering::SeqCst);

        // if there is work to do, exit critical section and loop back to polling
        unsafe {
            core::arch::asm!(
                "wsr.ps {0}",
                "rsync",
                in(reg) token
            );
        }
    } else {
        // `waiti` sets the `PS.INTLEVEL` when slipping into sleep because critical
        // sections in Xtensa are implemented via increasing `PS.INTLEVEL`.
        // The critical section ends here. Take care not add code after
        // `waiti` if it needs to be inside the CS.
        unsafe { core::arch::asm!("waiti 0") };
    }
}

/// Put the current core in a low power state until woken up by `wakeup`.
///
/// Note that this function is subject to spurious wakeups, in that it will
/// return immediately if `wakeup` on the same core was called when the core was
/// not asleep in `wait`.
///
/// Therefore, it is recommended to use this function in a loop.
#[cfg(riscv)]
pub fn wait() {
    let cpu = get_core() as usize;

    // we do not care about race conditions between the load and store operations,
    // interrupts will only set this value to true.
    critical_section::with(|_| {
        // if there is work to do, loop back to polling
        // TODO can we relax this?
        if SIGNAL_WAKEUP[cpu].load(Ordering::SeqCst) {
            SIGNAL_WAKEUP[cpu].store(false, Ordering::SeqCst);
        }
        // if not, wait for interrupt
        else {
            unsafe { core::arch::asm!("wfi") };
        }
    });
    // if an interrupt occurred while waiting, it will be serviced
    // here
}

/// A simple `block_on` utility for efficient execution of a future on the
/// current core.
pub mod task {
    use core::{
        future::Future,
        pin::Pin,
        task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
    };

    use super::{get_core, wait, wakeup, Cpu};

    static RAW_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
        |ctx| RawWaker::new(ctx, &RAW_WAKER_VTABLE),
        |ctx| wakeup(Cpu::from_primitive(ctx as usize).unwrap()),
        |ctx| wakeup(Cpu::from_primitive(ctx as usize).unwrap()),
        |_| {},
    );

    /// Run a future to completion.
    ///
    /// The execution is efficient in terms of power consumption, because the
    /// core would sleep (with `wait`) when the future cannot make progress.
    pub fn block_on<F: Future>(mut fut: F) -> F::Output {
        // Safety: we don't move the future after this line.
        let mut fut = unsafe { Pin::new_unchecked(&mut fut) };

        // Create a raw waker and then a waker for the current core.
        let raw_waker = RawWaker::new(get_core() as usize as _, &RAW_WAKER_VTABLE);
        let waker = unsafe { Waker::from_raw(raw_waker) };

        let mut cx = Context::from_waker(&waker);

        // Poll the future until completion.
        loop {
            match fut.as_mut().poll(&mut cx) {
                Poll::Ready(res) => break res,
                Poll::Pending => wait(),
            }
        }
    }
}

/// global atomic used to keep track of whether there is a wakeup signal sev()
/// is not available on either Xtensa or RISC-V
#[cfg(not(multi_core))]
static SIGNAL_WAKEUP: [AtomicBool; 1] = [AtomicBool::new(false)];
#[cfg(multi_core)]
static SIGNAL_WAKEUP: [AtomicBool; 2] = [AtomicBool::new(false), AtomicBool::new(false)];
