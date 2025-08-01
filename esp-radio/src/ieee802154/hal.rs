use core::ops::{BitAnd, BitOr};

use esp_hal::peripherals::IEEE802154;

use super::{IEEE802154_FRAME_EXT_ADDR_SIZE, pib::CcaMode};

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub(crate) enum Event {
    TxDone          = 1 << 0,
    RxDone          = 1 << 1,
    AckTxDone       = 1 << 2,
    AckRxDone       = 1 << 3,
    RxAbort         = 1 << 4,
    TxAbort         = 1 << 5,
    EdDone          = 1 << 6,
    Timer0Overflow  = 1 << 8,
    Timer1Overflow  = 1 << 9,
    ClockCountMatch = 1 << 10,
    TxSfdDone       = 1 << 11,
    RxSfdDone       = 1 << 12,
}

impl Event {
    pub(crate) fn mask() -> u16 {
        0x0000_1FFF
    }
}

impl BitAnd<Event> for u16 {
    type Output = u16;

    fn bitand(self, rhs: Event) -> Self::Output {
        self & rhs as u16
    }
}

impl BitOr for Event {
    type Output = u16;

    fn bitor(self, rhs: Self) -> Self::Output {
        self as u16 | rhs as u16
    }
}

impl BitOr<Event> for u16 {
    type Output = u16;

    fn bitor(self, rhs: Event) -> Self::Output {
        self | rhs as u16
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub(crate) enum TxAbortReason {
    RxAckStop       = 1,
    RxAckSfdTimeout = 2,
    RxAckCrcError   = 3,
    RxAckInvalidLen = 4,
    RxAckFilterFail = 5,
    RxAckNoRss      = 6,
    RxAckCoexBreak  = 7,
    RxAckTypeNotAck = 8,
    RxAckRestart    = 9,
    RxAckTimeout    = 16,
    TxStop          = 17,
    TxCoexBreak     = 18,
    TxSecurityError = 19,
    CcaFailed       = 24,
    CcaBusy         = 25,
}

impl TxAbortReason {
    pub fn bit(&self) -> u32 {
        1 << (*self as u32 - 1)
    }
}

impl BitOr for TxAbortReason {
    type Output = u32;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.bit() | rhs.bit()
    }
}

impl BitOr<TxAbortReason> for u32 {
    type Output = u32;

    fn bitor(self, rhs: TxAbortReason) -> Self::Output {
        self | rhs.bit()
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub(crate) enum RxAbortReason {
    RxStop              = 1,
    SfdTimeout          = 2,
    CrcError            = 3,
    InvalidLen          = 4,
    FilterFail          = 5,
    NoRss               = 6,
    CoexBreak           = 7,
    UnexpectedAck       = 8,
    RxRestart           = 9,
    TxAckTimeout        = 16,
    TxAckStop           = 17,
    TxAckCoexBreak      = 18,
    EnhackSecurityError = 19,
    EdAbort             = 24,
    EdStop              = 25,
    EdCoexReject        = 26,
}

impl RxAbortReason {
    pub fn bit(&self) -> u32 {
        1 << (*self as u32 - 1)
    }
}

impl BitOr for RxAbortReason {
    type Output = u32;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.bit() | rhs.bit()
    }
}

impl BitOr<RxAbortReason> for u32 {
    type Output = u32;

    fn bitor(self, rhs: RxAbortReason) -> Self::Output {
        self | rhs.bit()
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub(crate) enum EdSampleMode {
    Max = 0,
    Avg = 1,
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub(crate) enum Command {
    TxStart     = 0x41,
    RxStart     = 0x42,
    CcaTxStart  = 0x43,
    EdStart     = 0x44,
    Stop        = 0x45,
    DtmTxStart  = 0x46,
    DtmRxStart  = 0x47,
    DtmStop     = 0x48,
    Timer0Start = 0x4C,
    Timer0Stop  = 0x4D,
    Timer1Start = 0x4E,
    Timer1Stop  = 0x4F,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum MultipanIndex {
    Multipan0 = 0,
    Multipan1 = 1,
    Multipan2 = 2,
    Multipan3 = 3,
}

impl From<usize> for MultipanIndex {
    fn from(value: usize) -> Self {
        match value {
            0 => MultipanIndex::Multipan0,
            1 => MultipanIndex::Multipan1,
            2 => MultipanIndex::Multipan2,
            3 => MultipanIndex::Multipan3,
            _ => panic!(),
        }
    }
}

#[inline(always)]
pub(crate) fn mac_date() -> u32 {
    IEEE802154::regs().mac_date().read().bits()
}

#[inline(always)]
pub(crate) fn set_rx_on_delay(delay: u16) {
    IEEE802154::regs()
        .rxon_delay()
        .modify(|_, w| unsafe { w.rxon_delay().bits(delay) });
}

#[inline(always)]
pub(crate) fn enable_events(events: u16) {
    IEEE802154::regs()
        .event_en()
        .modify(|r, w| unsafe { w.event_en().bits(r.event_en().bits() | events) });
}

#[inline(always)]
pub(crate) fn disable_events(events: u16) {
    IEEE802154::regs()
        .event_en()
        .modify(|r, w| unsafe { w.event_en().bits(r.event_en().bits() & !events) });
}

#[inline(always)]
pub(crate) fn enable_tx_abort_events(events: u32) {
    IEEE802154::regs()
        .tx_abort_interrupt_control()
        .modify(|r, w| unsafe {
            w.tx_abort_interrupt_control()
                .bits(r.tx_abort_interrupt_control().bits() | events)
        });
}

#[inline(always)]
pub(crate) fn enable_rx_abort_events(events: u32) {
    IEEE802154::regs()
        .rx_abort_intr_ctrl()
        .modify(|r, w| unsafe {
            w.rx_abort_intr_ctrl()
                .bits(r.rx_abort_intr_ctrl().bits() | events)
        });
}

#[inline(always)]
pub(crate) fn set_ed_sample_mode(ed_sample_mode: EdSampleMode) {
    IEEE802154::regs()
        .ed_scan_cfg()
        .modify(|_, w| unsafe { w.ed_sample_mode().bits(ed_sample_mode as u8) });
}

#[inline(always)]
pub(crate) fn set_tx_addr(addr: *const u8) {
    IEEE802154::regs()
        .txdma_addr()
        .modify(|_, w| unsafe { w.txdma_addr().bits(addr as u32) });
}

#[inline(always)]
pub(crate) fn set_cmd(cmd: Command) {
    IEEE802154::regs()
        .command()
        .modify(|_, w| unsafe { w.opcode().bits(cmd as u8) });
}

#[inline(always)]
pub(crate) fn set_freq(freq: u8) {
    IEEE802154::regs()
        .channel()
        .modify(|_, w| unsafe { w.hop().bits(freq) });
}

#[inline(always)]
pub(crate) fn freq() -> u8 {
    IEEE802154::regs().channel().read().hop().bits()
}

#[inline(always)]
pub(crate) fn set_power(power: u8) {
    IEEE802154::regs()
        .tx_power()
        .modify(|_, w| unsafe { w.tx_power().bits(power) });
}

#[inline(always)]
pub(crate) fn set_multipan_enable_mask(mask: u8) {
    // apparently the REGS are garbage and the struct is right?
    IEEE802154::regs()
        .ctrl_cfg()
        .modify(|r, w| unsafe { w.bits(r.bits() & !(0b1111 << 29) | ((mask as u32) << 29)) });
}

#[inline(always)]
pub(crate) fn set_multipan_panid(index: MultipanIndex, panid: u16) {
    IEEE802154::regs()
        .inf(index as usize)
        .pan_id()
        .write(|w| unsafe { w.pan_id().bits(panid) });
}

#[inline(always)]
pub(crate) fn set_multipan_short_addr(index: MultipanIndex, value: u16) {
    IEEE802154::regs()
        .inf(index as usize)
        .short_addr()
        .write(|w| unsafe { w.short_addr().bits(value) });
}

#[inline(always)]
pub(crate) fn set_multipan_ext_addr(
    index: MultipanIndex,
    ext_addr: [u8; IEEE802154_FRAME_EXT_ADDR_SIZE],
) {
    let inf = IEEE802154::regs().inf(index as usize);

    let ext_addr0 = u32::from_le_bytes(unwrap!(ext_addr[0..3].try_into()));
    let ext_addr1 = u32::from_le_bytes(unwrap!(ext_addr[4..7].try_into()));

    inf.extend_addr0()
        .write(|w| unsafe { w.extend_addr0().bits(ext_addr0) });
    inf.extend_addr1()
        .write(|w| unsafe { w.extend_addr1().bits(ext_addr1) });
}

#[inline(always)]
pub(crate) fn set_cca_mode(cca_mode: CcaMode) {
    IEEE802154::regs()
        .ed_scan_cfg()
        .modify(|_, w| unsafe { w.cca_mode().bits(cca_mode as u8) });
}

#[inline(always)]
pub(crate) fn set_cca_threshold(cca_threshold: i8) {
    IEEE802154::regs()
        .ed_scan_cfg()
        .modify(|_, w| unsafe { w.cca_ed_threshold().bits(cca_threshold as u8) });
}

#[inline(always)]
pub(crate) fn set_tx_auto_ack(enable: bool) {
    IEEE802154::regs()
        .ctrl_cfg()
        .modify(|_, w| w.hw_auto_ack_tx_en().bit(enable));
}

#[inline(always)]
pub(crate) fn tx_auto_ack() -> bool {
    IEEE802154::regs()
        .ctrl_cfg()
        .read()
        .hw_auto_ack_tx_en()
        .bit_is_set()
}

#[inline(always)]
pub(crate) fn set_rx_auto_ack(enable: bool) {
    IEEE802154::regs()
        .ctrl_cfg()
        .modify(|_, w| w.hw_auto_ack_rx_en().bit(enable));
}

#[inline(always)]
pub(crate) fn set_tx_enhance_ack(enable: bool) {
    IEEE802154::regs()
        .ctrl_cfg()
        .modify(|_, w| w.hw_enhance_ack_tx_en().bit(enable));
}

#[inline(always)]
pub(crate) fn tx_enhance_ack() -> bool {
    IEEE802154::regs()
        .ctrl_cfg()
        .read()
        .hw_enhance_ack_tx_en()
        .bit_is_set()
}

#[inline(always)]
pub(crate) fn set_coordinator(enable: bool) {
    IEEE802154::regs()
        .ctrl_cfg()
        .modify(|_, w| w.pan_coordinator().bit(enable));
}

#[inline(always)]
pub(crate) fn set_promiscuous(enable: bool) {
    IEEE802154::regs()
        .ctrl_cfg()
        .modify(|_, w| w.promiscuous_mode().bit(enable));
}

#[inline(always)]
pub(crate) fn set_pending_mode(enable: bool) {
    IEEE802154::regs()
        .ctrl_cfg()
        .modify(|_, w| w.autopend_enhance().bit(enable));
}

#[inline(always)]
pub(crate) fn events() -> u16 {
    IEEE802154::regs().event_status().read().bits() as u16
}

#[inline(always)]
pub(crate) fn clear_events(events: u16) {
    IEEE802154::regs()
        .event_status()
        .modify(|r, w| unsafe { w.event_status().bits(r.event_status().bits() & events) });
}

#[inline(always)]
pub(crate) fn set_transmit_security(enable: bool) {
    IEEE802154::regs()
        .sec_ctrl()
        .modify(|_, w| w.sec_en().bit(enable));
}

#[inline(always)]
pub(crate) fn set_rx_addr(addr: *mut u8) {
    IEEE802154::regs()
        .rxdma_addr()
        .modify(|_, w| unsafe { w.rxdma_addr().bits(addr as u32) });
}

#[inline(always)]
pub(crate) fn abort_tx() {
    IEEE802154::regs()
        .tx_status()
        .modify(|_, w| unsafe { w.tx_abort_status().bits(0) });
}

#[inline(always)]
pub(crate) fn abort_rx() {
    IEEE802154::regs()
        .rx_status()
        .modify(|_, w| unsafe { w.rx_abort_status().bits(0) });
}
