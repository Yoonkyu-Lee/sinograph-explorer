//! Capture-overlay window management.
//!
//! The overlay is a separate, transparent, always-on-top window used for
//! on-screen hanzi recognition (doc/39). It is created on demand, hidden from
//! screen capture, and closed when a candidate is picked.

use tauri::{Emitter, Manager};

/// The monitor whose physical bounds contain the point (x, y).
fn monitor_at(app: &tauri::AppHandle, x: f64, y: f64) -> Option<tauri::Monitor> {
    let (px, py) = (x as i32, y as i32);
    app.available_monitors().ok()?.into_iter().find(|m| {
        let p = m.position();
        let s = m.size();
        px >= p.x
            && px < p.x + s.width as i32
            && py >= p.y
            && py < p.y + s.height as i32
    })
}

/// Exclude a window from screen capture (`SetWindowDisplayAffinity` /
/// `WDA_EXCLUDEFROMCAPTURE`). The overlay stays visible to the user but is
/// invisible to `xcap`, so the live capture loop never grabs its own
/// rectangle. doc/39 M7.
#[cfg(windows)]
fn exclude_from_capture(win: &tauri::WebviewWindow) -> Result<(), String> {
    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::WindowsAndMessaging::{
        SetWindowDisplayAffinity, WDA_EXCLUDEFROMCAPTURE,
    };
    let hwnd = HWND(win.hwnd().map_err(|e| e.to_string())?.0 as _);
    unsafe { SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE) }
        .map_err(|e| e.to_string())
}

/// Build and show the capture overlay. Shared by the toolbar command and the
/// global hotkey. `async` so it runs off the main thread — building a
/// WebView2 window from a synchronous (main-thread) command deadlocks.
pub async fn open_overlay(app: tauri::AppHandle) -> Result<(), String> {
    use tauri::{WebviewUrl, WebviewWindowBuilder};

    eprintln!("[overlay] open requested");
    if let Some(w) = app.get_webview_window("capture-overlay") {
        eprintln!("[overlay] already open — focusing");
        w.set_focus().map_err(|e| e.to_string())?;
        return Ok(());
    }

    let cursor = app.cursor_position().map_err(|e| e.to_string())?;
    let monitor = monitor_at(&app, cursor.x, cursor.y)
        .or(app.primary_monitor().map_err(|e| e.to_string())?)
        .ok_or("연결된 모니터를 찾을 수 없습니다.")?;
    let pos = *monitor.position();
    let size = *monitor.size();
    eprintln!(
        "[overlay] cursor=({:.0},{:.0}) monitor pos=({},{}) size=({}x{})",
        cursor.x, cursor.y, pos.x, pos.y, size.width, size.height
    );

    let win = WebviewWindowBuilder::new(
        &app,
        "capture-overlay",
        WebviewUrl::App("overlay.html".into()),
    )
    .title("한자 인식")
    .inner_size(size.width as f64, size.height as f64)
    .position(pos.x as f64, pos.y as f64)
    .transparent(true)
    .decorations(false)
    .always_on_top(true)
    .skip_taskbar(true)
    .resizable(false)
    .shadow(false)
    .visible(false)
    .build()
    .map_err(|e| {
        eprintln!("[overlay] build FAILED: {e}");
        format!("overlay build: {e}")
    })?;
    eprintln!("[overlay] window built");

    // hide the overlay from screen capture before it is ever shown
    #[cfg(windows)]
    if let Err(e) = exclude_from_capture(&win) {
        eprintln!("[overlay] exclude_from_capture failed: {e}");
    }

    win.set_position(tauri::PhysicalPosition::new(pos.x, pos.y))
        .map_err(|e| e.to_string())?;
    win.set_size(tauri::PhysicalSize::new(size.width, size.height))
        .map_err(|e| e.to_string())?;
    win.show().map_err(|e| e.to_string())?;
    win.set_focus().map_err(|e| e.to_string())?;
    eprintln!("[overlay] shown + focused");
    Ok(())
}

/// Command (toolbar / IPC entry point): open the capture overlay.
#[tauri::command]
pub async fn open_capture_overlay(app: tauri::AppHandle) -> Result<(), String> {
    open_overlay(app).await
}

/// Command: close the capture overlay, if it is open.
#[tauri::command]
pub fn close_capture_overlay(app: tauri::AppHandle) -> Result<(), String> {
    if let Some(w) = app.get_webview_window("capture-overlay") {
        w.close().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Command: a candidate was picked in the overlay — close the overlay, bring
/// the main window forward, and hand it the codepoint via a `lookup-request`
/// event for the dictionary to open.
#[tauri::command]
pub fn focus_main_with_lookup(
    app: tauri::AppHandle,
    codepoint: String,
) -> Result<(), String> {
    if let Some(overlay) = app.get_webview_window("capture-overlay") {
        let _ = overlay.close();
    }
    if let Some(main) = app.get_webview_window("main") {
        let _ = main.unminimize();
        main.show().map_err(|e| e.to_string())?;
        main.set_focus().map_err(|e| e.to_string())?;
        main.emit("lookup-request", codepoint)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}
