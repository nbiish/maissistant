use std::sync::Mutex;
use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    Manager,
};
use tauri_plugin_shell::ShellExt;
use xcap::{Monitor, Window};
use std::io::Cursor;
use base64::{Engine as _, engine::general_purpose};
use image::ImageFormat;

struct AppState {
    agent_count: Mutex<usize>,
}

#[derive(serde::Serialize)]
struct CaptureSource {
    id: String,
    name: String,
    kind: String, // "monitor" or "window"
}

#[tauri::command]
fn list_sources() -> Vec<CaptureSource> {
    let mut sources = Vec::new();
    
    // Monitors
    if let Ok(monitors) = Monitor::all() {
        for monitor in monitors {
            // Monitor ID returns Result in xcap 0.8? Or just u32?
            // The error said `monitor.id()` returns Result<u32, XCapError>
            if let Ok(id) = monitor.id() {
                 let name = monitor.name().unwrap_or_else(|_| "Unknown Monitor".to_string());
                 sources.push(CaptureSource {
                    id: id.to_string(),
                    name,
                    kind: "monitor".to_string(),
                });
            }
        }
    }

    // Windows
    if let Ok(windows) = Window::all() {
        for window in windows {
            // Window getters also return Result
            if let (Ok(title), Ok(app_name), Ok(id)) = (window.title(), window.app_name(), window.id()) {
                if !title.trim().is_empty() {
                     sources.push(CaptureSource {
                        id: id.to_string(), 
                        name: format!("{} - {}", app_name, title),
                        kind: "window".to_string(),
                    });
                }
            }
        }
    }
    
    sources
}

#[tauri::command]
fn capture_source(id: String, kind: String) -> Result<String, String> {
    let image = if kind == "monitor" {
        let monitors = Monitor::all().map_err(|e| e.to_string())?;
        let monitor = monitors.into_iter().find(|m| {
            m.id().map(|i| i.to_string()).unwrap_or_default() == id
        }).ok_or("Monitor not found")?;
        monitor.capture_image().map_err(|e| e.to_string())?
    } else {
        let windows = Window::all().map_err(|e| e.to_string())?;
        let window = windows.into_iter().find(|w| {
             w.id().map(|i| i.to_string()).unwrap_or_default() == id
        }).ok_or("Window not found")?;
        window.capture_image().map_err(|e| e.to_string())?
    };

    let mut buffer = Cursor::new(Vec::new());
    // Convert to PNG
    image.write_to(&mut buffer, ImageFormat::Png).map_err(|e| e.to_string())?;
    
    let base64_str = general_purpose::STANDARD.encode(buffer.into_inner());
    Ok(base64_str)
}

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            app.manage(AppState {
                agent_count: Mutex::new(0),
            });

            let sidecar = app.shell().sidecar("python-backend").map_err(|e| e.to_string())?;
            let (mut rx, _child) = sidecar.spawn().map_err(|e| e.to_string())?;

            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                     println!("Sidecar event: {:?}", event);
                }
            });

            let quit_i = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let new_agent_i = MenuItem::with_id(app, "new_agent", "New Agent", true, None::<&str>)?;
            let settings_i = MenuItem::with_id(app, "settings", "Settings", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&new_agent_i, &settings_i, &quit_i])?;

            let mut builder = TrayIconBuilder::new()
                .menu(&menu)
                .show_menu_on_left_click(true)
                .on_menu_event(|app, event| {
                    match event.id.as_ref() {
                        "quit" => {
                            app.exit(0);
                        }
                        "settings" => {
                            let _ = tauri::WebviewWindowBuilder::new(
                                app,
                                "settings",
                                tauri::WebviewUrl::App("settings.html".into()),
                            )
                            .title("MAIssistant Settings")
                            .inner_size(600.0, 800.0)
                            .build();
                        }
                        "new_agent" => {
                            let state = app.state::<AppState>();
                            let mut count = state.agent_count.lock().unwrap();
                            *count += 1;
                            let agent_id = *count;
                            println!("Spawning agent #{}", agent_id);
                            
                            let label = format!("agent-{}", agent_id);
                            let _ = tauri::WebviewWindowBuilder::new(
                                app,
                                label,
                                tauri::WebviewUrl::App("index.html".into()),
                            )
                            .title(format!("Agent {}", agent_id))
                            .inner_size(800.0, 600.0)
                            .build();
                        }
                        _ => {}
                    }
                });

            if let Some(icon) = app.default_window_icon() {
                builder = builder.icon(icon.clone());
            }

            builder.build(app)?;

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![greet, list_sources, capture_source])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
