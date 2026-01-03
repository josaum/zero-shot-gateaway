use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Tabs},
    Frame, Terminal,
};
use std::io;
use std::sync::Arc;
use tokio::sync::mpsc;
use duckdb::arrow::record_batch::RecordBatch;
use duckdb::arrow::array::Array;


use crate::{AppState, LearnedSchema};

#[derive(Debug, Clone)]
pub struct _EventLog {
    pub event_type: String,
    pub payload: String,
    pub received_at: String,
}

#[derive(Debug, Clone)]
pub struct Metrics {
    pub events_ingested: usize,
    pub schemas_learned: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum Tab {
    Events,
    Schemas,
    Chat,
    Metrics,
}

#[derive(Debug, Clone, PartialEq)]
enum InputMode {
    Normal,
    Editing,
}

impl Tab {
    fn titles() -> Vec<&'static str> {
        vec!["Events", "Schemas", "Chat", "Metrics"]
    }

    fn from_index(index: usize) -> Self {
        match index {
            0 => Tab::Events,
            1 => Tab::Schemas,
            2 => Tab::Chat,
            _ => Tab::Metrics,
        }
    }

    fn to_index(&self) -> usize {
        match self {
            Tab::Events => 0,
            Tab::Schemas => 1,
            Tab::Chat => 2,
            Tab::Metrics => 3,
        }
    }
}

pub struct App {
    events: Vec<RecordBatch>,
    schemas: Vec<LearnedSchema>,
    metrics: Metrics,
    chat_input: String,
    chat_history: Vec<(String, String)>,
    active_tab: Tab,
    input_mode: InputMode,
    scroll_offset: usize,
    should_quit: bool,
}

impl App {
    fn new() -> Self {
        Self {
            events: Vec::new(),
            schemas: Vec::new(),
            metrics: Metrics {
                events_ingested: 0,
                schemas_learned: 0,
            },
            chat_input: String::new(),
            chat_history: Vec::new(),
            active_tab: Tab::Events,
            input_mode: InputMode::Normal,
            scroll_offset: 0,
            should_quit: false,
        }
    }

    fn next_tab(&mut self) {
        let current = self.active_tab.to_index();
        let next = (current + 1) % 4;
        self.active_tab = Tab::from_index(next);
        self.scroll_offset = 0;
    }

    fn previous_tab(&mut self) {
        let current = self.active_tab.to_index();
        let next = if current == 0 { 3 } else { current - 1 };
        self.active_tab = Tab::from_index(next);
        self.scroll_offset = 0;
    }

    fn scroll_down(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_add(1);
    }

    fn scroll_up(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(1);
    }
}

pub async fn run_tui(state: Arc<AppState>) -> io::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let mut app = App::new();

    // Create channel for data updates
    let (tx, mut rx) = mpsc::channel::<AppUpdate>(100);

    // Spawn background task to poll data
    // Spawn background task to listen for updates (Real-time Broadcast)
    let state_clone = state.clone();
    tokio::spawn(async move {
        listen_for_updates(state_clone, tx).await;
    });

    // Main event loop
    loop {
        terminal.draw(|f| ui(f, &app))?;

        // Handle updates from background task
        while let Ok(update) = rx.try_recv() {
            match update {
                AppUpdate::Events(events) => app.events = events,
                AppUpdate::Schemas(schemas) => app.schemas = schemas,
                AppUpdate::Metrics(metrics) => app.metrics = metrics,
            }
        }

        // Handle keyboard input
        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match app.input_mode {
                        InputMode::Normal => match key.code {
                            KeyCode::Char('e') if app.active_tab == Tab::Chat => {
                                app.input_mode = InputMode::Editing;
                            }
                            KeyCode::Char('q') => app.should_quit = true,
                            KeyCode::Char('1') => app.active_tab = Tab::Events,
                            KeyCode::Char('2') => app.active_tab = Tab::Schemas,
                            KeyCode::Char('3') => app.active_tab = Tab::Chat,
                            KeyCode::Char('4') => app.active_tab = Tab::Metrics,
                            KeyCode::Tab => app.next_tab(),
                            KeyCode::BackTab => app.previous_tab(),
                            KeyCode::Char('j') | KeyCode::Down => app.scroll_down(),
                            KeyCode::Char('k') | KeyCode::Up => app.scroll_up(),
                            _ => {}
                        },
                        InputMode::Editing => match key.code {
                            KeyCode::Enter => {
                                let msg = app.chat_input.drain(..).collect::<String>();
                                if !msg.is_empty() {
                                    app.chat_history.push(("User".into(), msg.clone()));
                                    
                                    // Submit to API
                                    let client = reqwest::Client::new();
                                    let res = client.post("http://127.0.0.1:9382/api/chat")
                                        .json(&serde_json::json!({ "msg": msg }))
                                        .send()
                                        .await;
                                        
                                    if let Ok(res) = res {
                                        if let Ok(json) = res.json::<serde_json::Value>().await {
                                            if let Some(content) = json["content"].as_str() {
                                                app.chat_history.push(("AI".into(), content.into()));
                                            }
                                        }
                                    }
                                }
                            },
                            KeyCode::Char(c) => {
                                app.chat_input.push(c);
                            },
                            KeyCode::Backspace => {
                                app.chat_input.pop();
                            },
                            KeyCode::Esc => {
                                app.input_mode = InputMode::Normal;
                            },
                            _ => {}
                        }
                    }
                }
            }
        }

        if app.should_quit {
            break;
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(())
}

enum AppUpdate {
    Events(Vec<RecordBatch>),
    Schemas(Vec<LearnedSchema>),
    Metrics(Metrics),
}

async fn listen_for_updates(state: Arc<AppState>, tx: mpsc::Sender<AppUpdate>) {
    let mut rx_notify = state.tx_notify.subscribe();
    
    // Initial fetch
    fetch_and_send(&state, &tx).await;

    loop {
        match rx_notify.recv().await {
            Ok(_) => {
                fetch_and_send(&state, &tx).await;
            }
            Err(_e) => {
                // If lagged, just fetch anyway
                fetch_and_send(&state, &tx).await;
            }
        }
    }
}

async fn fetch_and_send(state: &Arc<AppState>, tx: &mpsc::Sender<AppUpdate>) {
    // Fetch events from DuckDB (Arrow Zero-Copy)
    let events = {
        let db = state.db.lock();
        let stmt = db.prepare("SELECT type, payload::VARCHAR, timestamp FROM events ORDER BY timestamp DESC LIMIT 100").ok();
        if let Some(mut stmt) = stmt {
            // Use arrow interface directly
            stmt.query_arrow([])
                .map(|arrow| arrow.collect::<Vec<RecordBatch>>())
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    };
    let _ = tx.send(AppUpdate::Events(events)).await;

    // Fetch schemas
    let schemas = {
        let schemas_lock = state.schemas.read();
        schemas_lock.values().cloned().collect()
    };
    let _ = tx.send(AppUpdate::Schemas(schemas)).await;

    // Fetch metrics
    let metrics = {
        let m = state.metrics.lock();
        Metrics {
            events_ingested: m.events_ingested,
            schemas_learned: m.schemas_learned,
        }
    };
    let _ = tx.send(AppUpdate::Metrics(metrics)).await;
}

fn ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(0),     // Content
            Constraint::Length(3),  // Footer
        ])
        .split(f.area());

    // Header with tabs
    let titles: Vec<Line> = Tab::titles()
        .iter()
        .map(|t| Line::from(*t))
        .collect();
    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title(format!(
            "⚡ One-File Gateway │ Events: {} │ Schemas: {}",
            app.metrics.events_ingested, app.metrics.schemas_learned
        )))
        .select(app.active_tab.to_index())
        .style(Style::default().fg(Color::Cyan))
        .highlight_style(
            Style::default()
                .add_modifier(Modifier::BOLD)
                .bg(Color::DarkGray),
        );
    f.render_widget(tabs, chunks[0]);

    // Content based on active tab
    match app.active_tab {
        Tab::Events => render_events(f, chunks[1], app),
        Tab::Schemas => render_schemas(f, chunks[1], app),
        Tab::Chat => render_chat(f, chunks[1], app),
        Tab::Metrics => render_metrics(f, chunks[1], app),
    }

    // Footer with keybindings
    let footer = Paragraph::new("[1-4] Tabs │ [Tab] Next │ [j/k] Scroll │ [q] Quit")
        .style(Style::default().fg(Color::DarkGray))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(footer, chunks[2]);
}

fn render_events(f: &mut Frame, area: Rect, app: &App) {
    if app.events.is_empty() {
        let empty = Paragraph::new("No events yet. Send a POST to /ingest to get started.")
            .block(Block::default().borders(Borders::ALL).title("Live Events (Arrow IPC)"))
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(empty, area);
        return;
    }

    let mut items: Vec<ListItem> = Vec::new();
    let mut total_count = 0;

    // Iterate over RecordBatches
    for batch in &app.events {
        total_count += batch.num_rows();
        
        // Zero-copy access to columns
        let types = batch.column(0).as_any().downcast_ref::<duckdb::arrow::array::StringArray>().unwrap();
        // DuckDB JSON type might come back as string or struct. Assuming string/varchar for now. 
        // Note: 'payload JSON' in DuckDB is often stored as String in Arrow unless cast
        let payloads = batch.column(1).as_any().downcast_ref::<duckdb::arrow::array::StringArray>().unwrap();
        // Timestamp might need casting or specialized handling. 
        // For simplicity/robustness in TUI, we might treat received_at as string in query or cast here.
        // Let's assume we update the query to cast to string for easiest TUI rendering, 
        // or handle TimestampMicrosecondArray.
        // For this iteration, let's assume implementation casts in SQL or we handle string.
        // Based on schema `received_at TIMESTAMP`, DuckDB -> Arrow usually maps to Timestamp(Microsecond, None).
        // To make it easy for TUI, let's cast in SQL.
        
        let timestamps = batch.column(2).as_any().downcast_ref::<duckdb::arrow::array::StringArray>();
        
        // Safety: If timestamp casting failed or types mismatch, skip or show error
        if timestamps.is_none() {
            continue; 
        }
        let timestamps = timestamps.unwrap();

        for i in 0..batch.num_rows() {
            // Respect scroll
            if items.len() < 20 { // just filling buffer
                 let event_type = types.value(i);
                 let payload = payloads.value(i);
                 let received_at = timestamps.value(i);

                 // Pretty print logic (reused)
                 let payload_display = if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(payload) {
                    if let Ok(pretty) = serde_json::to_string_pretty(&parsed) {
                        if pretty.len() > 200 { format!("{}...", &pretty[..200]) } else { pretty }
                    } else { payload.to_string() }
                 } else { payload.to_string() };

                 let content = vec![
                    Line::from(vec![
                        Span::styled("●", Style::default().fg(Color::Green)),
                        Span::raw(" "),
                        Span::styled(received_at.get(11..19).unwrap_or(received_at), Style::default().fg(Color::DarkGray)),
                        Span::raw(" │ "),
                        Span::styled(event_type, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                    ]),
                    Line::from(Span::styled(
                        format!("  {}", payload_display.lines().next().unwrap_or("")),
                        Style::default().fg(Color::Cyan)
                    )),
                    Line::from(""),
                ];
                items.push(ListItem::new(content));
            }
        }
    }

    // Handle scroll offset manually since we are constructing items on fly from batches
    // For TUI simplicity with Arrow, we usually just take top N. 
    // Implementing full scrolling through multiple RecordBatches is complex.
    // We will simplify: always show top N from the query (which is LIMIT 50 anyway).
    
    let list = List::new(items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(format!("Live Events (Arrow Zero-Copy | Total: {})", total_count))
        )
        .style(Style::default().fg(Color::White));
    f.render_widget(list, area);
}

fn render_schemas(f: &mut Frame, area: Rect, app: &App) {
    if app.schemas.is_empty() {
        let empty = Paragraph::new("No schemas learned yet. Ingest events to build the schema registry.")
            .block(Block::default().borders(Borders::ALL).title("Learned Schemas"))
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(empty, area);
        return;
    }

    let items: Vec<ListItem> = app
        .schemas
        .iter()
        .skip(app.scroll_offset)
        .enumerate()
        .map(|(_idx, schema)| {
            let mut lines = vec![
                Line::from(vec![
                    Span::styled("├─", Style::default().fg(Color::DarkGray)),
                    Span::raw(" "),
                    Span::styled(&schema.name, Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
                    Span::raw(" "),
                    Span::styled(
                        format!("({})", schema.fields.len()),
                        Style::default().fg(Color::DarkGray)
                    ),
                ]),
            ];

            // Display fields as badges
            let field_badges: Vec<Span> = schema.fields.iter()
                .take(8) // Limit to 8 fields for display
                .enumerate()
                .flat_map(|(i, field)| {
                    let mut spans = vec![
                        Span::raw("  "),
                        Span::styled(
                            format!("[{}]", field),
                            Style::default().fg(Color::Cyan)
                        ),
                    ];
                    if i < schema.fields.len().saturating_sub(1).min(7) {
                        spans.push(Span::raw(" "));
                    }
                    spans
                })
                .collect();
            
            if !field_badges.is_empty() {
                lines.push(Line::from(field_badges));
            }

            if schema.fields.len() > 8 {
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(
                        format!("... and {} more", schema.fields.len() - 8),
                        Style::default().fg(Color::DarkGray)
                    ),
                ]));
            }

            lines.push(Line::from(""));
            ListItem::new(lines)
        })
        .collect();

    let list = List::new(items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(format!("Learned Schemas ({})", app.schemas.len()))
        )
        .style(Style::default().fg(Color::White));
    f.render_widget(list, area);
}

fn render_chat(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),    // Messages
            Constraint::Length(3), // Input
        ])
        .split(area);

    // Messages Area
    let messages: Vec<ListItem> = app
        .chat_history
        .iter()
        .map(|(role, content)| {
            let role_style = if role == "User" {
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
            };
            
            let mut lines = vec![
                Line::from(vec![
                    Span::styled(format!("{}: ", role), role_style),
                ])
            ];
            
            // Basic wrapping (split by newlines for now)
            for line in content.lines() {
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(line, Style::default().fg(Color::White)),
                ]));
            }
            lines.push(Line::from(""));
            
            ListItem::new(lines)
        })
        .collect();

    let messages_list = List::new(messages)
        .block(Block::default().borders(Borders::ALL).title("Conversation"))
        .style(Style::default().fg(Color::White));
    f.render_widget(messages_list, chunks[0]);

    // Input Area
    let input_style = match app.input_mode {
        InputMode::Normal => Style::default().fg(Color::Gray),
        InputMode::Editing => Style::default().fg(Color::Yellow),
    };
    
    let input_title = match app.input_mode {
        InputMode::Normal => "Input (Press 'e' to edit, 'q' to quit)",
        InputMode::Editing => "Input (Press 'Esc' to stop editing, 'Enter' to send)",
    };

    let input = Paragraph::new(app.chat_input.as_str())
        .style(input_style)
        .block(Block::default().borders(Borders::ALL).title(input_title));
    f.render_widget(input, chunks[1]);
}

fn render_metrics(f: &mut Frame, area: Rect, app: &App) {
    let events_bar = create_simple_bar(app.metrics.events_ingested, 100);
    let schemas_bar = create_simple_bar(app.metrics.schemas_learned, 20);
    
    let text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("Events Ingested:  ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", app.metrics.events_ingested),
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
            ),
        ]),
        Line::from(Span::styled(events_bar, Style::default().fg(Color::Green))),
        Line::from(""),
        Line::from(vec![
            Span::styled("Schemas Learned:  ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", app.metrics.schemas_learned),
                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)
            ),
        ]),
        Line::from(Span::styled(schemas_bar, Style::default().fg(Color::Magenta))),
        Line::from(""),
        Line::from(""),
        Line::from(Span::styled("Real-time updates every 2 seconds", Style::default().fg(Color::DarkGray))),
    ];
    
    let paragraph = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title("Metrics Dashboard"))
        .style(Style::default());
    f.render_widget(paragraph, area);
}

fn create_simple_bar(value: usize, max: usize) -> String {
    let width: usize = 40;
    let filled = ((value as f64 / max as f64) * width as f64).min(width as f64) as usize;
    let empty = width.saturating_sub(filled);
    format!("  [{}{}]", "█".repeat(filled), "░".repeat(empty))
}
