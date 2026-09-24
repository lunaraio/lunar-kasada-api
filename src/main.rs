mod payload;
mod routes;
mod utils;
mod worktime;

use std::io;
use std::sync::Arc;

use actix_web::{App, HttpResponse, HttpServer, web};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

async fn health() -> HttpResponse {
    HttpResponse::Ok().finish()
}

#[tokio::main]
async fn main() -> io::Result<()> {
    if let Err(e) = dotenvy::dotenv() {
        if !e.not_found() {
            return Err(io::Error::other(e));
        }
    }

    let port: u16 = std::env::var("PORT")
        .map_err(|e| io::Error::other(format!("PORT: {e}")))?
        .parse()
        .map_err(|e| io::Error::other(format!("PORT: {e}")))?;
    let client = utils::client::build_client(None, false).map_err(io::Error::other)?;
    let cache = worktime::cache::Cache::new();
    let profiles = Arc::new(utils::profiles::load().await.map_err(io::Error::other)?);
    HttpServer::new(move || {
        App::new()
            .service(
                web::resource("/payload")
                    .app_data(web::ThinData(Arc::clone(&profiles)))
                    .route(web::post().to(routes::payload::payload)),
            )
            .service(
                web::resource("/worktime")
                    .app_data(web::ThinData(client.clone()))
                    .app_data(web::ThinData(cache.clone()))
                    .route(web::post().to(routes::worktime::worktime)),
            )
            .service(
                web::resource("/test")
                    .app_data(web::ThinData(client.clone()))
                    .app_data(web::ThinData(Arc::clone(&profiles)))
                    .route(web::post().to(routes::test::test)),
            )
            .route("/health", web::head().to(health))
    })
    .bind(("0.0.0.0", port))?
    .run()
    .await
}
