pub fn generate_visualization(path: &str) {
    #[cfg(feature = "enableVisualization")]
    {
        use tracing::{info, warn};
        info!("Generating computation graph visualization...");
        match crate::tensor::VisualizationGraph::render_to_svg(path) {
            Ok(_) => info!("SVG graph saved to '{}'", path),
            Err(e) => warn!("Failed to save SVG graph: {:?}", e),
        }
    }
    #[cfg(not(feature = "enableVisualization"))]
    {
        let _ = path;
    }
}
