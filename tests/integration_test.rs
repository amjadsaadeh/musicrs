// Integration tests for musicrs

use std::fs;
use std::path::PathBuf;

// Some helpers
fn parse_test_file_meta_params(path: &PathBuf) -> Option<Vec<&str>> {
    path.file_name()
        .and_then(|name| name.to_str())
        .and_then(|file_name| Some(file_name.strip_suffix(".wav").unwrap().split('_').collect::<Vec<&str>>()))
}

#[test]
fn test_single_source_doa() {
    let paths = fs::read_dir("./tests/data/single-source").unwrap()
        .filter(|p| {
            let p = p.as_ref().unwrap().path();
            p.extension().and_then(|ext| ext.to_str()) == Some("wav")
        });
    for path in paths {
        let path = path.unwrap().path();
        let parts = parse_test_file_meta_params(&path).unwrap();
        let freq: f32 = parts[1].strip_suffix("Hz").unwrap().parse::<f32>().unwrap();
        let doa: f32 = parts[2].strip_suffix("deg").unwrap().parse::<f32>().unwrap();
        let duration: f32 = parts[3].strip_suffix("s").unwrap().parse::<f32>().unwrap();

        // TODO read war file
        println!("Testing file: {:?}", parts);
    }
}
