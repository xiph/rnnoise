//! Given two files containing little-endian `i16`s, computes the correlation of the signals.

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        eprintln!("Usage: corr FILE1 FILE2");
        std::process::exit(1);
    }

    let f1 = &args[1];
    let f2 = &args[2];
    let data1 = std::fs::read(f1).unwrap_or_else(|e| {
        eprintln!("Failed to open \"{}\": {}", f1, e);
        std::process::exit(1);
    });
    let data2 = std::fs::read(f2).unwrap_or_else(|e| {
        eprintln!("Failed to open \"{}\": {}", f1, e);
        std::process::exit(1);
    });
    if data1.len() != data2.len() {
        eprintln!("File sizes differ");
        std::process::exit(1);
    }
    if data1.len() % 2 != 0 {
        eprintln!("File sizes are odd");
        std::process::exit(1);
    }

    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in data1.chunks(2) {
        x.push(i16::from_le_bytes([i[0], i[1]]) as f64);
    }
    for i in data2.chunks(2) {
        y.push(i16::from_le_bytes([i[0], i[1]]) as f64);
    }

    let xx: f64 = x.iter().map(|&n| n * n).sum();
    let yy: f64 = y.iter().map(|&n| n * n).sum();
    let xy: f64 = x.iter().zip(y.iter()).map(|(&n, &m)| n * m).sum();
    let corr = xy / (xx.sqrt() * yy.sqrt());
    println!("{}", corr);

    if (corr - 1.0).abs() > 1e-6 {
        eprintln!("Bad correlation");
        std::process::exit(1);
    }
}
