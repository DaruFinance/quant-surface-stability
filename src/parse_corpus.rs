//! Parse the strategy `.txt` files into a single parquet per asset.
//!
//! Walks `<root>/<family>/<strategy>/<strategy>.txt`, extracts every
//! `Wxx (IS|OOS)(+ENT|+FEE|+SLI|+ENT+IND)?` line, and writes one parquet row
//! per (strategy, window, sample, perturbation).
//!
//! Strategy-name parsing is done with permissive heuristics: any segment that
//! cannot be cleanly identified as `transformation` or `confluence` is kept
//! verbatim in the `tail` column, so no information is lost. Parsing is done
//! in parallel via rayon.

use arrow::array::{ArrayRef, Float32Array, Int32Array, Int8Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use clap::Parser as _;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use regex::Regex;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;
use walkdir::WalkDir;

#[derive(clap::Parser, Debug)]
#[command(name = "parse_corpus")]
struct Args {
    /// Asset directory under --root (e.g. SOL_1h_7W)
    #[arg(long)]
    asset: String,

    /// Strategies root holding one directory per asset
    #[arg(long)]
    root: PathBuf,

    /// Output directory
    #[arg(long, default_value = "data")]
    out: PathBuf,
}

#[derive(Debug, Clone)]
struct Row {
    family: String,
    strategy: String,
    base_param: String,
    transformation: String,
    confluence: String,
    sl_regime: String,
    tail: String,
    window: i8,
    sample: String,        // "IS" or "OOS"
    perturbation: String,  // "base", "ENT", "FEE", "SLI", "ENT+IND"
    trades: i32,
    roi: f32,
    pf: f32,
    sharpe: f32,
    win_rate: f32,
    expectancy: f32,
    max_dd: f32,
}

/// Regex for a per-window metric line. The `(?i)` is unnecessary but harmless;
/// case is consistent in the source files.
fn line_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        // Examples:
        //   W01 IS (LB 138) | Trades:  25  ROI:$537.32  PF:  2.01  Shp:  1.77  Win: 60.00%  Exp:$21.63  MaxDD:$213.57
        //   W01 OOS+ENT+IND (LB 139) | Trades:  12  ROI:$104.14  PF:  1.34  Shp:  0.50  Win: 50.00%  Exp:$8.97  MaxDD:$106.81
        Regex::new(concat!(
            r"^\s*W(?P<w>\d+)\s+",
            r"(?P<sample>IS|OOS)",
            r"(?:\+(?P<pert>[A-Za-z+]+))?",
            r"\s+\(LB\s+\d+\)\s+\|",
            r"\s*Trades:\s*(?P<trades>\d+)",
            r"\s+ROI:\$\s*(?P<roi>-?[\d\.]+)",
            r"\s+PF:\s*(?P<pf>-?[\d\.]+)",
            r"\s+Shp:\s*(?P<shp>-?[\d\.]+)",
            r"\s+Win:\s*(?P<win>-?[\d\.]+)%",
            r"\s+Exp:\$\s*(?P<exp>-?[\d\.]+)",
            r"\s+MaxDD:\$\s*(?P<dd>-?[\d\.]+)",
        ))
        .unwrap()
    })
}

/// Known transformation tokens (positions in the strategy name where a
/// transformation may live). Confluences are everything else; we are
/// permissive on purpose — the smoothness analysis uses these as categorical
/// neighbourhood axes, not as continuous parameters.
const TRANSFORMATIONS: &[&str] = &[
    "accel", "bias", "burstfreq", "disFromMedian", "fold_dev",
    "kurtosis", "kurtosis10", "normalized_price", "pi", "quant_stretch",
    "rank_resid", "roc", "skew", "skew0.75", "slope", "volZ",
    "vr", "zscore",
];

const CONFLUENCES: &[&str] = &[
    "BW_filter", "DeadFlat10", "EMAHug", "InsideBar", "NoNewLowGreen",
    "Pge0.7", "Pge0.8", "RSIge40", "RSIge50", "RangeSpike", "SameDirection",
    "TinyBody", "TopOfRange", "VolContraction", "YesterdayPeak",
    "atr_pct", "atr_pct0.8",
];

/// Parse a strategy directory name into its structured components. Best-effort:
/// anything we can't classify lands in `tail` so downstream code can still
/// group on it deterministically.
fn parse_name(family: &str, name: &str) -> (String, String, String, String, String) {
    // Strip SL regime suffix
    let (base, sl) = if let Some(stripped) = name.strip_suffix("_SL2") {
        (stripped.to_string(), "SL2".to_string())
    } else if let Some(stripped) = name.strip_suffix("_SL3") {
        (stripped.to_string(), "SL3".to_string())
    } else {
        (name.to_string(), "".to_string())
    };

    // Capture the base-parameter prefix (the family-specific numeric part).
    // EMA: "EMA_x_EMA<N>", SMA: "SMA_x_SMA<N>", RSI: "RSI_x_(EMA|SMA)<N>",
    // PPO: "PPO_x_(EMA|SMA)<N>", ATR: "ATR_x_(EMA|SMA)<N>",
    // STOCHK: "STOCHK_(EMA|SMA)_<N>", MACD: "MACD(<F>,<S>)",
    // RSI_LEVEL: "RSI_(EMA|SMA)_<N>"
    let base_param_re = match family {
        "MACD" => Regex::new(r"^MACD\(\d+,\d+\)").unwrap(),
        "STOCHK" => Regex::new(r"^STOCHK_(EMA|SMA)_\d+").unwrap(),
        "RSI_LEVEL" => Regex::new(r"^RSI_(EMA|SMA)_\d+").unwrap(),
        _ => Regex::new(&format!(r"^{}_x_(EMA|SMA)\d+", family)).unwrap(),
    };

    let (base_param, rest) = if let Some(m) = base_param_re.find(&base) {
        let bp = m.as_str().to_string();
        let after = base[m.end()..].trim_start_matches('_').to_string();
        (bp, after)
    } else {
        (base.clone(), String::new())
    };

    // Split remaining tokens, identify transformation and confluence.
    let tokens: Vec<&str> = rest.split('_').filter(|s| !s.is_empty()).collect();

    let mut transformation = String::new();
    let mut confluence = String::new();
    let mut tail_tokens: Vec<String> = Vec::new();

    // Greedy multi-token match for items containing underscores (e.g.
    // "BW_filter", "fold_dev", "atr_pct0.8").
    let mut i = 0;
    while i < tokens.len() {
        let mut matched = false;
        for span in (1..=4.min(tokens.len() - i)).rev() {
            let candidate = tokens[i..i + span].join("_");
            if TRANSFORMATIONS.iter().any(|t| *t == candidate) && transformation.is_empty() {
                transformation = candidate;
                i += span;
                matched = true;
                break;
            } else if CONFLUENCES.iter().any(|c| *c == candidate) && confluence.is_empty() {
                confluence = candidate;
                i += span;
                matched = true;
                break;
            }
        }
        if !matched {
            tail_tokens.push(tokens[i].to_string());
            i += 1;
        }
    }

    let tail = tail_tokens.join("_");
    if transformation.is_empty() {
        transformation = "raw".to_string();
    }
    if confluence.is_empty() {
        confluence = "raw".to_string();
    }
    let sl_out = if sl.is_empty() { "NA".to_string() } else { sl };

    (base_param, transformation, confluence, sl_out, tail)
}

fn parse_strategy_file(
    family: &str,
    strategy_name: &str,
    path: &Path,
    out: &mut Vec<Row>,
) -> std::io::Result<()> {
    let content = std::fs::read_to_string(path)?;
    let (base_param, transformation, confluence, sl_regime, tail) = parse_name(family, strategy_name);
    let re = line_re();

    for line in content.lines() {
        let Some(c) = re.captures(line) else { continue };
        let parse_f32 = |k: &str| -> f32 {
            c.name(k)
                .map(|m| m.as_str().parse::<f32>().unwrap_or(f32::NAN))
                .unwrap_or(f32::NAN)
        };
        let parse_i = |k: &str| -> i32 {
            c.name(k)
                .map(|m| m.as_str().parse::<i32>().unwrap_or(0))
                .unwrap_or(0)
        };
        let perturbation = c
            .name("pert")
            .map(|m| m.as_str().to_string())
            .unwrap_or_else(|| "base".to_string());
        out.push(Row {
            family: family.to_string(),
            strategy: strategy_name.to_string(),
            base_param: base_param.clone(),
            transformation: transformation.clone(),
            confluence: confluence.clone(),
            sl_regime: sl_regime.clone(),
            tail: tail.clone(),
            window: parse_i("w") as i8,
            sample: c["sample"].to_string(),
            perturbation,
            trades: parse_i("trades"),
            roi: parse_f32("roi"),
            pf: parse_f32("pf"),
            sharpe: parse_f32("shp"),
            win_rate: parse_f32("win"),
            expectancy: parse_f32("exp"),
            max_dd: parse_f32("dd"),
        });
    }
    Ok(())
}

fn rows_to_batch(rows: &[Row], asset: &str) -> RecordBatch {
    let n = rows.len();
    let asset_v = StringArray::from(vec![asset; n]);
    let family = StringArray::from(rows.iter().map(|r| r.family.as_str()).collect::<Vec<_>>());
    let strategy = StringArray::from(rows.iter().map(|r| r.strategy.as_str()).collect::<Vec<_>>());
    let base_param = StringArray::from(rows.iter().map(|r| r.base_param.as_str()).collect::<Vec<_>>());
    let transformation = StringArray::from(rows.iter().map(|r| r.transformation.as_str()).collect::<Vec<_>>());
    let confluence = StringArray::from(rows.iter().map(|r| r.confluence.as_str()).collect::<Vec<_>>());
    let sl_regime = StringArray::from(rows.iter().map(|r| r.sl_regime.as_str()).collect::<Vec<_>>());
    let tail = StringArray::from(rows.iter().map(|r| r.tail.as_str()).collect::<Vec<_>>());
    let window = Int8Array::from(rows.iter().map(|r| r.window).collect::<Vec<_>>());
    let sample = StringArray::from(rows.iter().map(|r| r.sample.as_str()).collect::<Vec<_>>());
    let perturbation = StringArray::from(rows.iter().map(|r| r.perturbation.as_str()).collect::<Vec<_>>());
    let trades = Int32Array::from(rows.iter().map(|r| r.trades).collect::<Vec<_>>());
    let roi = Float32Array::from(rows.iter().map(|r| r.roi).collect::<Vec<_>>());
    let pf = Float32Array::from(rows.iter().map(|r| r.pf).collect::<Vec<_>>());
    let sharpe = Float32Array::from(rows.iter().map(|r| r.sharpe).collect::<Vec<_>>());
    let win_rate = Float32Array::from(rows.iter().map(|r| r.win_rate).collect::<Vec<_>>());
    let expectancy = Float32Array::from(rows.iter().map(|r| r.expectancy).collect::<Vec<_>>());
    let max_dd = Float32Array::from(rows.iter().map(|r| r.max_dd).collect::<Vec<_>>());

    let schema = Arc::new(Schema::new(vec![
        Field::new("asset", DataType::Utf8, false),
        Field::new("family", DataType::Utf8, false),
        Field::new("strategy", DataType::Utf8, false),
        Field::new("base_param", DataType::Utf8, false),
        Field::new("transformation", DataType::Utf8, false),
        Field::new("confluence", DataType::Utf8, false),
        Field::new("sl_regime", DataType::Utf8, false),
        Field::new("tail", DataType::Utf8, false),
        Field::new("window", DataType::Int8, false),
        Field::new("sample", DataType::Utf8, false),
        Field::new("perturbation", DataType::Utf8, false),
        Field::new("trades", DataType::Int32, false),
        Field::new("roi", DataType::Float32, false),
        Field::new("pf", DataType::Float32, false),
        Field::new("sharpe", DataType::Float32, false),
        Field::new("win_rate", DataType::Float32, false),
        Field::new("expectancy", DataType::Float32, false),
        Field::new("max_dd", DataType::Float32, false),
    ]));

    let arrays: Vec<ArrayRef> = vec![
        Arc::new(asset_v),
        Arc::new(family),
        Arc::new(strategy),
        Arc::new(base_param),
        Arc::new(transformation),
        Arc::new(confluence),
        Arc::new(sl_regime),
        Arc::new(tail),
        Arc::new(window),
        Arc::new(sample),
        Arc::new(perturbation),
        Arc::new(trades),
        Arc::new(roi),
        Arc::new(pf),
        Arc::new(sharpe),
        Arc::new(win_rate),
        Arc::new(expectancy),
        Arc::new(max_dd),
    ];

    RecordBatch::try_new(schema, arrays).unwrap()
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let asset_root = args.root.join(&args.asset);
    if !asset_root.is_dir() {
        eprintln!("ERROR: {} is not a directory", asset_root.display());
        std::process::exit(2);
    }
    std::fs::create_dir_all(&args.out)?;

    eprintln!("Scanning {}...", asset_root.display());
    let mut strategy_files: Vec<(String, String, PathBuf)> = Vec::new();

    for fam_entry in std::fs::read_dir(&asset_root)? {
        let fam_entry = fam_entry?;
        if !fam_entry.file_type()?.is_dir() {
            continue;
        }
        let family = fam_entry.file_name().to_string_lossy().to_string();
        for s_entry in WalkDir::new(fam_entry.path())
            .min_depth(1)
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if !s_entry.file_type().is_dir() {
                continue;
            }
            let strategy_name = s_entry.file_name().to_string_lossy().to_string();
            let txt = s_entry.path().join(format!("{}.txt", strategy_name));
            if txt.is_file() {
                strategy_files.push((family.clone(), strategy_name, txt));
            }
        }
    }
    eprintln!("Found {} strategy .txt files", strategy_files.len());

    let all_rows: Vec<Vec<Row>> = strategy_files
        .par_iter()
        .map(|(family, name, path)| {
            let mut rows = Vec::with_capacity(7 * 5 * 2); // ~70 rows / strategy
            if let Err(e) = parse_strategy_file(family, name, path, &mut rows) {
                eprintln!("WARN: failed to parse {}: {}", path.display(), e);
            }
            rows
        })
        .collect();

    let mut flat: Vec<Row> = Vec::with_capacity(all_rows.iter().map(|r| r.len()).sum());
    for v in all_rows {
        flat.extend(v);
    }
    eprintln!("Parsed {} rows", flat.len());

    if flat.is_empty() {
        eprintln!("ERROR: no rows parsed");
        std::process::exit(3);
    }

    let batch = rows_to_batch(&flat, &args.asset);

    let out_path = args.out.join(format!("{}.parquet", args.asset));
    let file = File::create(&out_path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    eprintln!("Wrote {} ({} rows)", out_path.display(), flat.len());

    Ok(())
}
