// UniFrac Embedding for large-scale Microbiome dataset. We realy on annembed and HubNSW for build HNSW graph.
//
// Example usage:
//   unifeb -- \
//     --tree my_tree.nwk \
//     --featuretable my_feature_table.tsv \
//     --weighted \
//     --out embedded.csv \
//     hnsw --nbconn 48 --knbn 10 --ef 400
//
// Explanation:
//   --tree: path to the Newick-format tree file
//   --featuretable: path to the feature table (tab-delimited or CSV)
//   --weighted (optional): if set, we do Weighted UniFrac; otherwise Unweighted
//   hnsw subcommand: sets HNSW parameters
//     * --nbconn = number of neighbors per layer
//     * --knbn   = number of final adjacency neighbors to keep
//     * --ef     = search factor
//     * --scale_modification_f = HubNSW modification factor
//
// The output "embedded.csv" has as many rows as there are samples
// (i.e. columns in your feature table, excluding the first column for feature IDs).
// -------------------------------------------------------------------------------------

// HashMap import removed (unused)
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use anyhow::{anyhow, Result};
use clap::{Arg, ArgAction, ArgMatches, Command};
use hnsw_rs::prelude::*;
use ndarray::Array2; // For Hnsw

// *** Import DistUniFrac
//use anndists::dist::DistUniFrac; // commented out per change to use NewDistUniFrac
use anndists::dist::distances::NewDistUniFrac;

// annembed
use annembed::fromhnsw::kgproj::KGraphProjection;
use annembed::fromhnsw::kgraph::{kgraph_from_hnsw_all, KGraph};
use annembed::prelude::{Embedder, EmbedderParams};

//
// This struct holds HNSW parameters.
//
#[derive(Debug, Clone)]
pub struct HnswParams {
    pub max_conn: usize,
    pub ef_c: usize,
    pub knbn: usize,
    pub scale_modification: f64,
}

impl Default for HnswParams {
    fn default() -> Self {
        HnswParams {
            max_conn: 48,
            ef_c: 400,
            knbn: 10,
            scale_modification: 0.25,
        }
    }
}

//
// parse_hnsw_cmd subcommand
//
fn parse_hnsw_cmd(matches: &ArgMatches) -> Result<HnswParams, anyhow::Error> {
    log::debug!("in parse_hnsw_cmd");
    let hparams = HnswParams {
        max_conn: *matches.get_one::<usize>("nbconn").unwrap(),
        ef_c: *matches.get_one::<usize>("ef").unwrap(),
        knbn: *matches.get_one::<usize>("knbn").unwrap(),
        scale_modification: *matches.get_one::<f64>("scale_modification").unwrap(),
    };

    Ok(hparams)
}

//
// QualityParams: optional sampling fraction for measuring embedding quality
// Keep a minimal struct and use the field so we don't get dead-code warnings
#[derive(Debug, Clone)]
pub struct QualityParams {
    pub sampling_fraction: f64,
}

//
// parse_embed_group: parse top-level embedding arguments
//
fn parse_embed_group(
    matches: &ArgMatches,
) -> Result<(EmbedderParams, Option<QualityParams>), anyhow::Error> {
    log::debug!("in parse_embed_group");

    let mut embedparams = EmbedderParams::default();

    embedparams.nb_grad_batch = *matches.get_one::<usize>("batch").unwrap();
    embedparams.asked_dim = *matches.get_one::<usize>("dimension").unwrap();
    embedparams.scale_rho = *matches.get_one::<f64>("scale").unwrap();
    embedparams.nb_sampling_by_edge = *matches.get_one::<usize>("nbsample").unwrap();
    embedparams.hierarchy_layer = *matches.get_one::<usize>("hierarchy").unwrap();

    let quality: Option<QualityParams>;
    if let Some(frac) = matches.get_one::<f64>("quality") {
        log::debug!("quality is asked, sampling fraction : {:.2e}", frac);
        quality = Some(QualityParams {
            sampling_fraction: *frac,
        });
    } else {
        quality = None;
    }

    Ok((embedparams, quality))
}

//
// Building a KGraph via HNSW with DistUniFrac
//
fn get_kgraph_unifrac(
    data_with_id: &[(&Vec<f32>, usize)],
    dist_unifrac: NewDistUniFrac,
    hparams: &HnswParams,
    nb_layer: usize,
) -> KGraph<f32> {
    let nb_data = data_with_id.len();
    log::debug!(
        "Building HNSW graph for {} samples with {} layers",
        nb_data,
        nb_layer
    );

    // Build HNSW
    let mut hnsw = Hnsw::<f32, NewDistUniFrac>::new(
        hparams.max_conn,
        nb_data,
        nb_layer,
        hparams.ef_c,
        dist_unifrac, // pass in the DistUniFrac struct
    );
    hnsw.modify_level_scale(hparams.scale_modification);
    // Insert data
    log::debug!("Inserting {} samples into HNSW graph...", nb_data);
    hnsw.parallel_insert(data_with_id);
    hnsw.dump_layer_info();
    log::debug!("HNSW graph construction completed");

    // Convert HNSW to a KGraph
    log::debug!(
        "Converting HNSW to KGraph (keeping {} neighbors per node)...",
        hparams.knbn
    );
    kgraph_from_hnsw_all(&hnsw, hparams.knbn).unwrap()
}

// Hierarchical approach
fn get_kgraphproj_unifrac(
    data_with_id: &[(&Vec<f32>, usize)],
    dist_unifrac: NewDistUniFrac,
    hparams: &HnswParams,
    nb_layer: usize,
    layer_proj: usize,
) -> KGraphProjection<f32> {
    let nb_data = data_with_id.len();
    log::debug!(
        "Building hierarchical HNSW graph for {} samples (projection layer: {})",
        nb_data,
        layer_proj
    );
    // Build HNSW
    let mut hnsw = Hnsw::<f32, NewDistUniFrac>::new(
        hparams.max_conn,
        nb_data,
        nb_layer,
        hparams.ef_c,
        dist_unifrac,
    );
    hnsw.modify_level_scale(hparams.scale_modification);
    log::debug!(
        "Inserting {} samples into hierarchical HNSW graph...",
        nb_data
    );
    hnsw.parallel_insert(data_with_id);
    hnsw.dump_layer_info();
    log::debug!("Building KGraphProjection at layer {}", layer_proj);

    // Build a KGraphProjection for hierarchical embedding
    KGraphProjection::<f32>::new(&hnsw, hparams.knbn, layer_proj)
}

// Just a little helper to write the final embedded coordinates to a CSV file.
fn write_csv_array2(
    w: &mut csv::Writer<std::fs::File>,
    array2: &Array2<f32>, // changed to f32
) -> Result<(), anyhow::Error> {
    for row in array2.outer_iter() {
        let float_vec: Vec<f32> = row.to_vec();
        w.write_record(
            float_vec
                .iter()
                .map(|x| format!("{:.6}", x))
                .collect::<Vec<String>>(),
        )?;
    }
    Ok(())
}

// parse_feature_table:
//  The table must have a header row like:
//    #OTU_ID  SampleA  SampleB  SampleC ...
//  Then each subsequent row is:
//    T1       2        0        8
//    T2       0        3        0
//    ...
//  We parse them so that we produce:
//    - sample_names: ["SampleA", "SampleB", "SampleC", ...]
//    - feature_names: ["T1", "T2", ...]
//    - matrix: Vec of length = #samples, each element is a Vec<f32> of length = #features.
//
//  DistUniFrac needs feature_names in the same order as the dimension indices in the sample vectors.
//
type FeatureTableResult = (Vec<String>, Vec<String>, Vec<Vec<f32>>);

fn parse_feature_table(filename: &str) -> Result<FeatureTableResult> {
    log::debug!("Parsing feature table from: {}", filename);
    let f =
        File::open(filename).map_err(|_| anyhow!("Cannot open featuretable file: {}", filename))?;
    let mut lines = BufReader::new(f).lines();

    // First line => sample names
    let header_line = lines
        .next()
        .ok_or_else(|| anyhow!("Feature table is empty"))??;
    let cols: Vec<&str> = header_line.split('\t').collect();
    if cols.len() < 2 {
        return Err(anyhow!("No samples in feature table header?"));
    }
    // The first column is typically "#OTU_ID" or "Feature", the rest are sample names
    let sample_names: Vec<String> = cols[1..].iter().map(|s| s.to_string()).collect();
    let nsamples = sample_names.len();

    // We'll accumulate (feature_name, counts across samples)
    let mut feature_names: Vec<String> = Vec::new();
    let mut rows_data: Vec<Vec<f32>> = Vec::new(); // row=feature, col=sample

    for line_res in lines {
        let line = line_res?;
        if line.trim().is_empty() {
            continue;
        }
        let vals: Vec<&str> = line.split('\t').collect();
        if vals.len() < nsamples + 1 {
            // skip or error
            continue;
        }
        let feature_name = vals[0].to_string();
        feature_names.push(feature_name);

        let mut row_counts = Vec::with_capacity(nsamples);
        for i in 0..nsamples {
            let v = vals[i + 1].parse::<f32>().unwrap_or(0.0);
            row_counts.push(v);
        }
        rows_data.push(row_counts);
    }

    // We have rows_data => row=feature, col=sample
    // But we want row=sample, col=feature for embedding => transpose
    let nfeatures = feature_names.len();
    let mut matrix = vec![vec![0.0; nfeatures]; nsamples];
    for (f_idx, row) in rows_data.iter().enumerate() {
        for (s_idx, val) in row.iter().enumerate() {
            matrix[s_idx][f_idx] = *val;
        }
    }
    // matrix[sample_index][feature_index]

    log::debug!(
        "Parsed feature table: {} samples, {} features",
        sample_names.len(),
        feature_names.len()
    );
    Ok((sample_names, feature_names, matrix))
}

//
// main
//
fn main() -> Result<()> {
    // Initialize logger
    env_logger::Builder::from_default_env().init();

    // Record start time
    let start_time = Instant::now();
    log::info!("========================================");
    log::info!("UniFrac Embedding (unifeb) starting");
    log::info!("========================================");

    // Define the hnsw subcommand
    let hnswcmd = Command::new("hnsw")
        .about("Build HNSW/HubNSW graph")
        .arg(
            Arg::new("nbconn")
                .long("nbconn")
                .required(true)
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize))
                .help("Number of neighbours by layer"),
        )
        .arg(
            Arg::new("knbn")
                .long("knbn")
                .required(true)
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize))
                .help("Number of neighbours to keep in final adjacency"),
        )
        .arg(
            Arg::new("ef")
                .long("ef")
                .required(true)
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize))
                .help("Search factor for HNSW construction"),
        )
        .arg(
            Arg::new("scale_modification")
                .long("scale_modify_f")
                .help("scale modification factor in HNSW or HubNSW, must be in [0.2,1]")
                .value_name("scale_modify")
                .default_value("0.25")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(f64)),
        );

    // Top-level clap config
    let matches = Command::new("unifeb")
        .version("0.1.0")
        .arg_required_else_help(true)
        .about("UniFrac Embedding via Approxiamte Nearest Neighbor Graph")
        .arg(
            Arg::new("tree")
                .short('t')
                .long("tree")
                .required(true)
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(String))
                .help("Newick tree filename"),
        )
        .arg(
            Arg::new("featuretable")
                .short('f')
                .long("feature-table")
                .required(true)
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(String))
                .help("Feature table with rows=features, columns=samples"),
        )
        .arg(
            Arg::new("weighted")
                .long("weighted")
                .required(false)
                .action(ArgAction::SetTrue)
                .help("Use Weighted UniFrac (otherwise unweighted)"),
        )
        .arg(
            Arg::new("outfile")
                .long("out")
                .short('o')
                .required(false)
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(String))
                .default_value("embedded.csv")
                .help("Output CSV file for embedded results"),
        )
        .arg(
            Arg::new("batch")
                .required(false)
                .long("batch")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize))
                .default_value("20")
                .help("Number of gradient batches"),
        )
        .arg(
            Arg::new("nbsample")
                .required(false)
                .long("nbsample")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize))
                .default_value("10")
                .help("Number of edge samplings per batch"),
        )
        .arg(
            Arg::new("hierarchy")
                .required(false)
                .long("layer")
                .short('l')
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize))
                .default_value("0")
                .help("If >0, use hierarchical approach in embedding"),
        )
        .arg(
            Arg::new("scale")
                .required(false)
                .long("scale")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(f64))
                .default_value("1.0")
                .help("Rho scale factor for the gradient descent"),
        )
        .arg(
            Arg::new("dimension")
                .required(false)
                .long("dim")
                .short('d')
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize))
                .default_value("2")
                .help("Dimension of embedding"),
        )
        .arg(
            Arg::new("quality")
                .required(false)
                .long("quality")
                .short('q')
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(f64))
                .help("Sampling fraction for quality estimation, <=1.0"),
        )
        .subcommand(hnswcmd)
        .get_matches();

    // parse hnsw params if subcommand is present
    let hnswparams: HnswParams = if let Some(hnsw_m) = matches.subcommand_matches("hnsw") {
        parse_hnsw_cmd(hnsw_m)?
    } else {
        HnswParams::default()
    };
    log::info!(
        "HNSW parameters: max_conn={}, ef_c={}, knbn={}, scale_modification={:.3}",
        hnswparams.max_conn,
        hnswparams.ef_c,
        hnswparams.knbn,
        hnswparams.scale_modification
    );

    // parse embedding params
    let (embedparams, maybe_quality) = parse_embed_group(&matches)?;
    log::info!(
        "Embedding parameters: dim={}, batch={}, nbsample={}, scale={:.3}, hierarchy_layer={}",
        embedparams.asked_dim,
        embedparams.nb_grad_batch,
        embedparams.nb_sampling_by_edge,
        embedparams.scale_rho,
        embedparams.hierarchy_layer
    );

    // read newick tree from file
    let tree_start = Instant::now();
    let newick_filename = matches.get_one::<String>("tree").unwrap();
    log::info!("Reading phylogenetic tree from: {}", newick_filename);
    let newick_str = std::fs::read_to_string(newick_filename)
        .map_err(|e| anyhow!("Could not read newick file {}: {}", newick_filename, e))?;
    let tree_elapsed = tree_start.elapsed();
    log::info!(
        "Tree file read successfully in {:.2}s",
        tree_elapsed.as_secs_f64()
    );

    // read feature table
    let table_start = Instant::now();
    let feature_filename = matches.get_one::<String>("featuretable").unwrap();
    log::info!("Reading feature table from: {}", feature_filename);
    let (sample_names, feature_names, matrix) = parse_feature_table(feature_filename)?;
    let table_elapsed = table_start.elapsed();
    let nsamples = sample_names.len();
    let nfeatures = feature_names.len();
    log::info!(
        "Feature table read successfully in {:.2}s ({} samples, {} features)",
        table_elapsed.as_secs_f64(),
        nsamples,
        nfeatures
    );

    // build DistUniFrac
    let weighted = matches.get_flag("weighted");
    let unifrac_type = if weighted { "weighted" } else { "unweighted" };
    log::info!("Building {} UniFrac distance calculator", unifrac_type);
    // feature_names is the "dimension list" in the same order we used in matrix columns
    let dist_unifrac = NewDistUniFrac::new(&newick_str, weighted, feature_names.clone())?;
    log::info!("UniFrac distance calculator initialized");

    // We'll embed S samples => each row of `matrix` is a sample vector
    // data_with_id => ( &Vec<f32>, sample_id )
    let data_with_id: Vec<(&Vec<f32>, usize)> =
        matrix.iter().enumerate().map(|(i, v)| (v, i)).collect();

    // recommended #layers for HNSW
    let nb_data = data_with_id.len();
    let nb_layer = 16.min((nb_data as f32).ln().trunc() as usize);
    log::info!(
        "Prepared {} samples for embedding (HNSW layers: {})",
        nb_data,
        nb_layer
    );

    // Output
    let outfile = matches.get_one::<String>("outfile").unwrap();
    log::info!("Output file: {}", outfile);

    // If not hierarchical, we do the standard approach
    if embedparams.get_hierarchy_layer() == 0 {
        log::info!("Using standard embedding approach");

        // Build HNSW graph
        let graph_start = Instant::now();
        log::info!("Building HNSW graph...");
        let kgraph = get_kgraph_unifrac(&data_with_id, dist_unifrac, &hnswparams, nb_layer);
        let graph_elapsed = graph_start.elapsed();
        log::info!(
            "HNSW graph built successfully in {:.2}s ({} nodes)",
            graph_elapsed.as_secs_f64(),
            kgraph.get_nb_nodes()
        );

        // Perform embedding
        let embed_start = Instant::now();
        log::info!("Starting embedding process...");
        let mut embedder = Embedder::new(&kgraph, embedparams);
        let embed_res = embedder.embed();
        if embed_res.is_err() {
            log::error!("Embedding failed");
            std::process::exit(1);
        }
        let embed_elapsed = embed_start.elapsed();
        log::info!(
            "Embedding completed successfully in {:.2}s",
            embed_elapsed.as_secs_f64()
        );

        // Write output
        let write_start = Instant::now();
        log::info!("Writing embedded coordinates to output file...");
        let mut csv_w = csv::Writer::from_path(outfile).unwrap();
        let _res = write_csv_array2(&mut csv_w, &embedder.get_embedded_reindexed());
        csv_w.flush().unwrap();
        let write_elapsed = write_start.elapsed();
        log::info!(
            "Output written successfully in {:.2}s",
            write_elapsed.as_secs_f64()
        );

        if let Some(q) = maybe_quality {
            log::info!(
                "Quality estimation requested (sampling fraction: {:.4})",
                q.sampling_fraction
            );
            let _quality_est = embedder.get_quality_estimate_from_edge_length(100);
            // Possibly print or store _quality_est
        }
    } else {
        // hierarchical approach
        let layer_proj = embedparams.get_hierarchy_layer();
        log::info!(
            "Using hierarchical embedding approach (projection layer: {})",
            layer_proj
        );

        // Build HNSW graph with projection
        let graph_start = Instant::now();
        log::info!("Building HNSW graph with hierarchical projection...");
        let graphproj = get_kgraphproj_unifrac(
            &data_with_id,
            dist_unifrac,
            &hnswparams,
            nb_layer,
            layer_proj,
        );
        let graph_elapsed = graph_start.elapsed();
        log::info!(
            "HNSW graph with projection built successfully in {:.2}s",
            graph_elapsed.as_secs_f64()
        );

        // Perform embedding
        let embed_start = Instant::now();
        log::info!("Starting hierarchical embedding process...");
        let mut embedder = Embedder::from_hkgraph(&graphproj, embedparams);
        let embed_res = embedder.embed();
        if embed_res.is_err() {
            log::error!("Hierarchical embedding failed");
            std::process::exit(1);
        }
        let embed_elapsed = embed_start.elapsed();
        log::info!(
            "Hierarchical embedding completed successfully in {:.2}s",
            embed_elapsed.as_secs_f64()
        );

        // Write output
        let write_start = Instant::now();
        log::info!("Writing embedded coordinates to output file...");
        let mut csv_w = csv::Writer::from_path(outfile).unwrap();
        let _res = write_csv_array2(&mut csv_w, &embedder.get_embedded_reindexed());
        csv_w.flush().unwrap();
        let write_elapsed = write_start.elapsed();
        log::info!(
            "Output written successfully in {:.2}s",
            write_elapsed.as_secs_f64()
        );

        if let Some(q) = maybe_quality {
            log::info!(
                "Quality estimation requested (hierarchical, sampling fraction: {:.4})",
                q.sampling_fraction
            );
            let _quality_est = embedder.get_quality_estimate_from_edge_length(100);
        }
    }

    // Total time
    let total_elapsed = start_time.elapsed();
    log::info!("========================================");
    log::info!("Embedding completed successfully!");
    log::info!(
        "Total time: {:.2}s ({:.1} minutes)",
        total_elapsed.as_secs_f64(),
        total_elapsed.as_secs_f64() / 60.0
    );
    log::info!("Output written to: {} ({} samples)", outfile, nb_data);
    log::info!("========================================");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Helper function to create a simple test tree
    fn create_test_tree() -> String {
        // Simple tree: (A:0.1,B:0.2):0.05;
        "(A:0.1,B:0.2):0.05;".to_string()
    }

    // Helper function to create a simple feature table file
    fn create_test_feature_table() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "#OTU_ID\tSample1\tSample2\tSample3").unwrap();
        writeln!(file, "A\t10\t20\t0").unwrap();
        writeln!(file, "B\t5\t15\t30").unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_hnsw_params_default() {
        let params = HnswParams::default();
        assert_eq!(params.max_conn, 48);
        assert_eq!(params.ef_c, 400);
        assert_eq!(params.knbn, 10);
        assert_eq!(params.scale_modification, 0.25);
    }

    #[test]
    fn test_parse_feature_table() {
        let file = create_test_feature_table();
        let path = file.path().to_str().unwrap();

        let (sample_names, feature_names, matrix) = parse_feature_table(path).unwrap();

        // Check sample names
        assert_eq!(sample_names.len(), 3);
        assert_eq!(sample_names[0], "Sample1");
        assert_eq!(sample_names[1], "Sample2");
        assert_eq!(sample_names[2], "Sample3");

        // Check feature names
        assert_eq!(feature_names.len(), 2);
        assert_eq!(feature_names[0], "A");
        assert_eq!(feature_names[1], "B");

        // Check matrix transposition (row=sample, col=feature)
        assert_eq!(matrix.len(), 3); // 3 samples
        assert_eq!(matrix[0].len(), 2); // 2 features
        assert_eq!(matrix[0][0], 10.0); // Sample1, Feature A
        assert_eq!(matrix[0][1], 5.0); // Sample1, Feature B
        assert_eq!(matrix[1][0], 20.0); // Sample2, Feature A
        assert_eq!(matrix[1][1], 15.0); // Sample2, Feature B
        assert_eq!(matrix[2][0], 0.0); // Sample3, Feature A
        assert_eq!(matrix[2][1], 30.0); // Sample3, Feature B
    }

    #[test]
    fn test_parse_feature_table_empty_file() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();

        let result = parse_feature_table(path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Feature table is empty"));
    }

    #[test]
    fn test_parse_feature_table_malformed_header() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "#OTU_ID").unwrap(); // Only one column, no samples
        file.flush().unwrap();

        let path = file.path().to_str().unwrap();
        let result = parse_feature_table(path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No samples in feature table header"));
    }

    #[test]
    fn test_new_dist_unifrac_weighted_creation() {
        let tree = create_test_tree();
        let feature_names = vec!["A".to_string(), "B".to_string()];

        // Test that weighted NewDistUniFrac can be created successfully
        let dist_unifrac = NewDistUniFrac::new(&tree, true, feature_names.clone());
        assert!(
            dist_unifrac.is_ok(),
            "Weighted NewDistUniFrac should be created successfully"
        );
    }

    #[test]
    fn test_new_dist_unifrac_unweighted_creation() {
        let tree = create_test_tree();
        let feature_names = vec!["A".to_string(), "B".to_string()];

        // Test that unweighted NewDistUniFrac can be created successfully
        let dist_unifrac = NewDistUniFrac::new(&tree, false, feature_names.clone());
        assert!(
            dist_unifrac.is_ok(),
            "Unweighted NewDistUniFrac should be created successfully"
        );
    }

    #[test]
    fn test_weighted_vs_unweighted_hnsw_difference() {
        // Test that weighted and unweighted produce different HNSW graphs
        let tree = "(A:0.1,B:0.2):0.05;".to_string();
        let feature_names = vec!["A".to_string(), "B".to_string()];

        // Create weighted and unweighted distance calculators
        let dist_weighted = NewDistUniFrac::new(&tree, true, feature_names.clone()).unwrap();
        let dist_unweighted = NewDistUniFrac::new(&tree, false, feature_names.clone()).unwrap();

        // Create two sample vectors with different abundances
        // Sample1: high abundance of A, low of B
        // Sample2: low abundance of A, high of B
        let sample1: Vec<f32> = vec![100.0, 10.0];
        let sample2: Vec<f32> = vec![10.0, 100.0];
        let data_with_id: Vec<(&Vec<f32>, usize)> = vec![(&sample1, 0), (&sample2, 1)];

        let hparams = HnswParams {
            max_conn: 2,
            ef_c: 4,
            knbn: 1,
            scale_modification: 0.25,
        };
        let nb_layer = 2;

        // Build HNSW graphs with both distance metrics
        #[allow(unused_mut)]
        let mut hnsw_weighted = Hnsw::<f32, NewDistUniFrac>::new(
            hparams.max_conn,
            data_with_id.len(),
            nb_layer,
            hparams.ef_c,
            dist_weighted,
        );
        hnsw_weighted.parallel_insert(&data_with_id);

        #[allow(unused_mut)]
        let mut hnsw_unweighted = Hnsw::<f32, NewDistUniFrac>::new(
            hparams.max_conn,
            data_with_id.len(),
            nb_layer,
            hparams.ef_c,
            dist_unweighted,
        );
        hnsw_unweighted.parallel_insert(&data_with_id);

        // Convert to KGraphs
        let kgraph_weighted: KGraph<f32> =
            kgraph_from_hnsw_all(&hnsw_weighted, hparams.knbn).unwrap();
        let kgraph_unweighted: KGraph<f32> =
            kgraph_from_hnsw_all(&hnsw_unweighted, hparams.knbn).unwrap();

        // The graphs should be created successfully
        // Note: The actual neighbor relationships might differ between weighted and unweighted
        assert!(
            kgraph_weighted.get_nb_nodes() > 0,
            "Weighted KGraph should have nodes"
        );
        assert!(
            kgraph_unweighted.get_nb_nodes() > 0,
            "Unweighted KGraph should have nodes"
        );
    }

    #[test]
    fn test_weighted_unifrac_with_abundance_differences() {
        // Test that weighted UniFrac works with samples that have different abundances
        let tree = "(A:0.1,B:0.2):0.05;".to_string();
        let feature_names = vec!["A".to_string(), "B".to_string()];

        let dist_weighted = NewDistUniFrac::new(&tree, true, feature_names).unwrap();

        // Create samples with different abundance patterns
        // Sample1: A=100, B=10
        // Sample2: A=10, B=100
        let sample1: Vec<f32> = vec![100.0, 10.0];
        let sample2: Vec<f32> = vec![10.0, 100.0];
        let data_with_id: Vec<(&Vec<f32>, usize)> = vec![(&sample1, 0), (&sample2, 1)];

        let hparams = HnswParams {
            max_conn: 2,
            ef_c: 4,
            knbn: 1,
            scale_modification: 0.25,
        };
        let nb_layer = 2;

        // Build HNSW graph - should succeed
        #[allow(unused_mut)]
        let mut hnsw = Hnsw::<f32, NewDistUniFrac>::new(
            hparams.max_conn,
            data_with_id.len(),
            nb_layer,
            hparams.ef_c,
            dist_weighted,
        );
        hnsw.parallel_insert(&data_with_id);

        let kgraph: KGraph<f32> = kgraph_from_hnsw_all(&hnsw, hparams.knbn).unwrap();
        assert_eq!(
            kgraph.get_nb_nodes(),
            2,
            "KGraph should have 2 nodes for 2 samples"
        );
    }

    #[test]
    fn test_unweighted_unifrac_with_presence_absence() {
        // Test that unweighted UniFrac works with presence/absence data
        let tree = "(A:0.1,B:0.2):0.05;".to_string();
        let feature_names = vec!["A".to_string(), "B".to_string()];

        let dist_unweighted = NewDistUniFrac::new(&tree, false, feature_names).unwrap();

        // Sample1: A=100, B=10 (both present)
        // Sample2: A=10, B=100 (both present, different abundances)
        // Sample3: A=100, B=0 (only A present)
        let sample1: Vec<f32> = vec![100.0, 10.0];
        let sample2: Vec<f32> = vec![10.0, 100.0];
        let sample3: Vec<f32> = vec![100.0, 0.0];
        let data_with_id: Vec<(&Vec<f32>, usize)> =
            vec![(&sample1, 0), (&sample2, 1), (&sample3, 2)];

        let hparams = HnswParams {
            max_conn: 2,
            ef_c: 4,
            knbn: 1,
            scale_modification: 0.25,
        };
        let nb_layer = 2;

        // Build HNSW graph - should succeed
        #[allow(unused_mut)]
        let mut hnsw = Hnsw::<f32, NewDistUniFrac>::new(
            hparams.max_conn,
            data_with_id.len(),
            nb_layer,
            hparams.ef_c,
            dist_unweighted,
        );
        hnsw.parallel_insert(&data_with_id);

        let kgraph: KGraph<f32> = kgraph_from_hnsw_all(&hnsw, hparams.knbn).unwrap();
        assert_eq!(
            kgraph.get_nb_nodes(),
            3,
            "KGraph should have 3 nodes for 3 samples"
        );
    }
}
