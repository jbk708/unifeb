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
    hnsw.parallel_insert(data_with_id);
    hnsw.dump_layer_info();

    // Convert HNSW to a KGraph
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
    // Build HNSW
    let mut hnsw = Hnsw::<f32, NewDistUniFrac>::new(
        hparams.max_conn,
        nb_data,
        nb_layer,
        hparams.ef_c,
        dist_unifrac,
    );
    hnsw.modify_level_scale(hparams.scale_modification);
    hnsw.parallel_insert(data_with_id);
    hnsw.dump_layer_info();

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

    Ok((sample_names, feature_names, matrix))
}

//
// main
//
fn main() -> Result<()> {
    // Initialize logger
    println!("\n ************** initializing logger *****************\n");
    env_logger::Builder::from_default_env().init();

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
    log::debug!("hnswparams: {:?}", hnswparams);

    // parse embedding params
    let (embedparams, maybe_quality) = parse_embed_group(&matches)?;

    // read newick tree from file
    let newick_filename = matches.get_one::<String>("tree").unwrap();
    let newick_str = std::fs::read_to_string(newick_filename)
        .map_err(|e| anyhow!("Could not read newick file {}: {}", newick_filename, e))?;

    // read feature table
    let feature_filename = matches.get_one::<String>("featuretable").unwrap();
    let (_sample_names, feature_names, matrix) = parse_feature_table(feature_filename)?;

    // build DistUniFrac
    let weighted = matches.get_flag("weighted");
    // feature_names is the "dimension list" in the same order we used in matrix columns
    let dist_unifrac = NewDistUniFrac::new(&newick_str, weighted, feature_names.clone())?;

    // We'll embed S samples => each row of `matrix` is a sample vector
    // data_with_id => ( &Vec<f32>, sample_id )
    let data_with_id: Vec<(&Vec<f32>, usize)> =
        matrix.iter().enumerate().map(|(i, v)| (v, i)).collect();

    // recommended #layers for HNSW
    let nb_data = data_with_id.len();
    let nb_layer = 16.min((nb_data as f32).ln().trunc() as usize);

    // Output
    let outfile = matches.get_one::<String>("outfile").unwrap();
    let mut csv_w = csv::Writer::from_path(outfile).unwrap();

    // If not hierarchical, we do the standard approach
    if embedparams.get_hierarchy_layer() == 0 {
        let kgraph = get_kgraph_unifrac(&data_with_id, dist_unifrac, &hnswparams, nb_layer);

        let mut embedder = Embedder::new(&kgraph, embedparams);
        let embed_res = embedder.embed();
        if embed_res.is_err() {
            log::error!("embedding failed");
            std::process::exit(1);
        }
        //
        // we can use get_embedded_reindexed as we indexed DataId contiguously in hnsw!
        let _res = write_csv_array2(&mut csv_w, &embedder.get_embedded_reindexed());
        csv_w.flush().unwrap();

        if let Some(q) = maybe_quality {
            log::info!(
                "quality sampling fraction requested: {:.4}",
                q.sampling_fraction
            );
            let _quality_est = embedder.get_quality_estimate_from_edge_length(100);
            // Possibly print or store _quality_est
        }
    } else {
        // hierarchical approach
        let layer_proj = embedparams.get_hierarchy_layer();
        let graphproj = get_kgraphproj_unifrac(
            &data_with_id,
            dist_unifrac,
            &hnswparams,
            nb_layer,
            layer_proj,
        );

        let mut embedder = Embedder::from_hkgraph(&graphproj, embedparams);
        let embed_res = embedder.embed();
        if embed_res.is_err() {
            log::error!("embedding failed");
            std::process::exit(1);
        }
        //
        // we can use get_embedded_reindexed as we indexed DataId contiguously in hnsw!
        let _res = write_csv_array2(&mut csv_w, &embedder.get_embedded_reindexed());
        csv_w.flush().unwrap();

        if let Some(q) = maybe_quality {
            log::info!(
                "quality sampling fraction requested (hierarchical): {:.4}",
                q.sampling_fraction
            );
            let _quality_est = embedder.get_quality_estimate_from_edge_length(100);
        }
    }

    println!(
        "Embedding done. Output written in {} ({} samples).",
        outfile, nb_data
    );
    Ok(())
}
