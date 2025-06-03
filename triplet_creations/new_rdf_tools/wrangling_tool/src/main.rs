use std::fs::{File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use oxigraph::io::{RdfFormat, RdfParser, RdfSerializer};
use oxigraph::model::Triple;

#[derive(Parser,Debug)]
#[command(name = "TTL Splitter")]
#[command(about = "Will split the large turtle file dump so that our local server can import it with less failure risk.")]
struct Args{
    
    #[arg(short, long)]
    input: PathBuf,

    // #[arg(short, long)]
    // parts: usize,
    //
    // We let go of parts and instead we use number of linees or triplets
    #[arg(short, long, default_value = "1000000")]
    num_elements_per_part: usize,

    #[arg(short, long, default_value = "./ttl_chunks")]
    output_dir: PathBuf,

    #[arg(long, default_value = "prefixes.ttl")]
    out_prefix_filename: PathBuf,

    // We hardcode this because it is very expensive to calculate using `wc -l`
    // #[arg(long, default_value = "24672995429")]
    #[arg(long, default_value = "1000031")]
    num_lines_in_file: u64,
}


fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    /*
    * Data Preprocessing
    */
    //Set up the file loading
    let file = File::open(&args.input).expect(format!("Unable to open input file {}", args.input.to_str().unwrap()).as_str());
    // let metadata = file.metadata()?; TODO rm
    // let file_size = metadata.len(); TODO: rm
    let input_file_name = args.input.file_name().unwrap().to_str().unwrap();

    let input_file_noext: Vec<&str> = input_file_name.split(".").collect();
    let input_file_noext = input_file_noext[0];

    //Remove any basenames and leave only the directory
    let out_parent_dir = args.output_dir.as_path();
    if !out_parent_dir.exists() {
        println!("The output directory does not exist, creating it");
        std::fs::create_dir_all(out_parent_dir)?;
    } else {
        println!("The output directory {} already exists", out_parent_dir.to_str().unwrap());
    }

    println!("The input file name that we are dealing with is {input_file_name}");

    let progress = ProgressBar::new(args.num_lines_in_file);
    progress.set_style(ProgressStyle::with_template(
        "[{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})]"
    )?);

    // Create input file stream reader
    let input_file_reader = BufReader::new(file);

    /*
    * Iterative Parsing
    */

    //TODO: Replace this with triplet per element rather than line
    let mut num_lines: usize = 0;
    let mut cur_part = 0;
    let mut part_path_file = args.output_dir.join(format!("{input_file_noext}_part_{cur_part}.ttl"));
    println!("The output file name that we are dealing with is {}", part_path_file.to_str().unwrap());
    let mut buf_writer = BufWriter::new(File::create(&part_path_file)?);
    let parser = RdfParser::from_format(RdfFormat::Turtle).for_reader(input_file_reader);

    let num_elements_to_print = 35;
    for (idx, res_element) in parser.enumerate() {

        let rdf_element = res_element.expect("Error when trying to parse line element.");

        println!("{}", rdf_element);

        if idx > num_elements_to_print {break};
    }
    
    // for (_, line) in input_file_reader.lines().enumerate() {
    //     num_lines += 1;
    //
    //     buf_writer.write_all(line?.as_bytes())?;
    //
    //     if num_lines % args.num_elements_per_part == 0 {
    //         buf_writer.flush()?;
    //         cur_part += 1;
    //         part_path_file = args.output_dir.join(format!("{input_file_noext}_part_{cur_part}.ttl"));
    //         buf_writer = BufWriter::new(File::create(&part_path_file)?);
    //     }
    // }


    // let mut writers = Vec::with_capacity(args.parts);
    // for i in 0..args.parts {
    //     // First read the 
    //     let path = args.output_dir.join(format!("{input_file_noext}_part_{i}.ttl"));
    //     let writer = BufWriter::new(File::create(path)?);
    //     let sink = RdfSerializer::from_format(RdfFormat::Turtle).for_writer(writer);
    //     writers.push(sink);
    // }

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    }
}
