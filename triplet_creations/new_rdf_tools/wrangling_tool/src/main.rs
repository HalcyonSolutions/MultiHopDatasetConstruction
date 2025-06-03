use core::panic;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Parser, Debug)]
#[command(name = "TTL Splitter")]
#[command(
    about = "Will split the large turtle file dump so that our local server can import it with less failure risk."
)]
struct Args {
    #[arg(short, long)]
    input: PathBuf,

    // We let go of parts and instead we use number of linees or triplets
    #[arg(short, long, default_value = "2500000")]
    num_lines_per_output_file: usize,

    #[arg(short, long, default_value = "./ttl_chunks")]
    output_dir: PathBuf,

    #[arg(long, default_value = "prefixes.ttl")]
    out_prefix_filename: PathBuf,

    // We hardcode this because it is very expensive to calculate using `wc -l`
    #[arg(long, default_value = "24672995429")]
    // #[arg(long, default_value = "1000031")]
    num_lines_in_file: u64,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    /*
     * Data Preprocessing
     */
    //Set up the file loading
    let file = File::open(&args.input)
        .expect(format!("Unable to open input file {}", args.input.to_str().unwrap()).as_str());
    let input_file_name = args.input.file_name().unwrap().to_str().unwrap();

    let input_file_noext: Vec<&str> = input_file_name.split(".").collect();
    let input_file_noext = input_file_noext[0];

    // Create input file stream reader
    let input_file_reader = BufReader::new(file);

    //Remove any basenames and leave only the directory
    let out_parent_dir = args.output_dir.as_path();
    if !out_parent_dir.exists() {
        println!("The output directory does not exist, creating it");
        std::fs::create_dir_all(out_parent_dir)?;
    }

    //TODO: Replace this with triplet per element rather than line
    let mut cur_part_num_lines: usize = 0;
    let mut cur_part = 0;
    let mut part_path_file = args.output_dir.join(format!(
        "{input_file_noext}_part_{:0width$}.ttl",
        cur_part,
        width = 5
    ));
    let mut prefixes: Vec<String> = Vec::new();
    let mut prefixes_done = false;

    /*
     * Iterative Parsing
     */

    let mut cur_buf_writer = BufWriter::new(File::create(&part_path_file)?);
    let progress = ProgressBar::new(args.num_lines_in_file);
    progress.set_style(ProgressStyle::with_template(
        "[{elapsed_precise}] {msg} [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
    )?);

    for (line_no, line) in input_file_reader.lines().enumerate() {
        progress.inc(1); // Update progress bar for each line processed
        progress.set_message(format!("Writing to {}", part_path_file.to_str().unwrap()));
        cur_part_num_lines += 1;
        // Check if line contains `@prefix`
        let raw_line = line.unwrap();

        if raw_line.contains("@prefix") {
            // if prefixes_done {
            //     panic!(
            //         "Error on line {} There should be no more prefix to be read after the prefixes have been read.",
            //         line_no
            //     );
            // }
            prefixes.push(raw_line.clone());
            cur_buf_writer.write_all(raw_line.as_bytes())?;
            cur_buf_writer.write_all(b"\n")?;
            continue;
        } else {
            prefixes_done = true;
        }

        // Write to buffer/file
        cur_buf_writer.write_all(raw_line.as_bytes())?;
        cur_buf_writer.write_all(b"\n")?; // Add newline after each line

        // flush only if the last character is a dot
        let can_flush = raw_line.ends_with('.');

        // Check if we are past the number of lines per part
        if cur_part_num_lines >= args.num_lines_per_output_file && can_flush {
            // Write the current part to the file
            cur_buf_writer
                .flush()
                .expect("Encountered problem when trying to flush file.");
            progress.set_message(format!("Flusing part {cur_part}"));

            cur_part += 1;
            part_path_file = args.output_dir.join(format!(
                "{input_file_noext}_part_{:0width$}.ttl",
                cur_part,
                width = 5
            ));
            cur_buf_writer = BufWriter::new(File::create(&part_path_file).unwrap_or_else(|_| {
                panic!("Unable to create file {}", part_path_file.to_str().unwrap());
            }));
            cur_part_num_lines = 0;

            // Write the prefixes to the file
            for prefix in prefixes.iter() {
                cur_buf_writer.write_all(prefix.as_bytes())?;
                cur_buf_writer.write_all(b"\n")?;
            }
        }
    }

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    }
}
