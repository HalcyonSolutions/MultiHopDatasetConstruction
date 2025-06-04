use core::panic;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
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
    // #[arg(short, long, default_value = "10000")]
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

fn prepending_prefixes_to_files(
    chunks_dir: &PathBuf,
    prefixes: &HashSet<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Prefixes:");
    for prefix in prefixes {
        println!("\t- {}", prefix);
    }

    let read_dir_itr = std::fs::read_dir(chunks_dir).unwrap();
    let num_files = read_dir_itr.count();
    let progress = ProgressBar::new(num_files as u64);
    let progress_style = ProgressStyle::with_template(
        "[{elapsed_precise}] {msg} [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
    )?;
    progress.set_style(progress_style);

    // Form the prelude
    let prefix_prelude = prefixes
        .iter()
        .map(|prefix| format!("{prefix}\n"))
        .collect::<String>();

    // Get List of resulting chunks
    let read_dir_itr = std::fs::read_dir(chunks_dir).unwrap();
    for entry in read_dir_itr {
        progress.inc(1);
        let file_path = entry?.path();
        let file_path_str = file_path.to_str().unwrap();
        progress.set_message(format!("Modifying file {}", file_path_str));
        // We will now prepend the prefixes to each file
        let mut file = File::open(&file_path)?;
        let mut file_content = String::new();
        file.read_to_string(&mut file_content).unwrap();
        let mut file = File::create(&file_path)?;
        file_content = format!("{prefix_prelude}\n{file_content}");
        file.write_all(file_content.as_bytes()).unwrap();
    }
    Ok(())
}

fn split_file(
    file_path: &PathBuf,
    output_dir: &PathBuf,
    num_lines_per_output_file: u64,
    total_num_lines: u64,
) -> Result<HashSet<String>, Box<dyn std::error::Error>> {
    /*
     * Data Preprocessing
     */
    let file_path_str = file_path.to_str().unwrap();
    let file = File::open(file_path)
        .expect(format!("Unable to open input file {}", &file_path_str).as_str());
    let input_file_name = file_path.file_name().unwrap().to_str().unwrap();

    let input_file_noext: Vec<&str> = input_file_name.split(".").collect();
    let input_file_noext = input_file_noext[0];

    // Create input file stream reader
    let input_file_reader = BufReader::new(file);

    //Remove any basenames and leave only the directory
    let out_parent_dir = output_dir.as_path();
    if !out_parent_dir.exists() {
        println!("The output directory does not exist, creating it");
        std::fs::create_dir_all(out_parent_dir)?;
    }

    //TODO: Replace this with triplet per element rather than line
    let mut cur_part_num_lines: usize = 0;
    let mut cur_part = 0;
    let mut part_path_file = output_dir.join(format!(
        "{input_file_noext}_part_{:0width$}.ttl",
        cur_part,
        width = 5
    ));
    let mut prefixes: HashSet<String> = HashSet::new();
    let mut cur_buf_writer = BufWriter::new(File::create(&part_path_file)?);
    let progress = ProgressBar::new(total_num_lines);
    progress.set_style(ProgressStyle::with_template(
        "[{elapsed_precise}] {msg} [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
    )?);

    let mut filte_line_itr = input_file_reader.lines().enumerate();
    while let Some((_, line)) = filte_line_itr.next() {
        progress.inc(1); // Update progress bar for each line processed
        progress.set_message(format!("Writing to {}", part_path_file.to_str().unwrap()));
        cur_part_num_lines += 1;
        // Check if line contains `@prefix`
        let raw_line = line?;

        if raw_line.contains("@prefix") {
            prefixes.insert(raw_line);
            continue;
        }

        // Write to buffer/file
        cur_buf_writer.write_all(raw_line.as_bytes())?;
        cur_buf_writer.write_all(b"\n")?; // Add newline after each line

        // flush only if the last character is a dot or EOF
        let can_flush = raw_line.ends_with('.');

        // Check if we are past the number of lines per part
        if cur_part_num_lines >= (num_lines_per_output_file as usize) && can_flush {
            // Write the current part to the file
            cur_buf_writer
                .flush()
                .expect("Encountered problem when trying to flush file.");
            progress.set_message(format!("Flusing part {cur_part}"));

            cur_part += 1;
            part_path_file = output_dir.join(format!(
                "{input_file_noext}_part_{:0width$}.ttl",
                cur_part,
                width = 5
            ));
            cur_buf_writer = BufWriter::new(File::create(&part_path_file).unwrap_or_else(|_| {
                panic!("Unable to create file {}", part_path_file.to_str().unwrap());
            }));
            cur_part_num_lines = 0;
        }
    }
    cur_buf_writer.flush()?;
    Ok(prefixes)
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    /*
     * Iterative Parsing
     */
    let prefixes = split_file(
        &args.input,
        &args.output_dir,
        args.num_lines_per_output_file as u64,
        args.num_lines_in_file
    )?;

    /*
     * Preprending prefixes to all files
     */
    prepending_prefixes_to_files(&args.output_dir, &prefixes)?;

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    }
}
