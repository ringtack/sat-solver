use std::{path::Path, time::Instant};

use clap::Parser;
use dimacs::parser::DimacsParser;
use log::info;
use solver::{cdcl_solver::CDCLSolver, config::SolverConfig};

mod dimacs;
mod solver;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// File path of instance to parse
    #[arg(short, long)]
    pub path: String,
}

fn main() {
    let args = Args::parse();

    log::set_max_level(log::LevelFilter::Trace);
    env_logger::builder()
        .filter(None, log::LevelFilter::Info)
        .init();

    // Get instance
    let dimacs_parser = DimacsParser::new(&args.path).unwrap();
    let instance = dimacs_parser.parse().unwrap();

    // Initialize solver
    let mut cfg = SolverConfig::default();
    // Whether to enable random stuff / restarts / prefer true in selections
    cfg.restart = true;
    // cfg.decision_policy.prefer_true = false;
    // cfg.decision_policy.random_var = None;
    // cfg.decision_policy.random_pol = false;

    info!("Starting solve attempt...");
    let mut solver = CDCLSolver::new(cfg, instance);
    let start = Instant::now();
    let res = solver.solve();
    let elapsed = start.elapsed();
    match res {
        solver::types::SolveStatus::Unknown => panic!("Solver should never return Unknown"),
        solver::types::SolveStatus::SAT => {
            // Get all assignments.
            let assignments = solver.assignments();
            let display_str = assignments
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            info!("{}", display_str);
        }
        solver::types::SolveStatus::UNSAT => println!("UNSAT"),
    }
    let file = Path::new(&args.path).file_name().unwrap();
    println!(
        "[{}] Status: {}\tElapsed: {:#?}",
        file.to_str().unwrap(),
        res,
        elapsed
    );
}
