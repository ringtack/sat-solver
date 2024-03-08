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
    pub file: String,

    /// Whether to restart
    #[arg(short, long, default_value_t = false)]
    pub restart: bool,

    /// Whether to prefer true in decisions
    #[arg(short, long, default_value_t = false)]
    pub true_pref: bool,

    /// Whether to randomly decide vars (i.e. w/ what freq)
    #[arg(short, long, default_value_t = 0.0)]
    pub var_rand: f64,

    /// Whether to randomize polarity or remember it
    #[arg(short, long, default_value_t = false)]
    pub pol_rand: bool,

    /// Whether to run multithreaded (and how many cores)
    #[arg(short, long, default_value_t = 1)]
    pub mt: usize,
}

fn main() {
    let args = Args::parse();

    log::set_max_level(log::LevelFilter::Trace);
    env_logger::builder()
        .filter(None, log::LevelFilter::Info)
        .init();

    // Get instance
    let dimacs_parser = DimacsParser::new(&args.file).unwrap();
    let instance = dimacs_parser.parse().unwrap();

    // Initialize solver
    let mut cfg = SolverConfig::default();
    // Whether to enable random stuff / restarts / prefer true in selections
    cfg.restart = args.restart;
    cfg.decision_policy.prefer_true = args.true_pref;
    cfg.decision_policy.random_var = if args.var_rand > 0. {
        Some(args.var_rand)
    } else {
        None
    };
    cfg.decision_policy.random_pol = args.pol_rand;

    info!("Config: {:#?}", cfg);

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
    let file = Path::new(&args.file).file_name().unwrap();
    println!(
        "[{}] Status: {}\tElapsed: {:#?}",
        file.to_str().unwrap(),
        res,
        elapsed
    );
}
