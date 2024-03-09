use std::{
    path::{Path, PathBuf},
    thread,
    time::Instant,
};

use clap::Parser;
use crossbeam::channel;
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
    let dimacs_parser = DimacsParser::new(PathBuf::from(args.file)).unwrap();
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

    // If multithreaded, spawn that many threads to send to a channel
    let (s, r) = channel::bounded(args.mt);
    // Automatically take care of randomizing stuff if multithreaded
    if args.mt > 1 {
        cfg.restart = true;
        cfg.decision_policy.random_var = Some(0.0025);
        cfg.decision_policy.random_pol = true;
    }

    for i in 0..args.mt {
        info!("Starting solve attempt in solver {}...", i);

        let s = s.clone();
        let mut cfg = cfg.clone();
        let instance = instance.clone();
        thread::spawn(move || {
            // Only restart in half of the solvers
            if cfg.restart {
                cfg.restart = i % 2 == 0;
            }
            // For first two, don't use random decisions
            if cfg.decision_policy.random_var.is_some() {
                if i < 2 {
                    cfg.decision_policy.random_var = None
                }
            }

            let mut solver = CDCLSolver::new(cfg, instance);
            let res = solver.solve();
            s.send(res).unwrap();
        });
    }

    let res = r.recv().unwrap();
    println!("{}", res.result_str());
}
