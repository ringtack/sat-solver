use clap::Parser;
use dimacs::parser::DimacsParser;
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

    // Get instance
    let dimacs_parser = DimacsParser::new(args.path).unwrap();
    let instance = dimacs_parser.parse().unwrap();

    // Initialize solver
    let cfg = SolverConfig::default();
    let mut solver = CDCLSolver::new(cfg, instance);
    let res = solver.solve();
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
            println!("{}", display_str);
        }
        solver::types::SolveStatus::UNSAT => println!("UNSAT"),
    }
}
