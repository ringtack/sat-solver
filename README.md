# SAT Solver

A CDCL-based SAT solver implemented in Rust, with watched literals, EVSIDS, Glucose-based clause deletion, conflict clause minimization, and Luby-based restarts. Based heavily on [MiniSat](http://minisat.se/).

For implementation details, optimizations, and experimental results, see the [detailed report](./report.pdf).

> Implemented for project 1 (Vehicle Customization) in [CSCI2951O: Prescriptive Analytics](https://cs.brown.edu/courses/csci2951-o/index.html). 

# Usage

As a prerequisite, your machine must have [`rustup`](https://rustup.rs/) installed.

To build the program:

```bash
$ ./compile.sh
```

This updates the Rust toolchain to the latest stable version, then builds a release binary (`./target/release/sat-solver`).

> For [profile-guided optimization](https://doc.rust-lang.org/rustc/profile-guided-optimization.html) (PGO) support, set the environment variable `PGO=1`.

To run all inputs:

```bash
$ ./runAll.sh <inputFolder> <timeout (sec)> <outputFile> [<binary args...>]
```

To run a specific input:

```bash
$ ./run.sh <inputInstance> [<binary args...>]
```

> To run with profiling, set the environment variable `FLAMEGRAPH=1`. This outputs a flamegraph to `./flamegraphs/`.

View binary args with `./sat-solver -h`. Currently, the following parameters can be tuned:

```
Usage: sat-solver [OPTIONS] --file <FILE>

Options:
  -f, --file <FILE>          File path of instance to parse
  -r, --restart              Whether to restart
  -t, --true-pref            Whether to prefer true in decisions
  -v, --var-rand <VAR_RAND>  Whether to randomly decide vars (i.e. w/ what freq) [default: 0]
  -p, --pol-rand             Whether to randomize polarity or remember it
  -m, --mt <MT>              Whether to run multithreaded (and how many cores) [default: 1]
  -h, --help                 Print help
  -V, --version              Print version
```
