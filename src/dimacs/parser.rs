use anyhow::{Context, Result};
use fxhash::FxHashSet;
use log::debug;
use std::{
    fs::File,
    io::{BufRead, BufReader, Error, ErrorKind},
    path::Path,
};

use super::sat_instance::{Clause, Literal, SATInstance};

pub struct DimacsParser {
    file: File,
}

impl DimacsParser {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<DimacsParser> {
        let file = File::open(path)?;
        Ok(Self { file })
    }

    // Parses the file provided during construction into a SAT Instance.
    pub fn parse(self) -> Result<SATInstance> {
        let reader = BufReader::new(self.file);
        let mut lines = reader.lines();
        let mut line = lines.next().unwrap().unwrap();
        while line.starts_with("c") {
            line = lines.next().unwrap().unwrap();
        }
        // Line now has problem line; read it
        let tokens = line.split_whitespace().collect::<Vec<_>>();

        if tokens[0] != "p" {
            return Err(Error::from(ErrorKind::InvalidInput))
                .context("DIMACS file does not have problem line");
        }
        if tokens[1] != "cnf" {
            return Err(Error::from(ErrorKind::InvalidInput))
                .context("DIMACS file format is not cnf");
        }
        // Parse variables and clauses
        let n_vars = tokens[2].parse::<usize>()?;
        let n_clauses = tokens[3].parse::<usize>()?;

        let mut sat_instance = SATInstance {
            n_vars,
            n_clauses,
            clauses: Vec::with_capacity(n_clauses),
            vars: FxHashSet::default(),
        };
        // Parse clauses
        for line in lines {
            let line = line?;
            let tokens = line.split_whitespace().collect::<Vec<_>>();
            // Skip comments
            if tokens.len() == 0 || tokens[0].starts_with("c") {
                continue;
            }
            // Verify format
            if *tokens.last().unwrap() != "0" {
                return Err(Error::from(ErrorKind::InvalidInput)).context(format!(
                    "Clause line {} does not end with 0",
                    tokens.join(" ")
                ));
            }
            // Create clause
            let lits = tokens
                .iter()
                .take(tokens.len() - 1)
                .map(|i| {
                    let l = i.parse::<Literal>().unwrap();
                    sat_instance.vars.insert(l.abs());
                    l
                })
                .collect::<Vec<_>>();
            sat_instance.clauses.push(Clause { lits });
        }

        debug!("{:#?}", &sat_instance);

        Ok(sat_instance)
    }
}
