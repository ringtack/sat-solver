use std::{
    fs::File,
    io::{BufRead, BufReader, Error, Result},
    path::Path,
};

pub struct DimacsParser {
    file: File,
}

impl DimacsParser {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<DimacsParser> {
        let file = File::open(path)?;
        Ok(Self { file })
    }

    pub fn parse(self) -> Result<SATInstance> {
        // ...
        let reader = BufReader::new(self.file);
        let lines = reader.lines();
        for line in lines {
            let line = line?;
            // Skip until not a comment
            if !line.starts_with("c") {
                break;
            }
        }
        // Read problem line
        let line = lines.next()??;
    }
}
