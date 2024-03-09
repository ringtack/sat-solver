#!/bin/bash

########################################
############# CSCI 2951-O ##############
########################################

# Upgrade rust if necessary
rustup upgrade
rustup default stable

# Run only if PGO optimization specified by env var; disabled by default
if [[ -n $PGO ]]; then
  rustup component add llvm-tools-preview

  # Remove the old profile data
  rm -rf /tmp/pgo-data
  # Compile with PGO instrumentation
  echo "Compiling with PGO instrumentation"
  RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --release
  # Run the program on three inputs to gather profile data:
  echo "Running the program on three inputs (C1065_064.cnf, U50_1065_038.cnf, and U50_4450_035.cnf) to gather profile data"
  ./target/release/project1 -f input/C1065_064.cnf > /dev/null 2>&1
  echo "C1065_064.cnf profile done"
  ./target/release/project1 -f input/U50_1065_038.cnf > /dev/null 2>&1
  echo "U50_1065_038.cnf profile done"
  ./target/release/project1 -f input/U50_4450_035.cnf > /dev/null 2>&1
  echo "U50_4450_035.cnf profile done"

  # Merge profile data together
  llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data
  # Use profile data to optimize the program
  echo "Recompiling with PGO optimization"
  RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata" \
      cargo build --release
else
  # Compile without PGO optimization
  cargo build --release
fi
