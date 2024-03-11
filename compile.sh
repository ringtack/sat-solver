#!/bin/bash

########################################
############# CSCI 2951-O ##############
########################################

# Check that Rust exists
if ! command -v rustc &> /dev/null; then
    echo "rustc could not be found. Please ensure Rust is installed."
    exit 1
fi

# Upgrade rust if necessary
RUST_VER="1.76.0"
CURR_VER=$(rustc --version | awk '{print $2}')

if [ "$CURRENT_VERSION" != "$DESIRED_VERSION" ]; then
    echo "Rust version mismatch. Found: $CURR_VER, desired: $RUST_VER."
    echo "Upgrading Rust..."
    rustup upgrade &> /dev/null
fi

# Change to stable if necessary
CURRENT_TOOLCHAIN=$(rustup show active-toolchain | awk '{print $1}')
if [[ $CURRENT_TOOLCHAIN != *"stable"* ]]; then
  echo "The current Rust toolchain is not stable; setting default to stable..."
    rustup default stable
fi

# Run only if PGO optimization specified by env var; disabled by default
if [[ -n $PGO ]]; then
  echo "PGO optimization specified; installing LLVM tools if necessary (for \`llvm-profdata\`)"
  rustup component add llvm-tools-preview

  # Remove the old profile data
  rm -rf /tmp/pgo-data
  # Compile with PGO instrumentation
  echo "Compiling with PGO instrumentation..."
  RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --release &> /dev/null
  # Run the program on three inputs to gather profile data:
  echo "Running the program on three inputs (C1065_064.cnf, U50_1065_038.cnf, and U50_4450_035.cnf) to gather profile data..."
  ./target/release/sat-solver -f input/C1065_064.cnf > /dev/null 2>&1
  echo "C1065_064.cnf profile done."
  ./target/release/sat-solver -f input/U50_1065_038.cnf > /dev/null 2>&1
  echo "U50_1065_038.cnf profile done."
  ./target/release/sat-solver -f input/U50_4450_035.cnf > /dev/null 2>&1
  echo "U50_4450_035.cnf profile done."

  # Merge profile data together
  echo "Merging profile data together."
  llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data
  # Use profile data to optimize the program
  echo "Recompiling with PGO..."
  RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata" \
      cargo build --release &> /dev/null
else
  # Compile without PGO optimization
  echo "Compiling without PGO. To enable, set the environment variable \`PGO=1\`."
  cargo build --release &> /dev/null
fi

echo "Compilation done. To run all inputs, see \`./runAll.sh\`; for a specific input file, use \`./run.sh\`."
