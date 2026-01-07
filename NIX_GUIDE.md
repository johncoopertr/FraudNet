# Nix Development Environment

This project uses Nix flakes to provide a reproducible development environment.

## Features

- **Rust Toolchain**: Latest stable Rust from oxalica overlay with:
  - `rust-src` for IDE support
  - `rust-analyzer` for intelligent code completion
  - `clippy` for linting
  - `rustfmt` for code formatting
  - WASM target (`wasm32-unknown-unknown`) for web compilation

- **Development Tools**:
  - `cargo-watch` - Automatically rebuild on file changes
  - `cargo-edit` - Easily manage dependencies
  - `cargo-expand` - Expand macros for debugging
  - `bacon` - Background task runner for Rust
  - `just` - Command runner for project tasks

- **Dependencies**:
  - `onnxruntime` - For ONNX model inference
  - `protobuf` - Protocol Buffers compiler

## Usage

### First Time Setup

1. Install Nix with flakes support:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
   ```

2. Clone the repository and enter the directory:
   ```bash
   git clone https://github.com/johncoopertr/FraudNet.git
   cd FraudNet
   ```

### Option 1: Manual Shell Entry

Enter the development shell:
```bash
nix develop
```

This will download and setup all dependencies, then drop you into a shell with all tools available.

### Option 2: Automatic with direnv (Recommended)

1. Install direnv:
   ```bash
   # On macOS
   brew install direnv
   
   # On NixOS
   nix-env -iA nixpkgs.direnv
   ```

2. Hook direnv into your shell (add to `~/.bashrc` or `~/.zshrc`):
   ```bash
   eval "$(direnv hook bash)"  # or zsh, fish, etc.
   ```

3. Allow direnv in the project directory:
   ```bash
   direnv allow
   ```

Now the environment will automatically activate when you `cd` into the project!

## Building the Project

Once a Cargo.toml is created for the project:

```bash
# Build with Nix
nix build

# Or use cargo directly in the dev shell
cargo build --release
```

## Updating Dependencies

To update the Nix flake inputs:
```bash
nix flake update
```

This will update the `flake.lock` file with the latest versions of nixpkgs and the rust overlay.

## Troubleshooting

### Cache Issues

If you encounter issues with the flake, try:
```bash
nix flake update
nix develop --refresh
```

### Clearing Old Generations

To free up disk space from old Nix builds:
```bash
nix-collect-garbage -d
```
