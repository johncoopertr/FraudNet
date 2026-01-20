{
  description = "FraudNet - A deep-learning network built in Rust for detecting Fraud Waste and Abuse";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];
          targets = [ "wasm32-unknown-unknown" ];
        };

        nativeBuildInputs = with pkgs; [
          rustToolchain
          pkg-config
        ];

        buildInputs = with pkgs; [
          # ONNX runtime and dependencies
          onnxruntime
          protobuf
          
          # Development tools
          cargo-watch
          cargo-edit
          cargo-expand
          bacon
          
          # Additional utilities
          just
          git
        ];

      in
      {
        devShells.default = pkgs.mkShell {
          inherit buildInputs nativeBuildInputs;
          
          shellHook = ''
            export SHELL=/run/current-system/sw/bin/bash
            echo "FraudNet Development Environment"
            echo "Rust version: $(rustc --version)"
            echo "Cargo version: $(cargo --version)"
            echo ""
            echo "Available commands:"
            echo "  cargo build    - Build the project"
            echo "  cargo test     - Run tests"
            echo "  cargo run      - Run the project"
            echo "  bacon          - Continuous build/test runner"
            echo ""
          '';

          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
        };
      } // pkgs.lib.optionalAttrs (builtins.pathExists ./Cargo.toml && builtins.pathExists ./Cargo.lock) {
        # Only define the package if both Cargo.toml and Cargo.lock exist
        packages.default = pkgs.rustPlatform.buildRustPackage rec {
          pname = "fraudnet";
          version = "0.1.0";
          
          src = ./.;
          
          cargoLock = {
            lockFile = ./Cargo.lock;
            allowBuiltinFetchGit = true;
          };

          nativeBuildInputs = nativeBuildInputs;
          buildInputs = buildInputs;

          meta = with pkgs.lib; {
            description = "A deep-learning network built in Rust for detecting Fraud Waste and Abuse";
            homepage = "https://github.com/johncoopertr/FraudNet";
            license = licenses.mit;
            maintainers = [ ];
          };
        };
      }
    );
}
