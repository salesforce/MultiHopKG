{
  description = "Application packaged using poetry2nix";

  inputs = { flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix } @ inputs:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        # pkgs = nixpkgs.legacyPackages.${system};
        pkgs = import nixpkgs { 
          config.allowUnfree = true;
          cudaSupport = true;
          inherit 
            system;
        };
        # customPythonPackages = pkgs.python310.pkgs.extend (pythonSelf: pythonSuper: {
        #   pytorch = pythonSuper.pytorch.override {
        #     cudaSupport = true;
        #     cudatoolkit = pkgs.cudaPackages.cudatoolkit;
        #   };
        # });

        libs = [
          pkgs.stdenv.cc.cc.lib
          pkgs.xorg.libX11
          pkgs.ncurses5
          pkgs.libGL
          pkgs.libzip
          pkgs.glib
        ];

        poetry2nix = inputs.poetry2nix.lib.mkPoetry2Nix { inherit pkgs; };

        env = poetry2nix.mkPoetryEnv {
          projectDir = ./.;
          editablePackageSources = {
            my-app = ./src;
          };
          python = pkgs.python310;
          preferWheels = true;
          overrides = poetry2nix.overrides.withDefaults (
            final: prev: {
              maturin = prev.maturin.overrideAttrs(old: {
                nativeBuildInputs = old.nativeBuildInputs ++ [ pkgs.cargo pkgs.rustc ];
              });
              safetensors = prev.safetensors.override {
                preferWheel = true;
              };
              tokenizers = prev.tokenizers.override {
                preferWheel = true;
              };
              pysbd = prev.pysbd.overridePythonAttrs(old: rec {
                preferWheel = true;
                postInstall = ''
                    echo "Cleaning up __pycache__ directories..."
                    rm -rf $out/lib/python3.10/site-packages/benchmarks

                    # Optionally call the old postInstall hook if it exists
                    ${old.postInstall or ""}
                '';
              });
              torch = prev.torch.overrideAttrs(old: {
                cudaSupport = true;
                cudatoolkit = pkgs.cudaPackages.cudatoolkit;
                preferWheel = false;
                  nativeBuildInputs = old.nativeBuildInputs ++ [ pkgs.cudaPackages.cudnn ];
              });
              pytorch = prev.pytorch.overrideAttrs(old: {
                cudaSupport = true;
                cudatoolkit = pkgs.cudaPackages.cudatoolkit;
                preferWheel = false;
                  nativeBuildInputs = old.nativeBuildInputs ++ [ pkgs.cudaPackages.cudnn ];
              });
            }
          );
        };

      in
      {
      devShells.default = pkgs.mkShell {
          # nativeBuildInputs is for tools that are used in the moment of installation.
          nativeBuildInputs = [
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cudnn
          ];
          buildInputs = [
            # For all other python dependencies.
            env
            pkgs.cudaPackages.cudatoolkit
            pkgs.zsh
            pkgs.git
            pkgs.stdenv.cc.cc

          ];
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
          CUDA_PATH = pkgs.cudaPackages.cudatoolkit;

          # NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc pkgs.glibc pkgs.gcc.cc ];
          # NIX_LD = builtins.readFile "${pkgs.stdenv.cc}/nix-support/dynamic-linker";
          shellHook = ''
            # export SHELL=${pkgs.zsh}/bin/zsh
            export INFLAKE="RUN"
            export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit};
            export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit};
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:/usr/lib64:/usr/local/cuda/lib64:/run/opengl-driver/lib:/run/opengl-driver-32/lib;
            echo "Welcome to the ITL Benchmarking Environment."
            #  export RPROMPT="%F{cyan}(󱄅dev)%f"
            # exec zsh
          '';
      };

        # Shell for poetry.
        #
        #     nix develop .#poetry
        #
        # Use this shell for changes to pyproject.toml and poetry.lock.
        devShells.poetry = pkgs.mkShell {
          packages = [ pkgs.poetry ];
          shellHook = ''
            export NIX_LD = 
            export RPROMPT="%F{cyan}( Poetry)%f"
            exec zsh
          '';
        };
      });
}
