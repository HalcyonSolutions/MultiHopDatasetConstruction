{
  description = "Overall Linux Environment for Computer Related to MultiHop";

  # inputs = { nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-24.05"; };
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";
  };

  outputs = { self, nixpkgs }:
    let 
      # Try to guess system ourselves 
      # system = builtins.currentSystem;
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      buck_down = pkgs.stdenv.mkDerivation {
        pname = "buck_down";
        version = "1.0";

        buildInputs = [ pkgs.git ];

        # Skip unpackPhase since there's no source to unpack
        unpackPhase = "true";

        installPhase = ''
          mkdir -p $out/bin
          echo '#!/bin/sh' > $out/bin/buck_down
          # echo 'git checkout HEAD -- "$(git rev-parse --show-toplevel)"' >> $out/bin/buck_down
          # Turn on the smudge and clean filters
          echo 'git config --local filter.gcs-lfs.smudge ".githooks/gcs_smudge.sh %f"' >> $out/bin/buck_down
          echo 'git config --local filter.gcs-lfs.clean ".githooks/clean-filter.sh %f"' >> $out/bin/buck_down
          echo 'git config --local filter.gcs-lfs.required true' >> $out/bin/buck_down
          # echo 'git checkout HEAD -- "$(git rev-parse --show-toplevel)"' >> $out/bin/buck_down
          echo 'git rm --cached -r .' >> $out/bin/buck_down
          echo 'git checkout HEAD -- .' >> $out/bin/buck_down
          chmod +x $out/bin/buck_down
        '';
      };
      fupp = pkgs.stdenv.mkDerivation {
        pname = "fupp";
        version = "1.0";

        buildInputs = [ pkgs.git ];

        # Skip unpackPhase since there's no source to unpack
        unpackPhase = "true";

        installPhase = ''
          mkdir -p $out/bin
          echo '#!/bin/sh' > $out/bin/fupp
          # Turn on the smudge and clean filters
          echo 'git config --local filter.gcs-lfs.smudge ".githooks/gcs_smudge.sh %f"' >> $out/bin/fupp
          echo 'git config --local filter.gcs-lfs.clean ".githooks/clean-filter.sh %f"' >> $out/bin/fupp
          echo 'git config --local filter.gcs-lfs.required true' >> $out/bin/fupp
          chmod +x $out/bin/fupp
        '';
      };
      fdown = pkgs.stdenv.mkDerivation {
        pname = "fdown";
        version = "1.0";

        buildInputs = [ pkgs.git ];

        # Skip unpackPhase since there's no source to unpack
        unpackPhase = "true";

        installPhase = ''
          mkdir -p $out/bin
          echo '#!/bin/sh' > $out/bin/fdown
          # echo 'git checkout HEAD -- "$(git rev-parse --show-toplevel)"' >> $out/bin/fdown
          # Turn of the smudge and clean filters
          echo 'git config --local filter.gcs-lfs.smudge "cat"' >> $out/bin/fdown
          echo 'git config --local filter.gcs-lfs.clean "cat"' >> $out/bin/fdown
          echo 'git config --local filter.gcs-lfs.required false' >> $out/bin/fdown
          chmod +x $out/bin/fdown
        '';
      };
    in {
      devShell.${system} = pkgs.mkShell {
        buildInputs = with pkgs; [
          python310
          git
          lazygit
          poetry
          rsync
          zsh
          neo4j
          xxd
          fdown
	  google-cloud-sdk		
          fupp
          buck_down
        ];
        # Now we add zsh as the shell
        shell = pkgs.zsh;

        shellHook = ''
          echo "Welcome to the ITL remote workstation"
          # Create some aliases to my scripts
          export IN_FLAKE=1
          export SHELL=${nixpkgs.legacyPackages.x86_64-linux.zsh}/bin/zsh
          export RPROMPT="%F{cyan}( Poetry)%f"
          # Set out the hooks for blob versioning 
          #find .git/hooks -type l -exec rm {} \; && find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
          GIT_CONFIG_FILE="$PWD/.git/config"
      
          if [ -f "$GIT_CONFIG_FILE" ]; then
            if ! grep -q "\[filter \"gcs-lfs\"\]" "$GIT_CONFIG_FILE"; then
              # echo "Adding gcs-lfs filter configuration to .git/config"
              # git config --local filter.gcs-lfs.clean ".githooks/clean-filter.sh %f"
              # chmod +x .githooks/clean-filter.sh
              # git config --local filter.gcs-lfs.smudge ".githooks/gcs_smudge.sh %f"
              # chmod +x .githooks/gcs_smudge.sh
              # git config --local filter.gcs-lfs.required true
              # Just set a warning in yellow saying they have to install to install filters or run `buck_down`
              echo -e "\033[0;33m::Warning: You have not installed the gcs-lfs filter. Please run 'buck_down' to install it.\033[0m"
            else
              echo ""
            fi
          fi

          # Move the pre-push hook to the correct location
          if [ -f ".githooks/pre-push" ]; then
            mkdir -p .git/hooks
            cp .githooks/pre-push .git/hooks/pre-push
            chmod +x .git/hooks/pre-push
          else 
            echo -e "\033[0;31m ::Error: Missing pre-push hook. You will not be able to store your large files in the cloud. (A you perhaps not running `nix develop` from your repo root)? \033[0m"
          fi

          # Set all of them as executables
          #chmod +x .git/hooks/*
          source ./reproducibility/scripts/initialize_scripts.sh
          exec ${nixpkgs.legacyPackages.x86_64-linux.zsh}/bin/zsh          
        '';
      };
    };
}
