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
        ];
        # Now we add zsh as the shell
        shell = pkgs.zsh;

        shellHook = ''
          echo "Welcome to the ITL remote workstation"
          # Create some aliases to my scripts
          export IN_FLAKE=1
          export SHELL=${nixpkgs.legacyPackages.x86_64-linux.zsh}/bin/zsh
          # Set out the hooks for blob versioning 
          #find .git/hooks -type l -exec rm {} \; && find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
          GIT_CONFIG_FILE="$PWD/.git/config"
      
          if [ -f "$GIT_CONFIG_FILE" ]; then
            if ! grep -q "\[filter \"gcs-lfs\"\]" "$GIT_CONFIG_FILE"; then
              echo "Adding gcs-lfs filter configuration to .git/config"
              git config --local filter.gcs-lfs.clean ".githooks/clean-filter.sh %f"
              chmod +x .githooks/clean-filter.sh
              git config --local filter.gcs-lfs.smudge ".githooks/gcs_smudge.sh %f"
              chmod +x .githooks/gcs_smudge.sh
              git config --local filter.gcs-lfs.required true
            else
              echo ""
            fi
          fi

          # Set all of them as executables
          #chmod +x .git/hooks/*
          source ./reproducibility/scripts/initialize_scripts.sh
          exec ${nixpkgs.legacyPackages.x86_64-linux.zsh}/bin/zsh          
        '';
      };
    };
}
