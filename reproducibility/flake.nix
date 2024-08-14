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
        ];
        # Now we add zsh as the shell
        shell = pkgs.zsh;

        shellHook = ''
          echo "Welcome to the ITL remote workstation"
        '';
      };
    };
}
