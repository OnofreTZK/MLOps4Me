with import <nixpkgs> {};

pkgs.mkShell {

  name = "Shell name";
  buildInputs = [
    (let
      my-specific-libs = python-packages: with python-packages; [
        pandas
        scikit-learn
        matplotlib
        mlflow
        python3.pkgs.pip
      ];
      python3-with-libs = python3.withPackages my-specific-libs;
     in
     python3-with-libs)
  ];
}
