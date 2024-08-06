# Introduction

## Preparations

To start using the nvironment, including tools like `gcloud` and `gsutil`, simply run: 

```sh
nix develop
```

This will create a semi-isolated environment with all the necessary tools.
Once there, you will be prompted for a password that will decrypt all encrypted files necessary for operations.

## Three Main Commands

The three main commands in discussion are:
1. `buck_down`: Downloads everything for the first time.
2. `fdown`: Turns filters off.
3. `fupp`: Turns filters on.

### Steps

To start all downloads, use the command `buck_down` before anything has been downloaded. If you haven't downloaded anything yet, you can skip this step.

Then, you can turn the filters on and off with `fdown` and `fupp` as needed. Sometimes filters are not really necessary, especially if you plan to switch between different commits without the need to observe the different versions of files. By default, I leave them off.

## Upload

With your nix environment up (`nix develop`) make sure your filters are also up `fupp`. And then simply run 

``sh
git add <path/to/your/file.ext>
``

this will add the file to the cloud so only do this once you are certain of the file!

Afterwards, you must remember to `git commit -m "Your message` your changes so that git can track the reference of the file.


# See Also

A lot of the logic behind the concepts listed below is hidden. Reading about them could be helpful:

1. `clean` and `smudge` filters in git.
2. The mechanism of checking "in" and "out" and how those interact with filters.

File references for these can be found in: 

1. `.githooks/`
2. `.git/config`

