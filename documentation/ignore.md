# 
Lovpy allows control of which python source files to be verified through the use of `.lovpyignore` files. Inspired by gitignore files, they are used in quite a similar way. All you have to do is to place a file named `.lovpyignore` under any directory of your project and inside it define files or folders to be excluded. Paths are resolved relatively to the location of `.lovpyignore` file. `*` and `**` can be used as wildcards in they same way they are used in `glob` module. An example `.lovpyignore` file is presented below:
```
source
tests
venv
bin/*.py
```
