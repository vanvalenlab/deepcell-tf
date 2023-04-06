# Contributing guidelines

We hope this library will be broadly useful to those in the life sciences. For that to be the case, we intend to facilitate and foster a lively development community. To help you get started, please review the following:

## Pull Request Checklist

Before sending your pull requests, make sure you have followed this list.

- Read the guidelines below and ensure your changes are consistent with them.
- Read and abide by our [Code of Conduct](CODE_OF_CONDUCT.md).
- Ensure your changes are consistent with the coding style.
- Run the existing tests and write new tests for your code.

## How to become a contributor and submit new code

### Contributing code

If you have improvements to deepcell-tf, please send us your pull requests! If you are new to the process, Github has a
[how-to](https://help.github.com/articles/using-pull-requests/).

If you want to contribute, start working through the codebase. Navigate to the
[Github "issues" tab](https://github.com/vanvalenlab/deepcell-tf/issues) and start
looking through interesting issues. If you are not sure of where to start, look for one of the smaller/easier issues here i.e.
[issues with the "good first issue" label](https://github.com/vanvalenlab/deepcell-tf/labels/good%20first%20issue)
and then take a look at the
[issues with the "contributions welcome" label](https://github.com/vanvalenlab/deepcell-tf/labels/stat%3Acontributions%20welcome).
These are issues that we believe are well suited for outside contributions. If you decide to start on an issue, leave a comment so that other people know that you're working on it. If you want to help out, but not alone, use the issue comment thread to coordinate.

### Contribution guidelines and standards

Before sending your pull request for
[review](https://github.com/vanvalenlab/deepcell-tf/pulls),
make sure your changes are consistent with the guidelines and follow the
DeepCell coding style.

#### General guidelines and philosophy for contribution

*   Include unit tests when you contribute new features, as they help to a)
    prove that your code works correctly, and b) guard against future breaking
    changes to lower the maintenance cost.
*   Bug fixes also generally require unit tests, because the presence of bugs
    usually indicates insufficient test coverage.
*   When you contribute a new feature to deepcell-tf, the maintenance burden is
    (by default) transferred to the DeepCell team. This means that the benefit
    of the contribution must be compared against the cost of maintaining the
    feature.
*   As every PR requires CI/CD testing, we discourage
    submitting PRs to fix one typo, one warning,etc. We recommend fixing the
    same issue at the file level at least (e.g.: fix all typos in a file, fix
    all compiler warning in a file, etc.)

#### Python coding style

Changes to DeepCell should conform to our style. As our library uses TensorFlow, we follow [Google's Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)

Use `ruff` to check your Python changes. To install `ruff` and check a file
with `ruff` against our custom style definition:

```bash
python -m pip install ruff
ruff .
```

Note `ruff .` should run from the top level directory.

#### Running tests

Use the following commands to run all tests locally:

```bash
pip install -r requirements.txt
pip install -r requirements-test.txt

pytest --cov=deepcell --pep8
```
