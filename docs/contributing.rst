Contributing Guide
=================

We love your input! We want to make contributing to TileFormer as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

Development Process
-----------------

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

Development Setup
---------------

1. Clone your fork and install development dependencies:

   .. code-block:: bash

      git clone https://github.com/yourusername/tileformer.git
      cd tileformer
      pip install -e ".[dev,docs]"

2. Set up pre-commit hooks:

   .. code-block:: bash

      pre-commit install

3. Create a new branch:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

Code Style
---------

We use several tools to maintain code quality:

- `black` for code formatting
- `isort` for import sorting
- `flake8` for style guide enforcement
- `mypy` for type checking

Run all checks with:

.. code-block:: bash

   # Format code
   black src tests
   isort src tests

   # Run linters
   flake8 src tests
   mypy src tests

Testing
-------

We use pytest for testing. Run the test suite with:

.. code-block:: bash

   pytest

For coverage report:

.. code-block:: bash

   pytest --cov=tileformer tests/

Documentation
------------

We use Sphinx for documentation. Build the docs with:

.. code-block:: bash

   cd docs
   make html

View the docs by opening `_build/html/index.html` in your browser.

Pull Request Process
------------------

1. Update the README.md and documentation with details of changes to the interface.
2. Update the CHANGELOG.md with notes on your changes.
3. The PR will be merged once you have the sign-off of at least one maintainer.

Adding New Features
-----------------

When adding new features:

1. Start with an issue describing the feature.
2. Create a new branch for your feature.
3. Add appropriate tests.
4. Add documentation:
   - API reference
   - Usage examples
   - Update relevant guides
5. Submit a pull request.

Adding New Data Sources
--------------------

When adding a new data source:

1. Create a new module in `src/data_acquisition/sources/`.
2. Implement the standard interface methods:
   - `search_and_download`
   - `get_time_series` (if applicable)
3. Add appropriate tests in `tests/`.
4. Add documentation and examples.

Adding New Algorithms
------------------

When adding new processing algorithms:

1. Add your algorithm to the appropriate processor class.
2. Include docstrings with:
   - Method description
   - Parameter descriptions
   - Return value descriptions
   - Examples
3. Add tests covering:
   - Normal operation
   - Edge cases
   - Error conditions
4. Add documentation and examples.

Issue Labels
-----------

We use the following label categories:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed

Getting Help
-----------

If you need help, you can:

1. Check the documentation
2. Open an issue
3. Join our community discussions
4. Contact the maintainers

Code of Conduct
-------------

Please note that TileFormer has a Code of Conduct. By participating in this project you agree to abide by its terms. 