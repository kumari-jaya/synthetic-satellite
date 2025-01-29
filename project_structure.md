# Vortx Project Structure

```
vortx/
├── docs/                      # Documentation
│   ├── api/                   # API documentation
│   ├── examples/              # Example notebooks and tutorials
│   ├── getting-started/       # Getting started guides
│   ├── research/             # Research algorithms documentation
│   ├── user-guide/           # User guides
│   ├── Dockerfile            # Documentation server Dockerfile
│   └── overrides/            # MkDocs theme overrides
│
├── monitoring/               # Monitoring configuration
│   ├── grafana/             # Grafana dashboards and config
│   │   ├── dashboards/      # Dashboard definitions
│   │   └── provisioning/    # Grafana provisioning config
│   └── prometheus/          # Prometheus configuration
│       └── rules/           # Alert rules
│
├── tools/                   # Development and maintenance tools
│   ├── migrate.py          # Code migration script
│   └── migrate_docker.sh   # Docker image migration script
│
├── vortx/                  # Main package directory
│   ├── api/               # API endpoints
│   ├── cli/              # Command-line interface
│   ├── core/             # Core functionality
│   ├── ml/              # Machine learning models
│   ├── processing/      # Data processing modules
│   ├── synthetic/       # Synthetic data generation
│   └── utils/          # Utility functions
│
├── tests/              # Test suite
│   ├── integration/   # Integration tests
│   └── unit/         # Unit tests
│
├── examples/         # Example applications
│   ├── notebooks/   # Jupyter notebooks
│   └── scripts/     # Example scripts
│
├── docker/          # Docker-related files
│   ├── api/        # API service Dockerfile
│   ├── ml/         # ML service Dockerfile
│   └── worker/     # Worker service Dockerfile
│
├── .github/         # GitHub configuration
│   └── workflows/   # GitHub Actions workflows
│
├── Dockerfile       # Main Dockerfile
├── docker-compose.yml  # Docker Compose configuration
├── requirements.txt    # Python dependencies
├── setup.py           # Package setup
├── README.md          # Project readme
└── LICENSE           # License file
```

## Key Components

1. **Documentation (`docs/`)**
   - Comprehensive documentation using MkDocs
   - API reference, tutorials, and guides
   - Research algorithm documentation

2. **Monitoring (`monitoring/`)**
   - Grafana dashboards for visualization
   - Prometheus configuration for metrics
   - Alert rules for system monitoring

3. **Tools (`tools/`)**
   - Migration utilities
   - Development tools
   - Maintenance scripts

4. **Main Package (`vortx/`)**
   - Core functionality
   - API implementations
   - ML models and processing
   - Utility functions

5. **Tests (`tests/`)**
   - Unit tests for components
   - Integration tests
   - Test utilities

6. **Examples (`examples/`)**
   - Jupyter notebooks
   - Example applications
   - Usage demonstrations

7. **Docker (`docker/`)**
   - Service-specific Dockerfiles
   - Multi-stage builds
   - Development configurations

8. **CI/CD (`.github/`)**
   - GitHub Actions workflows
   - Automated testing
   - Deployment configurations

## Development Guidelines

1. **Code Organization**
   - Follow module-based architecture
   - Keep related functionality together
   - Use clear, descriptive names

2. **Documentation**
   - Document all public APIs
   - Include examples in docstrings
   - Keep documentation up-to-date

3. **Testing**
   - Write unit tests for new features
   - Maintain test coverage
   - Include integration tests

4. **Docker**
   - Use multi-stage builds
   - Optimize image sizes
   - Follow security best practices

5. **Monitoring**
   - Add relevant metrics
   - Create useful dashboards
   - Configure appropriate alerts 