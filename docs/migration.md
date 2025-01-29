# Migration Guide: TileFormer to Vortx

This guide helps you migrate from TileFormer to Vortx. The project has been renamed and enhanced with new features, improved performance, and better documentation.

## Key Changes

- Package name changed from `tileformer` to `vortx`
- CLI command changed from `tileformer` to `vo`
- New features and improvements
- Enhanced documentation
- Better performance
- Simplified API

## Installation

1. Uninstall TileFormer:
```bash
pip uninstall tileformer
```

2. Install Vortx:
```bash
pip install vortx

# Or with extras
pip install "vortx[all]"
```

## Code Changes

### Import Statements

Old:
```python
from vortx.synthetic import SyntheticDataGenerator
from vortx.processing import ImageProcessor
from vortx.ml import ModelRegistry
```

New:
```python
from vortx.generation import Generator
from vortx.processing import Processor
from vortx.ml import Registry
```

### CLI Commands

Old:
```bash
vo process input.tif output.tif
vo serve
vo pipeline config.yaml
```

New:
```bash
vo process input.tif output.tif
vo serve
vo pipeline config.yaml
```

### Configuration Files

Old:
```yaml
name: vortx-project
package: vortx
version: 0.1.0
```

New:
```yaml
name: vortx-project
package: vortx
version: 0.1.0
```

### Environment Variables

Old:
```bash
export VORTX_API_KEY=xxx
export VORTX_CONFIG_PATH=/path/to/config
```

New:
```bash
export VORTX_API_KEY=xxx
export VORTX_CONFIG_PATH=/path/to/config
```

### Docker Images

Old:
```bash
docker pull vortx/vortx:latest
docker run vortx/vortx:latest
```

New:
```bash
docker pull vortx/vortx:latest
docker run vortx/vortx:latest
```

## Migration Steps

1. **Update Dependencies**
   - Review your `requirements.txt` or `setup.py`
   - Update all imports from `vortx.*` to `vortx.*`
   - Update the CLI commands from `vo` to `vo`

2. **Update Configuration**
   - Rename configuration files if needed
   - Update environment variables
   - Update Docker configurations

3. **Test Your Code**
   - Run your test suite
   - Check for deprecation warnings
   - Verify functionality

## Common Issues

### Import Errors
- Error: `ModuleNotFoundError: No module named 'vortx'`
- Solution: Install Vortx with `pip install vortx`

### CLI Errors
- Error: `vo: command not found`
- Solution: Ensure Vortx is installed correctly

### Configuration Errors
- Error: `Package 'vortx' not found in configuration`
- Solution: Update your configuration files to use `vortx`

## Additional Resources

- [Vortx Documentation](https://vortx.ai/docs)
- [API Reference](https://vortx.ai/docs/api)
- [Examples](https://vortx.ai/docs/examples)
- [Support Forum](https://github.com/vortx-ai/vortx/discussions)

## Timeline

- **March 2024**: Initial release of Vortx
- **April 2024**: Enhanced features and documentation
- **May 2024**: Long-term support begins

## Need Help?

If you encounter any issues during migration:
1. Check our [FAQ](https://vortx.ai/docs/faq)
2. Join our [Discord community](https://discord.gg/vortx)
3. Open an issue on [GitHub](https://github.com/vortx-ai/vortx/issues)
4. Contact support at support@vortx.ai 