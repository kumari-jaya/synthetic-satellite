Welcome to TileFormer's documentation!
====================================

TileFormer is a comprehensive library for geospatial data acquisition and processing, with support for various data sources, advanced image processing algorithms, and 3D/AR/VR capabilities.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide/index
   api_reference/index
   examples/index
   contributing
   changelog

Features
--------

* **Multiple Data Sources**:
    * Google Earth Engine (Sentinel-2, Landsat-8, MODIS)
    * Planetary Computer
    * Cloud-Optimized GeoTIFFs (COG)
    * SpatioTemporal Asset Catalogs (STAC)
    * Planet Labs
    * Maxar Open Data

* **Advanced Image Processing**:
    * Cloud Detection (Statistical, ML, Threshold methods)
    * Spectral Indices (NDVI, NDWI, EVI, SAVI)
    * Change Detection
    * Super-resolution
    * Pansharpening
    * Object Segmentation
    * Time Series Analysis

* **3D and AR/VR**:
    * 3D Model Generation
    * AR Scene Creation
    * Terrain Modeling
    * Mobile Optimization

* **Data Management**:
    * Efficient Caching
    * Cloud Storage Integration
    * DuckDB-based Metadata
    * GeoParquet Support

Installation
------------

You can install TileFormer using pip:

.. code-block:: bash

   pip install tileformer

For development installation with additional tools:

.. code-block:: bash

   pip install tileformer[dev,docs]

Quick Example
------------

Here's a simple example of using TileFormer:

.. code-block:: python

   from tileformer.data_acquisition import DataManager
   from tileformer.processors import AdvancedProcessor

   # Initialize components
   data_manager = DataManager()
   processor = AdvancedProcessor()

   # Get satellite data
   data = data_manager.get_satellite_data(
       bbox=(-73.9857, 40.7484, -73.9798, 40.7520),  # NYC area
       start_date="2024-01-01",
       end_date="2024-01-31",
       collections=["sentinel-2"]
   )

   # Process data
   indices = processor.calculate_indices(
       data["sentinel-2"][0]["data"],
       bands={"NIR": 3, "RED": 2},
       indices=["NDVI"]
   )

Contributing
-----------

We welcome contributions! Please see our :doc:`contributing` guide for details.

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 