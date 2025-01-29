Quickstart Guide
===============

This guide will help you get started with TileFormer quickly.

Installation
-----------

First, install TileFormer using pip:

.. code-block:: bash

   pip install tileformer

For development features:

.. code-block:: bash

   pip install tileformer[dev,docs]

Basic Usage
----------

1. Data Acquisition
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tileformer.data_acquisition import DataManager

   # Initialize data manager
   data_manager = DataManager()

   # Get satellite data
   data = data_manager.get_satellite_data(
       bbox=(-73.9857, 40.7484, -73.9798, 40.7520),  # NYC area
       start_date="2024-01-01",
       end_date="2024-01-31",
       collections=["sentinel-2"],
       cloud_cover=20.0
   )

2. Image Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from tileformer.processors import AdvancedProcessor

   # Initialize processor
   processor = AdvancedProcessor()

   # Calculate indices
   indices = processor.calculate_indices(
       data["sentinel-2"][0]["data"],
       bands={"NIR": 3, "RED": 2},
       indices=["NDVI", "EVI"]
   )

   # Detect clouds
   cloud_mask = processor.detect_clouds(
       data["sentinel-2"][0]["data"],
       method="statistical"
   )

3. Time Series Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from datetime import datetime, timedelta

   # Create time series data
   dates = []
   images = []
   start_date = datetime(2024, 1, 1)

   for i in range(12):
       date = start_date + timedelta(days=30*i)
       data = data_manager.get_satellite_data(
           bbox=(-73.9857, 40.7484, -73.9798, 40.7520),
           start_date=date.strftime("%Y-%m-%d"),
           end_date=(date + timedelta(days=1)).strftime("%Y-%m-%d"),
           collections=["sentinel-2"]
       )
       if "sentinel-2" in data and data["sentinel-2"]:
           dates.append(date)
           images.append(data["sentinel-2"][0]["data"])

   # Analyze time series
   results = processor.analyze_time_series(
       images,
       dates,
       method="seasonal"
   )

4. 3D and AR
~~~~~~~~~~

.. code-block:: python

   from tileformer.data_acquisition.sources import MobileMetaverseAPI

   # Initialize API
   mobile_api = MobileMetaverseAPI()

   # Convert to 3D model
   model = mobile_api.convert_to_3d(
       vector_data=buildings_gdf,
       raster_data=data["sentinel-2"][0]["data"],
       format="glb",
       attributes=["height"]
   )

   # Create AR scene
   scene = mobile_api.create_ar_scene(
       models=[model],
       format="usdz",
       scene_scale=1.0
   )

Advanced Features
---------------

1. Cloud-Optimized GeoTIFFs
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tileformer.data_acquisition.sources import COGSTACAPI

   # Initialize API
   cog_api = COGSTACAPI()

   # Get COG data
   cog_data = cog_api.get_cog_data(
       url="https://example.com/data.tif",
       bbox=(-73.9857, 40.7484, -73.9798, 40.7520),
       resolution=10.0
   )

2. Google Earth Engine
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tileformer.data_acquisition.sources import EarthEngineAPI

   # Initialize API
   ee_api = EarthEngineAPI()

   # Get time series
   ts_data = ee_api.get_time_series(
       bbox=(-73.9857, 40.7484, -73.9798, 40.7520),
       start_date="2024-01-01",
       end_date="2024-12-31",
       collection="sentinel-2",
       band="B8"
   )

3. Data Fusion
~~~~~~~~~~~~

.. code-block:: python

   from tileformer.processors import DataFusion

   # Initialize fusion processor
   fusion = DataFusion()

   # Combine multiple sources
   fused_data = fusion.combine_sources(
       optical=data["sentinel-2"][0]["data"],
       sar=sar_data,
       dem=elevation_data
   )

Next Steps
---------

- Check out the :doc:`user_guide/index` for detailed information
- See :doc:`examples/index` for more examples
- Read the :doc:`api_reference/index` for complete API documentation 