import os
import argparse
import yaml
import numpy as np
import pandas as pd
import glob
import pyproj

import fiona
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from dateutil.rrule import rrule, DAILY
from datetime import datetime
import geopandas as gpd

def get_args():
    # Store all parameters for easy retrieval
    parser = argparse.ArgumentParser(
        description = 'Module for creating productive demand time series using irrigibility map')
    parser.add_argument('--params_filename',
                        type=str,
                        default='params.yaml',
                        help = 'Loads productive demand time series parameters')
    args = parser.parse_args()
    config = yaml.load(open(args.params_filename), Loader=yaml.FullLoader)
    for k,v in config.items():
        args.__dict__[k] = v

    return args

def load_regions_shapefile(args):
    # Load city-regions shapefiles
    shp_filename = os.path.join(args.data_dir, 'region_shapefiles', 'country_Theissen_Inters.shp')
    regions_shp = gpd.read_file(shp_filename).to_crs(epsg=32637)
    return regions_shp

def load_eth_shapefile(args):
    # Load shapefile for all of Ethiopia
    shp_filename = os.path.join(args.data_dir, 'eth_shapefile', 'ETH_outline_epsg4326.shp')
    regions_shp = gpd.read_file(shp_filename).to_crs(epsg=4326)
    return regions_shp


def load_irrigibility(args):
    '''
    Irrigation map is at 1km resolution
    '''

    filename = os.path.join(args.data_dir, 'worqlul_irrigibility', 'land_suitability.tif')

    # Determine whether you want to save updated irrigibility map
    save_cropped = True

    # Load irrigibility tif
    with rasterio.open(filename, 'r+') as src:
        # Threshold based on the irrigibility threshold
        cropped_src = src.read() >= args.irrigibility_lb

        # Save values of thresholded irrigibility map -- more helpful than binaries
        cropped_src = np.array(cropped_src, dtype = np.uint8)
        metadata = src.meta

        # Create new save image destination
        save_image_dest = os.path.join(args.data_dir, 'worqlul_irrigibility',
                                       'land_suitability_cropped_irrigibility_lb_{}.tif'.format(args.irrigibility_lb))

        # Save new cropped irrigibility map
        if save_cropped:
            with rasterio.open(save_image_dest, 'w+', **metadata) as dest:
                dest.write(cropped_src)

        return save_image_dest

def rescale_reprojected_rainfall():
    '''
    This rescaling method can change -- my best guess at what it could be.
    '''

    # Find file names for original and reprojected CHIRPS rainfall images
    original    = glob.glob(args.tmp_dir + '/eth.tif')[0]
    reprojected = glob.glob(args.tmp_dir + '/eth_reproject.tif')[0]

    # Load images
    with rasterio.open(original, 'r') as orig:
        with rasterio.open(reprojected, 'r') as reproj:
            orig_min, orig_max     = np.min(orig.read()),   np.max(orig.read())
            reproj_min, reproj_max = np.min(reproj.read()), np.max(reproj.read())

            meta = reproj.meta

            # Rescale reprojected image if necessary to rainfall maximum on the original image
            if orig_max > reproj_max:
                reproj_array = np.maximum(reproj.read(), 0) * orig_max/reproj_max
            else:
                reproj_array = np.maximum(reproj.read(), 0) * reproj_max / orig_max

    # Removed reprojected, unscaled image
    os.remove(reprojected)

    # Save reprojected, scaled image
    with rasterio.open(reprojected, 'w+', **meta) as new_reproj:
        new_reproj.write(reproj_array)


def reproject_chirps_rainfall(args, chirps_filename, irrigibility_filename, eth_shp):
    '''
    CHIRPS data is at 0.05 degree spatial resolution (~5.5 km)
    Reproject to 1000m resolution
    Can also change the reprojection method
    '''

    # Load Ethiopia shapefile -- Thanks Tim!
    eth_poly = eth_shp['geometry'].iloc[0]



    # Load irrigibility map
    with rasterio.open(irrigibility_filename, 'r+') as irrig_src:

        dst_crs = irrig_src.meta['crs']

        # Open CHIRPS rainfall file and crop to Ethiopia
        with rasterio.open(chirps_filename, 'r+') as chirps_src:
            # print(chirps_src.meta)
            # Crop CHIRPS data to Ethiopia
            out_rainfall_image, out_rainfall_transform = mask(chirps_src, [eth_poly], crop=True)
            cropped_chirps_metadata = chirps_src.meta
            out_rainfall_height = out_rainfall_image.shape[1]
            out_rainfall_width = out_rainfall_image.shape[2]
            cropped_chirps_metadata.update({"transform": out_rainfall_transform,
                             "height": out_rainfall_height,
                             "width": out_rainfall_width,
                             "crs": chirps_src.crs
                             })

            # Save cropped Ethiopia CHIRPS data
            tmp_save_low_res = os.path.join(args.data_dir, 'chirps', 'tmp', 'eth.tif')
            with rasterio.open(tmp_save_low_res, 'w', **cropped_chirps_metadata) as eth_chirps:
                eth_chirps.write(out_rainfall_image)

                # Copy the meta data for the irrigiblity map
                kwargs = irrig_src.meta.copy()

                # Set the rainfall data type to float
                kwargs['dtype'] = chirps_src.meta['dtype']

                # Create temporary save for reporjected CHIRPS tif
                tmp_save = tmp_save_low_res[0:-4] + '_reproject.tif'

                ## Reproject and upsample Ethiopia CHIRPS data (this also saves the image)
                with rasterio.open(tmp_save, 'w+', **kwargs) as reproj_dst:
                    reproject(
                        source        = rasterio.band(eth_chirps, 1),
                        destination   = rasterio.band(reproj_dst, 1),
                        # Resampling method: this can change! Really don't know what would be best
                        resampling    = Resampling.cubic_spline)

        # Rescale the reprojected CHIRPS file if desired
        if args.rescale_reprojected:
            rescale_reprojected_rainfall()


# def project_wsg_shape_to_csr(shape, csr):
#
#     # Need to reproject the city-region shapefiles into the same coordinate reference system as the irrigiblity map
#     project = lambda x, y: pyproj.transform(
#     pyproj.Proj('epsg:4326'),
#     pyproj.Proj(csr),
#     x, y)
#     return transform(project, shape)


def calculate_energy_deficit(args, thresholded_irrig_filename, regions_shp):

    # Find reprojected image file
    chirps_reproj = glob.glob(args.tmp_dir + '/eth_reproject.tif')[0]

    num_regions = len(regions_shp)

    # Create array for saving model outputs. For now, this is given in GWh
    regional_energy_deficit = np.zeros((num_regions))

    # Open thresholded irrigibility map
    with rasterio.open(thresholded_irrig_filename, 'r') as irrig_src:
        # print(irrig_src.meta)

        # Open reprojected daily CHIRPS tif
        with rasterio.open(chirps_reproj, 'r') as chirps_src:
            # print(chirps_src.meta)

            # Loop through the city-regions
            for i in range(len(regions_shp)):
                # Load the city-region polygon
                region_poly = regions_shp['geometry'].iloc[i]

                # Reproject the city-region polygon

                # Crop reprojected CHIRPS and irrigibility map
                out_chirps_image, out_chirps_transform = mask(chirps_src, [region_poly], crop=True)
                out_irrig_image,  out_irrig_transform  = mask(irrig_src,  [region_poly], crop=True)

                # Count the number of pixels that are irrigible and get indices
                irrig_pixels_indices = np.nonzero(out_irrig_image)

                # Find rainfall data at these pixels
                rainfall_overlap = out_chirps_image[irrig_pixels_indices]

                # Calculate water deficit at pixelwise basis
                water_deficit = np.maximum(args.h20_req - rainfall_overlap, 0) # mm

                # Calculate volume of water deficit, each pixel is 1000m x 1000m
                total_water_deficit_m3 = 1e-3 * np.sum(water_deficit) * 1e6

                # Calculate total energy required for water deficit, density = 1000 kg/m3, convert to GWh
                total_daily_energy = (1000 * total_water_deficit_m3 * 9.81 * args.gw_depth * args.frac_irrig *
                                      1/args.w2w_eff) / (1e9 * 3600)

                # Store daily energy requirements in an array
                regional_energy_deficit[i] = total_daily_energy

                ## Can use this code to visualize areas with energy/water deficits
                # meta = chirps_src.meta
                # height = out_chirps_image.shape[1]
                # width = out_chirps_image.shape[2]
                # meta.update({"transform": out_chirps_transform,
                #              "height": height,
                #              "width": width,
                #              "crs": chirps_src.crs
                #              })
                #
                # temp_name = chirps_reproj[0:-4] + '_region' + str(i) + '.tif'
                # with rasterio.open(temp_name, 'w', **meta) as dest:
                #     dest.write(array_overlap)

    return regional_energy_deficit


if __name__ == '__main__':
    # Get arguments from .yaml
    args = get_args()

    # Threshold irrigibility map based on irrigibility lowerbound
    thresholded_irrig_filename = load_irrigibility(args)

    # Load Ethiopia shapefile
    eth_shp = load_eth_shapefile(args)

    # Load city-region shapefile
    regions_shp = load_regions_shapefile(args)

    # Extract city-region names from shapefile
    cities = [i + ' [GWh]' for i in list(regions_shp['CityName'])]

    # Set the CHIRPS directory name for the year in question
    chirps_dirname = os.path.join(args.chirps_dir, str(args.year))

    # Sort the images so that they're in chronological order
    chirps_list = sorted(glob.glob(chirps_dirname + '/*.tif'))

    # Create dataframe for exporting results
    start_date = datetime(2018, 1, 1)
    date_list = list(rrule(freq=DAILY, dtstart= start_date, count = 365))

    exports_df = pd.DataFrame(index = [i.strftime('%m/%d/%Y') for i in date_list], columns=cities)

    # Loop through all the CHIRPS files in the directory
    for i, chirps_filename in enumerate(chirps_list):
        # Print what number file you're currently on -- sanity check
        print(i)

        # Reproject the CHIRPS tif
        reproject_chirps_rainfall(args, chirps_filename, thresholded_irrig_filename, eth_shp)

        # Calculate city-regional energy deficit
        regional_energy_deficit = calculate_energy_deficit(args, thresholded_irrig_filename, regions_shp)

        # Store energy deficit calculations for export
        exports_df.iloc[i, :] = regional_energy_deficit

        # Clean up tmp dir
        tmp_files = glob.glob(args.tmp_dir + '/*')
        for f in tmp_files:
            os.remove(f)

    # Create filename for results export
    out_file = 'irrig_elec_results_year_{}_irriglb_{}_fracirrig_{}_h20req_{}mm.csv'.format(args.year,
                                                                            args.irrigibility_lb,
                                                                            args.frac_irrig,
                                                                            args.h20_req)
    outpath = os.path.join(args.elec_results_dir, out_file)

    # Export results!
    exports_df.to_csv(outpath)

