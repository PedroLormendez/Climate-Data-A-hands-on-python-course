import xarray as xr
import numpy as np


def convert_and_sort_coords(data_array: xr.DataArray):
    """
    Convert longitude from 0-360 to -180 to 180, validate latitude values,
    and sort the DataArray by both longitude and latitude.

    Parameters:
    data_array (xarray.DataArray): The input DataArray with latitude and longitude coordinates.

    Returns:
    xarray.DataArray: The DataArray with converted and sorted coordinates.
    """

    def adjust_lon(lon):
        """Adjust longitude values from 0-360 to -180 to 180."""
        return np.where(lon > 180, lon - 360, lon)

    def validate_lat(lat):
        """Validate latitude values to be within -90 to 90."""
        return np.clip(lat, -90, 90)

    # Adjust longitude
    if 'longitude' in data_array.coords:
        adjusted_lons = adjust_lon(data_array.longitude.values)
        data_array = data_array.assign_coords(longitude=('longitude', adjusted_lons))
    elif 'lon' in data_array.coords:
        adjusted_lons = adjust_lon(data_array.lon.values)
        data_array = data_array.assign_coords(lon=('lon', adjusted_lons))
    else:
        raise ValueError("No 'longitude' or 'lon' coordinates found in the DataArray.")

    # Validate latitude
    if 'latitude' in data_array.coords:
        validated_lats = validate_lat(data_array.latitude.values)
        data_array = data_array.assign_coords(latitude=('latitude', validated_lats))
    elif 'lat' in data_array.coords:
        validated_lats = validate_lat(data_array.lat.values)
        data_array = data_array.assign_coords(lat=('lat', validated_lats))
    else:
        raise ValueError("No 'latitude' or 'lat' coordinates found in the DataArray.")

    # Sort by longitude and latitude
    lon_coord_name = 'longitude' if 'longitude' in data_array.coords else 'lon'
    lat_coord_name = 'latitude' if 'latitude' in data_array.coords else 'lat'
    data_array = data_array.sortby([lon_coord_name, lat_coord_name])

    return data_array