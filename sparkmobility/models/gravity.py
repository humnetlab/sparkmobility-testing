import copy
import math

from ..utils.county_tesslation import tesselate_county
import folium
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels as sm
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.generalized_linear_model import GLM
from tqdm import tqdm

tqdm.pandas()


def h3_index_to_centroid(h3_index):
    lat, lng = h3.h3_to_geo(h3_index)
    return Point(lng, lat)


def h3_index_to_centroid(h3_index):
    lat, lng = h3.h3_to_geo(h3_index)
    return Point(lng, lat)


def get_polygon_centroid(polygon):
    if not isinstance(polygon, Polygon):
        raise ValueError("Input must be a Shapely Polygon object.")
    return polygon.centroid


def get_h3_index_at_resolution(h3_index, resolution):
    current_resolution = h3.get_resolution(h3_index)

    if resolution > current_resolution:
        # Convert to a finer resolution
        raise ValueError(
            "Cannot directly get a higher resolution parent. Use `h3.h3_to_children()` for this."
        )
    elif resolution < current_resolution:
        # Convert to a coarser resolution
        return h3.cell_to_parent(h3_index, resolution)
    else:
        return h3_index


def create_normalization_constraint_vector(origin_index, total_locations):
    normalization_vector = list(np.zeros(total_locations))
    normalization_vector[origin_index] = 1.0
    return normalization_vector


def compute_distance_matrix(spatial_tessellation, origins):

    # coordinates = spatial_tessellation.geometry.apply(lambda geom: (geom.y, geom.x)).values if #hasattr(spatial_tessellation, 'geometry') else spatial_tessellation

    # Extract centroid coordinates (latitude and longitude) for each polygon
    if not hasattr(spatial_tessellation, "geometry"):
        raise ValueError(
            "The provided spatial_tessellation must have a 'geometry' column."
        )

    # centroids = spatial_tessellation.geometry.apply(lambda geom: (geom.centroid.y, geom.centroid.x)).values
    centroids = (
        spatial_tessellation["index"]
        .apply(lambda h3_index: h3_index_to_centroid(h3_index))
        .apply(lambda point: (point.y, point.x))
        .values
    )

    n = len(spatial_tessellation)
    distance_matrix = np.zeros((n, n))

    for id_i in tqdm(range(n)):  # origins
        lat_i, lng_i = centroids[id_i]
        for id_j in range(id_i + 1, n):
            lat_j, lng_j = centroids[id_j]
            distance = geodesic(
                (lat_i, lng_i), (lat_j, lng_j)
            ).kilometers  # .kilometers
            distance_matrix[id_i, id_j] = distance
            distance_matrix[id_j, id_i] = distance

    return distance_matrix


def h3_to_polygon(h3_index):
    boundary = h3.cell_to_boundary(
        h3_index
    )  # True indicates the GeoJSON coordinate standard (latitude, longitude) used when returning geo coordinates
    # boundary return [(lat1, lon1), (lat2, lon2), ...]
    # need to convert to [(lon1, lat1), (lon2, lat2), ...]
    polygon_coords = [(coord[1], coord[0]) for coord in boundary]
    return Polygon(polygon_coords)


class Gravity:
    def __init__(
        self,
        deterrence_func_type="power_law",
        deterrence_func_args=[-2.0],
        origin_exp=1.0,
        destination_exp=1.0,
        gravity_type="single",  # 'single' or 'global'
        name="Gravity model",
        is_h3_hexagon=True,
    ):
        """
        Initialize the Gravity model instance.

        Parameters:
        - deterrence_func_type (str): Type of deterrence function ('power_law' or 'exponential')
        - deterrence_func_args (list): Parameters for the deterrence function
        - origin_exp (float): Exponent for origin (only for globally constrained models)
        - destination_exp (float): Exponent for destination
        - gravity_type (str): Type of gravity model ('single' or 'global')
        - name (str): Name of the gravity model instance
        """
        # Initialize gravity model parameters
        self.deterrence_func_type = deterrence_func_type
        self.deterrence_func_args = deterrence_func_args
        self.origin_exp = origin_exp
        self.destination_exp = destination_exp
        self.gravity_type = gravity_type
        self.name = name
        self.is_h3_hexagon = is_h3_hexagon

        # Set the deterrence function based on the type
        if self.deterrence_func_type == "power_law":
            exponent = self.deterrence_func_args[0]
            self.deterrence_function = lambda d: d**exponent
        elif self.deterrence_func_type == "exponential":
            coefficient = self.deterrence_func_args[0]
            self.deterrence_function = lambda d: np.exp(-coefficient * d)
        else:
            raise ValueError(
                f"Invalid deterrence function type: {self.deterrence_func_type}"
            )
            # exponent = self.deterrence_func_args[0]
            # self.deterrence_function = lambda d: d ** exponent

    def __str__(self):
        return (
            'Gravity(name="%s", deterrence_func_type="%s", '
            'deterrence_func_args=%s, origin_exp=%s, destination_exp=%s, gravity_type="%s")'
            % (
                self.name,
                self.deterrence_func_type,
                self.deterrence_func_args,
                self.origin_exp,
                self.destination_exp,
                self.gravity_type,
            )
        )

    def get_str(self, *args):
        values = [getattr(self, arg, None) for arg in args]
        return values

    def get_gravity_score(self, distance_matrix, relevances_orig, relevances_dest):

        trip_probs_matrix = self.deterrence_function(distance_matrix)

        # relevances_dest_weights = relevances_dest ** self.destination_exp
        # relevances_orig_weights = np.expand_dims(relevances_orig ** self._origin_exp, #axis=1)
        # trip_probs_matrix = trip_probs_matrix * relevances_dest_weights * #relevances_orig_weights

        relevances_dest_adj = np.power(relevances_dest, self.destination_exp)
        relevances_orig_adj = np.power(relevances_orig, self.origin_exp)

        trip_probs_matrix *= relevances_dest_adj
        trip_probs_matrix = (trip_probs_matrix.T * relevances_orig_adj).T

        trip_probs_matrix = np.nan_to_num(
            trip_probs_matrix, nan=0.0, posinf=0.0, neginf=0.0
        )

        np.fill_diagonal(trip_probs_matrix, 0.0)

        return trip_probs_matrix

    def from_matrix_to_flowdf(self, flow_matrix, origins, spatial_tessellation):
        """
        Converts an origin-destination flow matrix into a FlowDataFrame format.

        Parameters:
            flow_matrix (2D array): Origin-to-destination flow matrix.
            origins (1D array): List of origin indices.
            spatial_tessellation (GeoDataFrame): GeoDataFrame containing spatial tessellation.

        Returns:
            FlowDataFrame: A DataFrame containing origin, destination, and corresponding flow.
        """
        # Create index-to-location mapping
        index2tileid = {
            index: spatial_tessellation.iloc[index]["index"]
            for index in range(len(spatial_tessellation))
        }
        # print('index2tileid', index2tileid)
        # print('origins', origins)

        # Generate flow data list
        flow_data = []
        # for i, origin_idx in enumerate(origins):
        for origin_idx in origins:
            for dest_idx, flow in enumerate(flow_matrix[origin_idx]):
                if flow > 0 and origin_idx != dest_idx:
                    origin_id = index2tileid.get(origin_idx)
                    dest_id = index2tileid.get(dest_idx)
                    flow_data.append(
                        {"origin": origin_id, "destination": dest_id, "flow": flow}
                    )
        flow_df = pd.DataFrame(flow_data)
        return flow_df

    def generate(
        self,
        spatial_tessellation,
        tile_id_column,  # =constants.TILE_ID,
        relevance_column,  # =constants.RELEVANCE,
        tot_outflows_column="total_outflow",  # =constants.TOT_OUTFLOW,
        out_format="flows",  # 'flows', 'flows_sample', 'probabilities'
    ):
        """
        Generate flow data based on the gravity model and the given spatial tessellation.

        Parameters:
        - spatial_tessellation (GeoDataFrame): Spatial tessellation containing location information.
        - tile_id_column (str): Column name for location identifiers.
        - tot_outflows_column (str): Column name for total outflows.
        - relevance_column (str): Column name for location relevances.
        - out_format (str): Output format ('flows', 'flows_sample', 'probabilities').

        Returns:
        - FlowDataFrame: Generated flow data.
        """
        # 1. Check output format
        valid_formats = ["flows", "flows_sample", "probabilities"]
        if out_format not in valid_formats:
            raise ValueError(
                f"Invalid out_format '{out_format}'. Valid options are {valid_formats}."
            )

        self._tile_id_column = tile_id_column
        # 2. Extract necessary data
        relevances = spatial_tessellation[relevance_column].fillna(0).values
        if out_format in ["flows", "flows_sample"]:
            tot_outflows = spatial_tessellation[tot_outflows_column].fillna(0).values

        origins = np.arange(len(spatial_tessellation))
        # origins = spatial_tessellation[tile_id_column].values

        # 3. Compute distance matrix
        distance_matrix = compute_distance_matrix(
            spatial_tessellation, origins
        )  # tile_id_column=tile_id_column

        # 4. Compute score matrix
        trip_probs_matrix = self.get_gravity_score(
            distance_matrix, relevances, relevances
        )

        # 5. Process according to model type
        if self.gravity_type == "global":
            # Normalize trip_probs_matrix so that the sum is 1
            trip_probs_matrix /= np.sum(trip_probs_matrix)
            # trip_probs_matrix = trip_probs_matrix / trip_probs_matrix.sum(axis=1, keepdims=True)
            trip_probs_matrix = np.nan_to_num(
                trip_probs_matrix, nan=0.0, posinf=0.0, neginf=0.0
            )

            if out_format == "probabilities":
                flow_df = self.from_matrix_to_flowdf(
                    trip_probs_matrix, origins, spatial_tessellation
                )
                return flow_df

            elif out_format == "flows":
                # total_flow = tot_outflows.sum()
                # avg_flows_matrix = trip_probs_matrix * total_flow

                od_matrix = (trip_probs_matrix.T * tot_outflows).T

                flow_df = self.from_matrix_to_flowdf(
                    od_matrix, origins, spatial_tessellation
                )
                return flow_df

            elif out_format == "flows_sample":
                total_flow = int(tot_outflows.sum())
                flow_probs_flat = trip_probs_matrix.flatten()
                flows_sampled = np.random.multinomial(total_flow, flow_probs_flat)
                flows_sampled_matrix = flows_sampled.reshape(trip_probs_matrix.shape)
                flow_df = self.from_matrix_to_flowdf(
                    flows_sampled_matrix, origins, spatial_tessellation
                )
                return flow_df

        elif self.gravity_type == "single":
            # Normalize trip_probs_matrix for each origin
            trip_probs_matrix = np.transpose(
                trip_probs_matrix / np.sum(trip_probs_matrix, axis=0)
            )
            # trip_probs_matrix = trip_probs_matrix / trip_probs_matrix.sum(axis=1, keepdims=True)
            # Handle NaN and Inf values
            trip_probs_matrix = np.nan_to_num(
                trip_probs_matrix, nan=0.0, posinf=0.0, neginf=0.0
            )

            if out_format == "probabilities":
                flow_df = self.from_matrix_to_flowdf(
                    trip_probs_matrix, origins, spatial_tessellation
                )
                return flow_df

            elif out_format == "flows":
                avg_flows_matrix = trip_probs_matrix * tot_outflows[:, np.newaxis]
                print(avg_flows_matrix)
                flow_df = self.from_matrix_to_flowdf(
                    avg_flows_matrix, origins, spatial_tessellation
                )
                return flow_df

            elif out_format == "flows_sample":
                flows_sampled_matrix = np.zeros(trip_probs_matrix.shape)
                for i, (probs, N) in enumerate(zip(trip_probs_matrix, tot_outflows)):
                    flows_sampled_matrix[i, :] = np.random.multinomial(int(N), probs)
                flow_df = self.from_matrix_to_flowdf(
                    flows_sampled_matrix, origins, spatial_tessellation
                )
                return flow_df

        else:
            raise ValueError(
                f"Invalid model_type '{self.gravity_type}'. Expected 'global' or 'single'."
            )

    def fit(
        self,
        flow_df,
        relevance_df,  # relevance_df, with geometry info and relecance column
        relevance_column,
    ):
        """
        Fit the gravity model using observed flow data.

        Parameters
        ----------
        flow_df : FlowDataFrame
            The observed flow data.
        relevance_column : str, optional
            The column name for location relevances (weights).
            Default is constants.RELEVANCE.

        """
        if self.is_h3_hexagon:
            unique_h3_index = relevance_df["index"].unique()
            h3center_mapping = {
                h3_index: h3_index_to_centroid(h3_index) for h3_index in unique_h3_index
            }
            relevance_df["center"] = relevance_df["index"].map(h3center_mapping)

        # self.lats_lngs = flow_df.tessellation.geometry.apply(utils.get_geom_centroid, args=[True]).values
        self.weights = relevance_df[relevance_column].fillna(0).values
        self.tileid2index = dict(
            [(tileid, i) for i, tileid in enumerate(relevance_df["index"].values)]
        )

        self.X, self.y = [], []  # independent (X) and dependent (y) variables
        flow_df.progress_apply(
            lambda flow_example: self.update_training_set(flow_example), axis=1
        )
        # flow_df.apply(lambda flow_example: self.update_training_set(flow_example), axis=1)

        # pd.DataFrame(self.X).to_csv('selfX.csv')
        # pd.DataFrame(self.y).to_csv('selfY.csv')

        # 3. Model fitting
        # Poisson Model
        print("Fitting GLM Model...")
        # print(np.array(self.X).shape)
        # print(pd.Series(self.y).describe())

        # Standardize
        # scaler = StandardScaler()
        # self.X = scaler.fit_transform(self.X)

        model = GLM(
            self.y,
            self.X,
            family=sm.genmod.families.family.Poisson(
                link=sm.genmod.families.links.Log()
            ),
        )
        result = model.fit(disp=True)

        # 4. Update model parameters
        if self.gravity_type == "global":
            self.origin_exp = result.params[1]
            self.destination_exp = result.params[2]
            self.deterrence_func_args = [result.params[3]]
        elif self.gravity_type == "single":
            self.origin_exp = 1.0
            self.destination_exp = result.params[-2]
            self.deterrence_func_args = [result.params[-1]]

        # Clean up temporary variables
        del self.X
        del self.y

    def update_training_set(self, flow_example):
        """
        Update the training set for fitting the gravity model parameters.

        Parameters
        ----------
        flow_example : pd.Series
            A single flow instance containing origin, destination, and flow values.
        x_coords : pd.Series
            X-coordinates of the origins and destinations.
        y_coords : pd.Series
            Y-coordinates of the origins and destinations.
        relevances : pd.Series
            Relevance (weights) of the origins and destinations.

        """
        # Extract flow instance information
        origin_id = flow_example["origin"]
        destination_id = flow_example["destination"]
        flow_value = flow_example["flow"]

        # Exclude self-loops
        if origin_id == destination_id:
            return  # Skip self-loops (flow from a location to itself)

        if self.is_h3_hexagon:
            distance = flow_example["distance"]
        # else:

        try:
            origin_relevance = self.weights[self.tileid2index[origin_id]]
            destination_relevance = self.weights[self.tileid2index[destination_id]]
        except KeyError:
            print("Missing infor for location " + '"%s" Skipping ...' % origin_id)
            return

        if origin_relevance <= 0 or destination_relevance <= 0:
            return
        if distance <= 0:
            return

        if self.gravity_type == "global":
            sc_vars = [np.log(origin_relevance)]
        elif self.gravity_type == "single":
            sc_vars = create_normalization_constraint_vector(
                self.tileid2index[origin_id], len(self.tileid2index)
            )

        if self.deterrence_func_type == "exponential":
            # exponential deterrence function
            self.X += [[1.0] + sc_vars + [np.log(destination_relevance), -distance]]
        elif self.deterrence_func_type == "power_law":
            # power law deterrence function
            self.X += [
                [1.0] + sc_vars + [np.log(destination_relevance), np.log(distance)]
            ]

        self.y += [float(flow_value)]


# for generating
def get_outflow_hex_df(
    outflow_df_path, name_outflow_origin, name_outflow_outflow, hex_resolution, pop_df
):

    flow_hex_df = pd.read_csv(outflow_df_path)
    flow_hex_df = flow_hex_df.rename(
        columns={name_outflow_origin: "origin", name_outflow_outflow: "total_outflow"}
    )
    # Extract unique H3 indices from origin and destination
    unique_h3_indices = flow_hex_df["origin"].unique()
    # Map original H3 indices to the desired resolution
    resolution_mapping = {
        h3_index: get_h3_index_at_resolution(h3_index, hex_resolution)
        for h3_index in unique_h3_indices
    }
    # Update origin and destination columns with new resolution
    flow_hex_df["origin"] = flow_hex_df["origin"].map(resolution_mapping)

    # Obtain outflow data
    outflow_hex_df = flow_hex_df.groupby("origin", as_index=False).agg(
        {"total_outflow": "sum"}
    )

    # Filtering according to county code
    valid_indices = set(pop_df["index"])
    filtered_outflow_df = outflow_hex_df[outflow_hex_df["origin"].isin(valid_indices)]
    outflow_hex_df_with_pop = pd.merge(
        filtered_outflow_df, pop_df, left_on="origin", right_on="index", how="left"
    )
    return outflow_hex_df_with_pop


# for fitting
def get_actual_flow_hex_df(
    actual_flow_df_path,
    name_caid,
    name_origin,
    name_destination,
    name_distance,
    hex_resolution,
    pop_df=None,
):

    flow_hex_df = pd.read_parquet(actual_flow_df_path)
    flow_hex_df = flow_hex_df.rename(
        columns={
            name_caid: "caid",
            name_origin: "origin",
            name_destination: "destination",
            name_distance: "distance",
        }
    )

    # Extract unique H3 indices from origin and destination
    unique_origins = flow_hex_df["origin"].unique()
    unique_destinations = flow_hex_df["destination"].unique()
    unique_h3_indices = set(unique_origins).union(set(unique_destinations))
    # Map original H3 indices to the desired resolution
    resolution_mapping = {
        h3_index: get_h3_index_at_resolution(h3_index, hex_resolution)
        for h3_index in unique_h3_indices
    }
    # Update origin and destination columns with new resolution
    flow_hex_df["origin"] = flow_hex_df["origin"].map(resolution_mapping)
    flow_hex_df["destination"] = flow_hex_df["destination"].map(resolution_mapping)

    # Obtain flow data
    flow_hex_df = flow_hex_df.groupby(["origin", "destination"], as_index=False).agg(
        flow=("caid", "count"), distance=("distance", "first")
    )

    # Drop origin == destination
    flow_hex_df = flow_hex_df[flow_hex_df["origin"] != flow_hex_df["destination"]]

    if pop_df is not None:
        # Filtering according to county code
        valid_indices = set(pop_df["index"])
        filtered_flow_df = flow_hex_df[
            flow_hex_df["origin"].isin(valid_indices)
            & flow_hex_df["destination"].isin(valid_indices)
        ]
        filtered_flow_df.reset_index(drop=True, inplace=True)
        return filtered_flow_df
    elif pop_df is None:
        flow_hex_df.reset_index(drop=True, inplace=True)
        return flow_hex_df


def obtain_population_data(
    state_fips_codes,
    county_fips_codes,
    hex_resolution,
    year,
    census_dataset,
    projection_crs,
    census_variables,
):
    pop_df = gpd.GeoDataFrame()
    for state_fips in state_fips_codes:
        for county_fips in county_fips_codes.get(state_fips, []):

            pop_df_i = tesselate_county(
                state_fips_code=state_fips,
                county_fips_code=county_fips,
                hex_resolution=hex_resolution,
                year=year,
                census_dataset=census_dataset,
                projection_crs=projection_crs,
                census_variables=census_variables,
            )
            # Append the processed DataFrame to the list
            pop_df = pd.concat([pop_df, pop_df_i], ignore_index=True)
            # except Exception as e:
            #     print(
            #         f"Failed to process state {state_fips}, county {county_fips}: {e}"
            #     )

    # Concatenate all the DataFrames
    return pop_df


def obtain_relevance_data_at_resolution(
    relevance_df, colname_h3_index, colname_caid, hex_resolution
):

    relevance_df = relevance_df.rename(columns={colname_h3_index: "index"})

    # Extract unique H3 indices from origin and destination
    unique_h3_indices = relevance_df["index"].unique()
    # Map original H3 indices to the desired resolution
    resolution_mapping = {
        h3_index: get_h3_index_at_resolution(h3_index, hex_resolution)
        for h3_index in unique_h3_indices
    }
    # Update origin and destination columns with new resolution
    relevance_df["index"] = relevance_df["index"].map(resolution_mapping)

    # relevance_df['index'] = relevance_df['index'].map(lambda h3_index: get_h3_index_at_resolution(h3_index, hex_resolution))

    relevance_df = relevance_df.groupby("index", as_index=False).agg(
        stay_count=(colname_caid, "count")
    )

    relevance_df["geometry"] = relevance_df["index"].apply(h3_to_polygon)
    relevance_gdf = gpd.GeoDataFrame(relevance_df, geometry="geometry", crs="EPSG:4326")

    return relevance_gdf
