import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import rc

from collections import Counter as counter
from datetime import datetime

# Configure parquet engine to handle PyArrow issues
def configure_parquet_engine():
    """Configure the best available parquet engine."""
    try:
        import pyarrow.parquet
        return 'pyarrow'
    except (ImportError, ValueError):
        try:
            import fastparquet
            return 'fastparquet'
        except ImportError:
            os.system("conda install fastparquet -y")
            return 'fastparquet'

PARQUET_ENGINE = configure_parquet_engine()

def read_parquet_robust(file_path):
    """Read parquet file with fallback engine support."""
    try:
        return pd.read_parquet(file_path, engine=PARQUET_ENGINE)
    except Exception as e:
        print(f"Error reading {file_path} with {PARQUET_ENGINE}: {e}")
        alt_engine = 'fastparquet' if PARQUET_ENGINE == 'pyarrow' else 'pyarrow'
        print(f"Trying alternative engine: {alt_engine}")
        return pd.read_parquet(file_path, engine=alt_engine)

# 5_MapReduceInput/AggregatedPlots/1-type_of_trip_hourly.py
def get_simulation_trip_counts(mapped_dir='./Mapped/'):
    """
    Fully replicate the original script's logic:
    - Increment timeslot by 1 for each line and wrap at 144.
    - Detect new-user lines (single field) to reset timeslot and count users.
    - Track runningLocation, runningCoordinates, and work_flag.
    - Increment HBW, HBO, NHB based on state transitions.
    - Record work begin/end timestamps and commuter counts.
    """
    # List only simulationResults_ files
    filenames = [x for x in os.listdir(mapped_dir) if 'simulationResults_' in x]
    hbw = [0] * 24   # Home-to-work counts per hour
    hbo = [0] * 24   # Home-to-other counts per hour
    nhb = [0] * 24   # Non-home/work counts per hour

    numUsers = 0
    numUsersWork = 0
    work_flag = False
    workBeginTimestamps = []
    workEndTimestamps = []

    for filename in filenames:
        path = os.path.join(mapped_dir, filename)
        timeslot = 0
        runningCoordinates = []
        runningLocation = None

        with open(path, 'r') as f:
            for raw in f:
                # 1) Advance and wrap the timeslot
                timeslot = (timeslot + 1) % 144

                parts = raw.strip().split(' ')

                # 2) New user line: only userID present
                if len(parts) == 1:
                    numUsers += 1
                    runningLocation = 'h'
                    timeslot = 0
                    if work_flag:
                        numUsersWork += 1
                        work_flag = False

                # 3) Same location as before
                elif parts[0] == runningLocation:
                    # Initialize runningCoordinates on first visit
                    if not runningCoordinates:
                        runningCoordinates = [parts[1], parts[2]]
                    # If still 'o' but coords changed, count NHB
                    if parts[0] == 'o' and runningCoordinates != [parts[1], parts[2]]:
                        nhb[(timeslot - 1) // 6] += 1
                    runningCoordinates = [parts[1], parts[2]]

                # 4) Transition to 'h' (home)
                elif parts[0] == 'h':
                    if runningLocation == 'o':
                        hbo[(timeslot - 1) // 6] += 1
                    elif runningLocation == 'w':
                        work_flag = True
                        hbw[(timeslot - 1) // 6] += 1
                        workEndTimestamps.append(timeslot / 6.0)

                # 5) Transition to 'w' (work)
                elif parts[0] == 'w':
                    work_flag = True
                    workBeginTimestamps.append(timeslot / 6.0)
                    if runningLocation == 'o':
                        nhb[(timeslot - 1) // 6] += 1
                    elif runningLocation == 'h':
                        hbw[(timeslot - 1) // 6] += 1

                # 6) Transition to 'o' (other)
                elif parts[0] == 'o':
                    if runningLocation == 'h':
                        hbo[(timeslot - 1) // 6] += 1
                    elif runningLocation == 'w':
                        work_flag = True
                        workEndTimestamps.append(timeslot / 6.0)
                        nhb[(timeslot - 1) // 6] += 1

                else:
                    # Unexpected identifier
                    print('Unidentified:', parts[0])

                # Update runningLocation for next iteration
                runningLocation = parts[0]

    total = [hbw[i] + hbo[i] + nhb[i] for i in range(24)]
    return hbw, hbo, nhb, total, numUsers, numUsersWork


def plot_hourly_trip_counts(
    mapped_dir='./Mapped/',
    output='./1-HourlyTripCount.png'
):
    """
    Plot hourly trip counts from simulation results.
    """
    hbw, hbo, nhb, total, numUsers, numUsersWork = \
        get_simulation_trip_counts(mapped_dir)

    # Print counts exactly as in the original script
    print('HBW = ' + str(hbw))
    print('HBO = ' + str(hbo))
    print('NHB = ' + str(nhb))
    print('Total = ' + str(total))
    print('Number of users : ' + str(numUsers))
    print('Number of commuter users : ' + str(numUsersWork))
    print('Number of noncommuter users : ' +
          str(numUsers - numUsersWork))

    # Plot each series with the same colors and marker style
    plt.figure()
    plt.plot(hbw, marker='o', color='b', label='HBW')
    plt.plot(hbo, marker='o', color='g', label='HBO')
    plt.plot(nhb, marker='o', color='r', label='NHB')
    plt.plot(total, marker='o', color='k', label='All')
    plt.legend(loc='upper right')
    plt.xlim(0, 24)
    plt.xlabel('Time of day')
    plt.ylabel('Number of trips')
    plt.savefig(output)
    plt.close()


def plot_dept_validation(
    mapped_dir='./results/Simulation/Mapped/',
    nhts_file='./data/NHTSDep.txt',
    mts_file='./data/NHTSDep.txt',
    output='./results/figs/FigS_dept_validation.png',
    fsize=40,
    msize=20,
    alpha_=0.7,
    lw=3,
    figsize=(24, 16)
):
    """
    Plot department-validation comparison subplots (HBW, HBO, NHB, All) against NHTS/MTS baselines.
    """
    # Retrieve simulation counts - handle the return values properly
    result = get_simulation_trip_counts(mapped_dir)
    if len(result) >= 6:
        hbw, hbo, nhb, total, numUsers, numUsersWork = result
    elif len(result) >= 4:
        hbw, hbo, nhb, total = result[:4]
    else:
        print(f"Unexpected return from get_simulation_trip_counts: {len(result)} values")
        return

    # Check if validation data files exist
    if not os.path.exists(nhts_file) or not os.path.exists(mts_file):
        print(f"Validation data files not found. Creating simple simulation plot instead.")
        # Create simple plot with just simulation data
        plt.figure(figsize=(10, 6))
        hours = list(range(24))
        plt.plot(hours, hbw, 'b-o', label='HBW', linewidth=2, markersize=6)
        plt.plot(hours, hbo, 'g-o', label='HBO', linewidth=2, markersize=6)
        plt.plot(hours, nhb, 'r-o', label='NHB', linewidth=2, markersize=6)
        plt.plot(hours, total, 'k-o', label='Total', linewidth=2, markersize=6)
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Trips')
        plt.title('Hourly Trip Distribution - TimeGeo Simulation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 23)
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Simple simulation plot saved to: {output}")
        return

    # Load NHTS departure data (comma-separated)
    df1_hw = pd.read_csv(nhts_file, sep=',', header=None)
    df1_hw.columns = ['hour', 'hbw', 'hbo', 'nhb', 'hbw_we', 'hbo_we', 'nhb_we']
    df1_hw.drop(['hbw_we', 'hbo_we', 'nhb_we'], axis=1, inplace=True)
    df1_hw['all'] = df1_hw['hbw'] + df1_hw['hbo'] + df1_hw['nhb']

    # Load MTS departure data (space-separated)
    df2_hw = pd.read_csv(mts_file, sep=' ', header=None)
    df2_hw.columns = ['hour', 'hbw_b', 'hbo_b', 'nhb_b', 'all_b']

    # Merge baselines on hour
    df_hw = pd.merge(df1_hw, df2_hw, on='hour', how='inner')

    # Add simulation results to DataFrame
    df_hw['hbw_tg'] = pd.Series(hbw, index=df_hw.index)
    df_hw['hbo_tg'] = pd.Series(hbo, index=df_hw.index)
    df_hw['nhb_tg'] = pd.Series(nhb, index=df_hw.index)
    df_hw['all_tg'] = pd.Series(total, index=df_hw.index)

    # Normalize each series
    for col in ['hbw', 'hbo', 'nhb', 'all']:
        df_hw[f'{col}_r'] = df_hw[col] / df_hw[col].sum()
    for col in ['hbw_b', 'hbo_b', 'nhb_b', 'all_b']:
        df_hw[f'{col}_r'] = df_hw[col] / df_hw[col].sum()
    for col in ['hbw_tg', 'hbo_tg', 'nhb_tg', 'all_tg']:
        df_hw[f'{col}_r'] = df_hw[col] / df_hw[col].sum()

    # Styling for plots
    sns.set_style('ticks')
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    # Create 2x2 subplots
    fig = plt.figure(figsize=figsize)
    title_ = ['HBW', 'HBO', 'NHB', 'All']
    label_ = ['2009 NHTS data', '2010 MTS data', 'TimeGeo Simulation']

    for i in range(4):
        X = df_hw['hour'].tolist()
        Y0 = df_hw.iloc[:, i + 13].tolist()
        Y1 = df_hw.iloc[:, i + 17].tolist()
        Y2 = df_hw.iloc[:, i + 21].tolist()

        plt.subplot(2, 2, i + 1)
        plt.plot(X, Y0, 'o-', markersize=msize, linewidth=lw, alpha=alpha_, label=label_[0])
        plt.plot(X, Y1, 's-', markersize=msize, linewidth=lw, alpha=alpha_, label=label_[1])
        plt.plot(X, Y2, '^-', markersize=msize, linewidth=lw, alpha=alpha_, label=label_[2])
        
        plt.title(title_[i], fontsize=fsize)
        plt.xlabel('Time of day', fontsize=fsize)
        plt.ylabel('Fraction of trips', fontsize=fsize)
        plt.legend(fontsize=fsize - 10)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()


def load_simulation_cdf_parquet(parquet_file):
    """
    Load simulation results from parquet file and prepare CDF data.
    """
    df = read_parquet_robust(parquet_file)
    
    # Handle empty dataframe
    if df.empty:
        print("⚠️  Warning: Simulation results are empty - returning empty CDF data")
        return [], []
    
    # Check if required columns exist
    required_cols = ['user_id', 'timeslot']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Warning: Missing columns {missing_cols} in simulation data - returning empty CDF data")
        return [], []
        
    df_sorted = df.sort_values(['user_id', 'timeslot'])
    
    # Calculate stay durations
    stay_durations = []
    for user_id in df_sorted['user_id'].unique():
        user_data = df_sorted[df_sorted['user_id'] == user_id]
        for i in range(len(user_data) - 1):
            duration = user_data.iloc[i+1]['timeslot'] - user_data.iloc[i]['timeslot']
            if duration > 0:  # Valid duration
                stay_durations.append(duration)
    
    if not stay_durations:
        print("⚠️  Warning: No valid stay durations found - returning empty CDF data")
        return [], []
    
    # Calculate CDF
    stay_durations.sort()
    n = len(stay_durations)
    cdf_values = [i/n for i in range(1, n+1)]
    
    return stay_durations, cdf_values


def load_cdr_cdf_parquet(parquet_file):
    """
    Load CDR data and calculate stay duration CDF.
    """
    df = read_parquet_robust(parquet_file)
    
    # Handle empty dataframe
    if df.empty:
        print("⚠️  Warning: CDR data is empty - returning empty CDF data")
        return [], []
    
    # Check if required columns exist
    required_cols = ['caid', 'timestamp']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Warning: Missing columns {missing_cols} in CDR data - returning empty CDF data")
        return [], []
    
    # Sort by user and timestamp
    df_sorted = df.sort_values(['caid', 'timestamp'])
    
    # Calculate stay durations
    stay_durations = []
    for user_id in df_sorted['caid'].unique():
        user_data = df_sorted[df_sorted['caid'] == user_id]
        if len(user_data) < 2:
            continue
            
        for i in range(len(user_data) - 1):
            duration = user_data.iloc[i+1]['timestamp'] - user_data.iloc[i]['timestamp']
            if duration > 0:  # Valid duration
                # Convert from seconds to 10-minute slots to match simulation
                duration_slots = duration / 600  # 600 seconds = 10 minutes
                stay_durations.append(duration_slots)
    
    if not stay_durations:
        print("⚠️  Warning: No valid stay durations found in CDR data - returning empty CDF data")
        return [], []
    
    # Calculate CDF
    stay_durations.sort()
    n = len(stay_durations)
    cdf_values = [i/n for i in range(1, n+1)]
    
    return stay_durations, cdf_values


def plot_stay_durations_parquet(
    sim_parquet='./results/Simulation/simulation_results.parquet',
    cdr_parquet='./results/SRFiltered_to_SimInput/FAUsers_Cleaned_Formatted.parquet',
    output_file='./results/figs/2-StayDuration_All.png',
    xlim=(0, 24), ylim=(0.0001, 1), figsize=(4, 3), bin_width=0.25
):
    """
    Plot simulation and CDR stay duration PDFs on a log-scaled Y axis.
    Durations are in hours.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # Helper to compute durations in hours from parquet
    def get_durations_in_hours(df, user_col, time_col):
        durations = []
        df_sorted = df.sort_values([user_col, time_col])
        for user_id in df_sorted[user_col].unique():
            user_data = df_sorted[df_sorted[user_col] == user_id]
            if len(user_data) < 2:
                continue
            
            # Track location changes (similar to original load_simulation_cdf)
            running_location = None
            trip_start_slot = None
            
            for _, row in user_data.iterrows():
                current_location = (row['longitude'], row['latitude'])
                current_slot = row[time_col]
                
                if running_location is None:
                    # First location
                    running_location = current_location
                    trip_start_slot = current_slot
                elif current_location != running_location:
                    # Location change: record stay duration
                    trip_end_slot = current_slot
                    duration = float(trip_end_slot - trip_start_slot) * (10/60)  # Convert slots to hours
                    if duration > 0:
                        durations.append(duration)
                    
                    # Start new stay
                    running_location = current_location
                    trip_start_slot = current_slot
            
        return np.array(durations)

    # Read simulation durations
    sim_df = read_parquet_robust(sim_parquet)
    sim_user_col = 'user_id'
    sim_time_col = 'timeslot' if 'timeslot' in sim_df.columns else 'time_slot'
    sim_durations = get_durations_in_hours(sim_df, sim_user_col, sim_time_col)

    # Read CDR durations
    cdr_df = read_parquet_robust(cdr_parquet)
    cdr_user_col = 'caid'
    cdr_time_col = 'timestamp'
    
    # For CDR data, calculate durations between consecutive events (original logic)
    cdr_durations = []
    cdr_sorted = cdr_df.sort_values([cdr_user_col, cdr_time_col])
    
    for user_id in cdr_sorted[cdr_user_col].unique():
        user_data = cdr_sorted[cdr_sorted[cdr_user_col] == user_id]
        if len(user_data) < 2:
            continue
            
        # Calculate durations between consecutive events
        times = user_data[cdr_time_col].values
        diffs = np.diff(times)
        # Convert from seconds to hours
        diffs_hours = diffs / 3600.0
        cdr_durations.extend(diffs_hours[diffs_hours > 0])
    
    cdr_durations = np.array(cdr_durations)

    # Bin edges for histogram (in hours)
    bins = np.arange(xlim[0], xlim[1] + bin_width, bin_width)

    # Compute PDFs
    sim_hist, bin_edges = np.histogram(sim_durations, bins=bins, density=True)
    cdr_hist, _ = np.histogram(cdr_durations, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yscale('log')
    plt.xlim(*xlim)

    # Scatter CDR data
    ax.scatter(bin_centers, cdr_hist, color='g', marker='o', facecolor='white', label='Observed', s=20)
    # Plot simulation PDF
    ax.plot(bin_centers, sim_hist, color='g', label='Simulated')

    # Output durations to console
    print("Stay Durations (Simulated):", bin_centers, sim_hist)
    print("Stay Durations (Observed):", bin_centers, cdr_hist)

    plt.ylim(*ylim)
    plt.xlabel('Duration of Stay (h)')
    plt.ylabel('PDF')
    ax.legend(loc='upper right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.close()
    print(f"Stay duration PDF plot saved to: {output_file}")


def analyze_mobility_patterns_parquet(
    sim_parquet='./results/Simulation/simulation_results.parquet',
    cdr_parquet='./results/SRFiltered_to_SimInput/FAUsers_Cleaned_Formatted.parquet',
    output_dir='./results/figs/'
):
    """
    Comprehensive mobility pattern analysis from parquet files including:
    - Trip distance distribution 
    - Daily visited locations distribution
    - Location rank frequency analysis
    """
    print("Running comprehensive mobility pattern analysis...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all missing analysis plots
    try:
        # 1. Trip distance analysis
        print("Generating trip distance analysis...")
        plot_trip_distance_parquet(sim_parquet, cdr_parquet, 
                                 os.path.join(output_dir, '3-TripDistance.png'))
        
        # 2. Daily visited locations
        print("Generating daily visited locations analysis...")
        plot_daily_visited_locations_parquet(sim_parquet, cdr_parquet,
                                           os.path.join(output_dir, '4-numVisitedLocations.png'))
        
        # 3. Location rank frequency
        print("Generating location rank analysis...")
        plot_location_rank_parquet(sim_parquet, cdr_parquet,
                                 os.path.join(output_dir, '5-LocationRank-User1.png'))
        
        print("Mobility pattern analysis completed successfully!")
        
    except Exception as e:
        print(f"Error in mobility pattern analysis: {e}")
        import traceback
        traceback.print_exc()


def haversine(lon1, lat1, lon2, lat2):
    """Calculate haversine distance between two points."""
    from math import radians, cos, sin, asin, sqrt
    
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km


def parse_simulation_results_parquet(sim_parquet):
    """Parse simulation results from parquet and compute daily location counts and trip distances."""
    print("Parsing simulation results from parquet...")
    
    try:
        df = read_parquet_robust(sim_parquet)
        print(f"Loaded {len(df)} simulation records")
        
        if len(df) == 0:
            print("No simulation data found!")
            return {}, {}
        
        # Check column names and fix if needed
        print(f"Available columns: {df.columns.tolist()}")
        
        # Handle different possible column names
        time_col = None
        if 'time_slot' in df.columns:
            time_col = 'time_slot'
        elif 'timeslot' in df.columns:
            time_col = 'timeslot'
        else:
            print("Error: No time column found in simulation data")
            return {}, {}
            
        # Group by user
        usersDailyLocationCount = {}
        usersTripDistances = {}
        
        # Process each user
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id].sort_values(time_col)
            
            # Calculate daily location counts
            # Group by day (assuming 144 slots per day)
            user_data['day'] = user_data[time_col] // 144
            daily_locations = []
            
            for day in user_data['day'].unique():
                day_data = user_data[user_data['day'] == day]
                unique_locations = set()
                for _, row in day_data.iterrows():
                    unique_locations.add((row['longitude'], row['latitude']))
                daily_locations.append(len(unique_locations))
            
            usersDailyLocationCount[user_id] = daily_locations
            
            # Calculate trip distances
            trip_distances = []
            prev_coords = None
            
            for _, row in user_data.iterrows():
                curr_coords = (row['longitude'], row['latitude'])
                if prev_coords is not None and prev_coords != curr_coords:
                    distance = haversine(prev_coords[0], prev_coords[1], 
                                       curr_coords[0], curr_coords[1])
                    if distance > 0:  # Only count actual trips
                        trip_distances.append(distance)
                prev_coords = curr_coords
            
            usersTripDistances[user_id] = trip_distances
        
        print(f"Processed {len(usersDailyLocationCount)} users for daily location counts")
        print(f"Processed {len(usersTripDistances)} users for trip distances")
        
        return usersDailyLocationCount, usersTripDistances
        
    except Exception as e:
        print(f"Error parsing simulation results: {e}")
        import traceback
        traceback.print_exc()
        return {}, {}


def parse_observed_data_parquet(cdr_parquet):
    """Parse observed CDR data from parquet and compute location counts and trip distances."""
    print("Parsing observed data from parquet...")
    
    try:
        df = read_parquet_robust(cdr_parquet)
        print(f"Loaded {len(df)} CDR records")
        
        if len(df) == 0:
            print("No CDR data found!")
            return [], []
        
        # Take first user for comparison (as in original)
        first_user = df['caid'].iloc[0]
        user_data = df[df['caid'] == first_user].sort_values('timestamp')
        
        # Calculate daily location counts
        user_data['date'] = pd.to_datetime(user_data['timestamp'], unit='s').dt.date
        daily_locations = []
        
        for date in user_data['date'].unique():
            day_data = user_data[user_data['date'] == date]
            unique_locations = set()
            for _, row in day_data.iterrows():
                unique_locations.add((row['Longitude'], row['Latitude']))
            daily_locations.append(len(unique_locations))
        
        # Calculate trip distances
        trip_distances = []
        prev_coords = None
        
        for _, row in user_data.iterrows():
            curr_coords = (row['Longitude'], row['Latitude'])
            if prev_coords is not None and prev_coords != curr_coords:
                distance = haversine(prev_coords[0], prev_coords[1], 
                                   curr_coords[0], curr_coords[1])
                if distance > 0:
                    trip_distances.append(distance)
            prev_coords = curr_coords
        
        print(f"Observed data: {len(daily_locations)} days, {len(trip_distances)} trips")
        
        return daily_locations, trip_distances
        
    except Exception as e:
        print(f"Error parsing observed data: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def plot_trip_distance_parquet(sim_parquet, cdr_parquet, output_file):
    """Plot trip distance distributions for observed and simulated data."""
    print("Generating trip distance plot...")
    
    try:
        # Parse data
        usersDailyLocationCount, usersTripDistances = parse_simulation_results_parquet(sim_parquet)
        user1ObsLocationCount, user1ObsTripDistances = parse_observed_data_parquet(cdr_parquet)
        
        if not usersTripDistances:
            print("No simulation trip distance data available")
            return
            
        if not user1ObsTripDistances:
            print("No observed trip distance data available") 
            return
        
        # Process observed trip distances
        from math import ceil
        user1ObsTripDistancesCeil = [ceil(x) for x in user1ObsTripDistances]
        u1_obsDist = counter(user1ObsTripDistancesCeil)
        total = sum(u1_obsDist.values())
        for k in u1_obsDist:
            u1_obsDist[k] = float(u1_obsDist[k]) / total

        # Process simulated trip distances
        simDistList = []
        for v in usersTripDistances.values():
            simDistList.extend(v)
        simDistList = [ceil(x) for x in simDistList]
        u1_simDist = counter(simDistList)
        total = sum(u1_simDist.values())
        for k in u1_simDist:
            u1_simDist[k] = float(u1_simDist[k]) / total
        
        if u1_simDist:
            m = max(u1_simDist.keys())
            u1_simDist[m+5] = 0

        # Create plot
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_yscale('log')
        ax.set_xscale('log')

        # Plot observed data (green circles with white fill)
        if u1_obsDist:
            ax.scatter(
                u1_obsDist.keys(), 
                u1_obsDist.values(),
                color='g',
                marker='o',
                facecolor='white',
                s=20,
                label='Observed'
            )
        
        # Plot simulated data (green line)
        if u1_simDist:
            ax.plot(
                sorted(u1_simDist.keys()),
                [u1_simDist[k] for k in sorted(u1_simDist.keys())],
                color='g',
                label='Simulated'
            )

        ax.legend()
        plt.xlim(0.9, 100)
        plt.ylim(0.0001, 1)
        plt.xlabel('Trip Distance, r (km)')
        plt.ylabel('P(r)')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        print(f"Trip distance plot saved to {output_file}")
        
    except Exception as e:
        print(f"Error generating trip distance plot: {e}")
        import traceback
        traceback.print_exc()


def plot_daily_visited_locations_parquet(sim_parquet, cdr_parquet, output_file):
    """Plot the distribution of daily visited locations from observed and simulated data."""
    print("Generating daily visited locations plot...")
    
    try:
        # Parse data
        usersDailyLocationCount, usersTripDistances = parse_simulation_results_parquet(sim_parquet)
        user1ObsLocationCount, user1ObsTripDistances = parse_observed_data_parquet(cdr_parquet)
        
        if not usersDailyLocationCount:
            print("No simulation daily location data available")
            return
            
        if not user1ObsLocationCount:
            print("No observed daily location data available")
            return
        
        # Process observed data
        c1_count = counter(user1ObsLocationCount)
        total = sum(c1_count.values())
        for k in c1_count:
            c1_count[k] = float(c1_count[k]) / total

        # Process simulated data - flatten all users' daily counts
        usersDailyLocationCountFlat = []
        for v in usersDailyLocationCount.values():
            usersDailyLocationCountFlat.extend(v)

        c1_simCount = counter(usersDailyLocationCountFlat)
        total = sum(c1_simCount.values())
        for k in c1_simCount:
            c1_simCount[k] = float(c1_simCount[k]) / total

        # Create plot
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_yscale('log')

        # Plot observed data
        if c1_count:
            ax.scatter(
                c1_count.keys(), c1_count.values(), color='g', marker='o',
                facecolor='white', label='Observed', s=20
            )
        
        # Plot simulated data
        if c1_simCount:
            ax.plot(
                sorted(c1_simCount.keys()), [c1_simCount[k] for k in sorted(c1_simCount.keys())],
                color='g', label='Simulated'
            )

        ax.legend()
        plt.xlim(0, 10)
        plt.ylim(0.0005, 1)
        plt.xticks(range(1, 10))
        plt.xlabel('Daily visited locations, N')
        plt.ylabel('P(N)')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        print(f"Daily visited locations plot saved to {output_file}")
        
    except Exception as e:
        print(f"Error generating daily visited locations plot: {e}")
        import traceback
        traceback.print_exc()


def plot_location_rank_parquet(sim_parquet, cdr_parquet, output_file, top_n=50):
    """Plot location rank frequency distribution on log-log scales."""
    print("Generating location rank plot...")
    
    try:
        # Parse simulation data for location frequencies
        df_sim = read_parquet_robust(sim_parquet)
        
        if len(df_sim) == 0:
            print("No simulation data available")
            return
        
        # Calculate location visit frequencies for simulation
        usersLocations = {}
        for user_id in df_sim['user_id'].unique():
            user_data = df_sim[df_sim['user_id'] == user_id]
            locations = []
            for _, row in user_data.iterrows():
                locations.append((row['longitude'], row['latitude']))
            usersLocations[user_id] = locations

        # Convert counts to sorted frequency lists
        for k in usersLocations.keys():
            usersLocations[k] = counter(usersLocations[k])
        for k in usersLocations.keys():
            counts = sorted(usersLocations[k].values(), reverse=True)
            # Ensure list length == top_n
            for i in range(top_n):
                if i >= len(counts):
                    counts.append(0)
            usersLocations[k] = counts[:top_n]

        # Aggregate frequencies and normalize
        cum_probs = [0] * top_n
        for v in usersLocations.values():
            cum_probs = [sum(x) for x in zip(cum_probs, v)]
        total = sum(cum_probs)
        if total > 0:
            cum_probs = [float(x) / total for x in cum_probs]

        # Parse observed data
        df_cdr = read_parquet_robust(cdr_parquet)
        
        if len(df_cdr) == 0:
            print("No CDR data available")
            return
        
        # Process observed data - use first user as representative
        first_user = df_cdr['caid'].iloc[0]
        user_data = df_cdr[df_cdr['caid'] == first_user]
        user1Locations = []
        for _, row in user_data.iterrows():
            user1Locations.append((row['Longitude'], row['Latitude']))
        
        user1LocationCounts = counter(user1Locations)
        counts1 = sorted(user1LocationCounts.values(), reverse=True)
        for i in range(top_n):
            if i >= len(counts1):
                counts1.append(0)
        counts1 = counts1[:top_n]
        
        total1 = sum(counts1)
        if total1 > 0:
            cum_probs1 = [float(x) / total1 for x in counts1]
        else:
            cum_probs1 = [0] * top_n

        # Create plot
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_yscale('log')
        ax.set_xscale('log')
        
        # Plot observed data
        if any(p > 0 for p in cum_probs1):
            ax.scatter(range(1, len(cum_probs1)+1), cum_probs1, color='g', marker='s', 
                      facecolor='white', label='Observed', s=20)
        
        # Plot simulated data
        if any(p > 0 for p in cum_probs):
            ax.plot(range(1, len(cum_probs)+1), cum_probs, color='g', label='Simulated')
        
        ax.legend()
        plt.xlim(1, top_n)
        plt.ylim(0.002, 1)
        plt.xlabel('Lth most visited location')
        plt.ylabel('f(L)')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        print(f"Location rank plot saved to {output_file}")
        
    except Exception as e:
        print(f"Error generating location rank plot: {e}")
        import traceback
        traceback.print_exc() 