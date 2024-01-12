import numpy as np
import geopandas as gpd
from copy import deepcopy
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon

def Envelope(GPD):
    # Convert each geometry to its envelope
    for i in range(0, len(GPD)):
        GPD.geometry[i] = GPD.geometry[i].envelope
    return GPD

def Mean_NIoU(ARP_filtered, CMM_filtered, top):
    niou_values = []
    for idx1, row1 in ARP_filtered.iterrows():
        for idx2, row2 in CMM_filtered.iterrows():
            # Calculate intersection and union areas
            intersection_area = row1['geometry'].intersection(row2['geometry']).area
            union_area = row1['geometry'].union(row2['geometry']).area

            # Calculate areas of larger and smaller polygons
            area_L = max(row1['geometry'].area, row2['geometry'].area)
            area_S = min(row1['geometry'].area, row2['geometry'].area)

            # Calculate normalized IoU (Intersection over Union)
            niou_values.append((intersection_area / union_area) * (area_L / area_S))

    # Reshape niou_values into a matrix
    niou_values_reshaped = np.reshape(niou_values, [top, top])
    matrix = niou_values_reshaped

    # Select the maximum values in each row 
    # while avoiding repeated column selections
    tar_niou = matrix[0][0]
    selected_values = []

    for row_idx in range(0, matrix.shape[0]):
        row = matrix[row_idx]
        max_value = np.max(row)
        max_col = np.argmax(row)

        while any(col == max_col for _, col in selected_values):
            row[max_col] = -1
            max_value = np.max(row)
            max_col = np.argmax(row)

        selected_values.append((max_value, max_col))

    niou_list = [value[0] for value in selected_values]

    return np.mean(niou_list), tar_niou, niou_list

def tree_matching(wrap_path, ref_path, top=3, dist_max=3.0, k=10):
    '''
    wrap_path (*.shp, *.geojson): the bounding box file to be matched
    ref_path (*.shp, *.geojson): reference bounding box file 
    ** keep wrap and reference files same geographic information and projection
    top (int): the number of reference trees closest to the candidate tree 
    dist_max (float): Maximum possible offset distance over the entire range
    k (int): the number of trees used for searching the 
    '''

    # Load and envelope the geometries
    ARP = Envelope(gpd.read_file(wrap_path))
    CMM = Envelope(gpd.read_file(ref_path))

    ARP['id'] = ARP.index.values
    CMM['id'] = CMM.index.values

    # Create coordinate matrices
    ARP_coords = np.array([list(item) for item in
                           ARP.geometry.centroid.apply(lambda point: (point.x, point.y))])
    CMM_coords = np.array([list(item) for item in
                           CMM.geometry.centroid.apply(lambda point: (point.x, point.y))])

    # Create cKDTree
    ARP_cdtree = cKDTree(ARP_coords)
    CMM_cdtree = cKDTree(CMM_coords)

    stop = len(ARP)
    match_list = []

    # Initial variable declaration
    final_offset = [[0, 0]] * stop
    final_offset_index_list = [[-1, -1]] * stop
    NIoU_mean_list = list(range(stop))
    tar_niou_list = list(range(stop))
    niou_list_list = [[]] * stop
    ARP_rectified = deepcopy(ARP[:stop])

    print(f"top: {top}")

    for c, ARP_coord in enumerate(ARP_coords):
        if c != stop:
            print(f"preliminary matching: {c + 1}/{stop}", end='\r')
            ARP_dis, ARP_top_indices = ARP_cdtree.query(ARP_coord, k=top)
            CMM_dis, CMM_top_indices = CMM_cdtree.query(ARP_coord, k=top)

            ARP_filtered = ARP.loc[ARP_top_indices]
            CMM_filtered = CMM.loc[CMM_top_indices]

            offset_vector = [0, 0]
            last_NIoU_mean, last_tar_niou, last_niou_list = Mean_NIoU(ARP_filtered, CMM_filtered, top)

            last_offset_index = [-1,-1]
            
            for i in ARP_top_indices:
                cx = ARP_filtered.loc[i].geometry.centroid.x
                cy = ARP_filtered.loc[i].geometry.centroid.y

                for j in CMM_top_indices:
                    # Calculate offset
                    x_offset = cx - CMM_filtered.loc[j].geometry.centroid.x
                    y_offset = cy - CMM_filtered.loc[j].geometry.centroid.y

                    offset_dist = np.sqrt((x_offset) ** 2 + (y_offset) ** 2)

                    if offset_dist > dist_max:
                        continue

                    CMM_offset = deepcopy(CMM_filtered)

                    # Extract bounding rectangle coordinates
                    pointList = []
                    for polygon in CMM_offset.geometry:
                        pointList.append(list(polygon.exterior.coords))

                    # Rectify coordinates according to offset vector
                    for h in range(0, len(pointList)):
                        points = []
                        for point_element in pointList[h]:
                            points.append(Point(np.array(point_element) +
                                                np.array([x_offset, y_offset])))
                        poly = Polygon([[p.x, p.y] for p in points])

                        h_id = list(CMM_offset.id)[h]
                        CMM_offset.geometry[h_id] = poly  # replace the original geographic attributes

                    NIoU_mean, tar_niou, niou_list = Mean_NIoU(ARP_filtered, CMM_offset, top)

                    if NIoU_mean > last_NIoU_mean:
                        offset_vector = [x_offset, y_offset]
                        last_offset_index = [i, j]
                        last_NIoU_mean = NIoU_mean
                        last_niou_list = niou_list
                        last_tar_niou = tar_niou

            final_offset[c] = offset_vector
            final_offset_index_list[c] = last_offset_index
            NIoU_mean_list[c] = last_NIoU_mean
            niou_list_list[c] = last_niou_list
            tar_niou_list[c] = last_tar_niou

    ARP_rectified['x_offset'] = [row[0] for row in final_offset]
    ARP_rectified['y_offset'] = [row[1] for row in final_offset]
    ARP_rectified['dis'] = [np.sqrt(row[0] * row[0] + row[1] * row[1]) for row in final_offset]
    ARP_rectified['NIoU'] = [row for row in NIoU_mean_list]
    ARP_rectified['tar_niou'] = [row for row in tar_niou_list]
    ARP_rectified['niou_list'] = ['_'.join([str(num)[:5] for num in row]) for row in niou_list_list]
    ARP_rectified['i_j'] = ['_'.join([str(num) for num in row]) for row in final_offset_index_list]

    print("preliminary matching finished")
    print("\033[1;35m-\033[0m" * 40)

    gdf_filtered = ARP_rectified[(ARP_rectified.x_offset == 0) & (ARP_rectified.y_offset == 0)]
    coords_filtered = np.array([list(item) for item in
                                gdf_filtered.geometry.centroid.apply(lambda point: (point.x, point.y))])

    for c, coord in enumerate(coords_filtered):
        print(f"post matching: {c + 1}/{len(coords_filtered)}", end='\r')
        while True:
            dis, top_indices = ARP_cdtree.query(coord, k)

            gdf_top = ARP_rectified.loc[top_indices]

            gdf_top_valid = gdf_top[~((gdf_top.x_offset == 0) & (gdf_top.y_offset == 0))]

            # Calculate the mode of x_offset and y_offset simultaneously
            mode_counts = gdf_top_valid.groupby(['x_offset', 'y_offset', 'dis']).size().reset_index(name='count')
            max_count = mode_counts['count'].max()
            filtered_modes = mode_counts[mode_counts['count'] == max_count]
            filtered_modes = filtered_modes.sort_values('dis')
            if len(filtered_modes['count']) == 0:
                k += 5
                continue
            else:
                break

        ARP_rectified.loc[gdf_filtered.id.iloc[c], 'x_offset'] = round(filtered_modes.x_offset.values[0], 2)
        ARP_rectified.loc[gdf_filtered.id.iloc[c], 'y_offset'] = round(filtered_modes.y_offset.values[0], 2)

    print("post matching finished")
    print("\033[1;35m-\033[0m" * 40)

    # Rectification
    for c, row in ARP_rectified.iterrows():
        print(f"rectification: {c + 1}/{stop}", end='\r')
        offset = [row.x_offset, row.y_offset]
        coordinates = list(ARP_rectified.geometry[c].exterior.coords)
        points = []
        for coord in coordinates:
            points.append(Point(np.array(coord) -
                                np.array([offset[0], offset[1]])))
        poly = Polygon([[p.x, p.y] for p in points])
        ARP_rectified.geometry[c] = poly

    print("rectification finished")
    print("\033[1;35m-\033[0m" * 40)

    # Export rectified file
    output_path = wrap_path.replace('.geojson', f'_rectified.geojson')
    print(f'output_path:{output_path}')
    print("\033[1;33m=\033[0m" * (len(output_path) + 20))
    ARP_rectified.to_file(output_path)
